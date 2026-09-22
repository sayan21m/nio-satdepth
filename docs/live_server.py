#!/usr/bin/env python3
"""Live surface + ViT prediction API for the OceanEmbed web demo.

Serves filled-grid JSON for colormap maps (surface channels + predicted θ).
Advances through the training cube on a timer so the UI looks continuous.

  cd docs && python live_server.py
  open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import xarray as xr
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
# Site files live in web/ (index, css, videos). docs/ only has the Flask server.
WEB = ROOT / "web"
DATA = ROOT / "data" / "processed" / "train_daily_2015_2024"
CKPT_DIR = ROOT / "ml" / "checkpoints"
CHANNELS = ["sst", "sss", "sla", "adt", "uo", "vo", "u10", "v10"]
PRED_DEPTHS_M = [0.0, 50.0, 100.0, 150.0]
ADVANCE_SEC = 20  # seconds between day advances

# Prefer newest available ViT weights
for _tag in ("full_2015_2024", "jfmamjjas_2015_2024", "jfmamj_2015_2024", "jfm_2015_2024"):
    _w = CKPT_DIR / f"vit_{_tag}_best.weights.h5"
    _n = CKPT_DIR / f"vit_{_tag}_norm.npz"
    if _w.exists() and _n.exists():
        RUN_TAG = _tag
        WEIGHTS = _w
        NORM_PATH = _n
        break
else:
    RUN_TAG = WEIGHTS = NORM_PATH = None

# Do not mount Flask's built-in static at "". It shadows GET / and 404s on Render.
app = Flask(__name__, static_folder=None)

_state = {
    "ready": False,
    "error": None,
    "idx": 0,
    "n": 0,
    "times": None,
    "lat": None,
    "lon": None,
    "depths": None,
    "surface": None,
    "target": None,
    "model": None,
    "x_mean": None,
    "x_std": None,
    "y_mean": None,
    "y_std": None,
    "pred_cache": {},  # idx -> pred maps
    "lock": threading.Lock(),
    "demo": False,
    "demo_days": None,
}


def _hide_tf_metal():
    """Avoid broken tensorflow-metal plugin crashes on import."""
    for p in sys.path:
        plug = Path(p) / "tensorflow-plugins"
        if plug.is_dir() and (plug / "libmetal_plugin.dylib").exists():
            off = plug.with_name("tensorflow-plugins.off")
            if not off.exists():
                plug.rename(off)


def _load_vit():
    if WEIGHTS is None:
        return None
    _hide_tf_metal()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    class PatchEmbed(layers.Layer):
        def __init__(self, patch_size, embed_dim, l2=0.0, **kwargs):
            super().__init__(**kwargs)
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.proj = layers.Conv2D(
                embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid",
                kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None,
            )

        def call(self, x):
            x = self.proj(x)
            b = tf.shape(x)[0]
            return tf.reshape(x, (b, -1, self.embed_dim))

    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, l2=0.0, **kwargs):
            super().__init__(**kwargs)
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.attn = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=max(embed_dim // num_heads, 1),
                dropout=dropout,
                kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None,
            )
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            reg = regularizers.l2(l2) if l2 > 0 else None
            self.fc1 = layers.Dense(mlp_dim, activation="gelu", kernel_regularizer=reg)
            self.drop1 = layers.Dropout(dropout)
            self.fc2 = layers.Dense(embed_dim, kernel_regularizer=reg)
            self.drop2 = layers.Dropout(dropout)

        def call(self, x, training=None):
            y = self.attn(self.norm1(x), self.norm1(x), training=training)
            x = x + y
            y = self.fc1(self.norm2(x))
            y = self.drop1(y, training=training)
            y = self.fc2(y)
            y = self.drop2(y, training=training)
            return x + y

    class SmallViT(keras.Model):
        def __init__(
            self,
            in_ch=8,
            out_ch=15,
            patch_size=10,
            embed_dim=128,
            depth=4,
            num_heads=4,
            mlp_dim=256,
            dropout=0.1,
            l2=1e-4,
            max_tokens=512,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.out_ch = out_ch
            self.patch_size = patch_size
            self.patch_embed = PatchEmbed(patch_size, embed_dim, l2=l2)
            self.pos_embed = layers.Embedding(max_tokens, embed_dim)
            self.blocks = [
                TransformerBlock(embed_dim, num_heads, mlp_dim, dropout=dropout, l2=l2)
                for _ in range(depth)
            ]
            self.encoder_norm = layers.LayerNormalization(epsilon=1e-6)
            reg = regularizers.l2(l2) if l2 > 0 else None
            self.head = layers.Dense(patch_size * patch_size * out_ch, kernel_regularizer=reg)
            self.refine = layers.Conv2D(out_ch, 3, padding="same", kernel_regularizer=reg)

        def call(self, inputs, training=None):
            patch = self.patch_size
            h0, w0 = tf.shape(inputs)[1], tf.shape(inputs)[2]
            pad_h = (patch - h0 % patch) % patch
            pad_w = (patch - w0 % patch) % patch
            x = tf.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
            hp, wp = tf.shape(x)[1] // patch, tf.shape(x)[2] // patch
            tokens = self.patch_embed(x)
            n = tf.shape(tokens)[1]
            tokens = tokens + self.pos_embed(tf.range(n))
            for block in self.blocks:
                tokens = block(tokens, training=training)
            tokens = self.encoder_norm(tokens)
            pixels = self.head(tokens)
            pixels = tf.reshape(pixels, (-1, hp, wp, patch, patch, self.out_ch))
            pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5])
            logits = tf.reshape(pixels, (-1, hp * patch, wp * patch, self.out_ch))
            logits = self.refine(logits)
            return logits[:, :h0, :w0, :]

    norm = np.load(NORM_PATH)
    l2 = float(norm["l2"]) if "l2" in norm.files else 1e-4
    model = SmallViT(in_ch=8, out_ch=15, l2=l2)
    # build weights
    model(tf.zeros((1, 100, 240, 8), dtype=tf.float32))
    model.load_weights(str(WEIGHTS))
    _state["x_mean"] = norm["x_mean"].astype(np.float32)
    _state["x_std"] = norm["x_std"].astype(np.float32)
    _state["y_mean"] = norm["y_mean"].astype(np.float32)
    _state["y_std"] = norm["y_std"].astype(np.float32)
    return model


def _demo_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "data" / "live_demo.json",
        ROOT / "docs" / "data" / "live_demo.json",
        WEB / "data" / "live_demo.json",
        Path.cwd() / "data" / "live_demo.json",
    ]


def _load_demo() -> bool:
    if _state.get("demo") and _state.get("ready"):
        return True
    path = next((p for p in _demo_paths() if p.is_file()), None)
    if path is None:
        print("Demo JSON not found. Tried:", *[str(p) for p in _demo_paths()], flush=True)
        return False
    data = json.loads(path.read_text())
    days = data.get("days") or []
    if not days:
        return False
    _state["demo"] = True
    _state["demo_days"] = days
    _state["lat"] = data["lat"]
    _state["lon"] = data["lon"]
    _state["n"] = len(days)
    _state["idx"] = 0
    _state["ready"] = True
    _state["demo_model"] = data.get("model") or "demo"
    print(f"Demo playback · {len(days)} days from {path}", flush=True)
    return True


def _init():
    try:
        surface_path = DATA / "surface.nc"
        if not surface_path.exists():
            if _load_demo():
                return
            print(f"No training cubes at {surface_path} — site-only mode", flush=True)
            return
        print("Loading cubes…", flush=True)
        surface = xr.open_dataset(surface_path, engine="h5netcdf")
        target = xr.open_dataset(DATA / "target.nc", engine="h5netcdf")
        _state["surface"] = surface
        _state["target"] = target
        _state["times"] = surface.time.values.astype("datetime64[D]")
        _state["n"] = len(_state["times"])
        _state["lat"] = surface.latitude.values.astype(np.float32)
        _state["lon"] = surface.longitude.values.astype(np.float32)
        _state["depths"] = target.depth.values.astype(np.float32)
        # start near a known good March day if present
        prefer = np.where(_state["times"].astype(str) == "2024-03-15")[0]
        _state["idx"] = int(prefer[0]) if prefer.size else max(0, _state["n"] // 2)
        print(f"days={_state['n']}  start={_state['times'][_state['idx']]}", flush=True)

        if WEIGHTS is not None:
            print(f"Loading ViT ({RUN_TAG})…", flush=True)
            _state["model"] = _load_vit()
            print("ViT ready", flush=True)
        else:
            print("No ViT weights found — surface-only mode", flush=True)
        _state["ready"] = True
    except Exception as e:
        _state["error"] = str(e)
        print("INIT ERROR:", e, flush=True)


def _advance_loop():
    while True:
        time.sleep(ADVANCE_SEC)
        if not _state["ready"]:
            continue
        with _state["lock"]:
            _state["idx"] = (_state["idx"] + 1) % _state["n"]


def _grid_to_list(arr: np.ndarray) -> list:
    """Replace NaN with null for JSON."""
    out = arr.astype(np.float32)
    # round for smaller payloads
    out = np.round(out, 3)
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in out]


def _predict(day_idx: int) -> dict[str, list]:
    model = _state["model"]
    if model is None:
        return {}
    cached = _state["pred_cache"].get(day_idx)
    if cached is not None:
        return cached
    import tensorflow as tf

    surf = np.stack(
        [_state["surface"][c].isel(time=day_idx).values.astype(np.float32) for c in CHANNELS],
        axis=-1,
    )
    x = (surf - _state["x_mean"]) / _state["x_std"]
    x = np.nan_to_num(x, nan=0.0)[None, ...]
    pred_n = model(tf.constant(x), training=False).numpy()[0]  # H,W,15
    pred = pred_n * _state["y_std"] + _state["y_mean"]
    depths = _state["depths"]
    out = {}
    for d in PRED_DEPTHS_M:
        di = int(np.argmin(np.abs(depths - d)))
        key = f"{int(depths[di])}m"
        out[key] = _grid_to_list(pred[:, :, di])
    # keep cache small
    if len(_state["pred_cache"]) > 8:
        _state["pred_cache"].clear()
    _state["pred_cache"][day_idx] = out
    return out


@app.get("/data/live_demo.json")
def demo_json():
    path = next((p for p in _demo_paths() if p.is_file()), None)
    if path is None:
        return jsonify({"ok": False, "error": "demo missing"}), 404
    return send_from_directory(path.parent, path.name)


@app.get("/api/live")
def api_live():
    if _state["error"]:
        return jsonify({"ok": False, "error": _state["error"]}), 500
    if not _state["ready"]:
        _load_demo()
    if not _state["ready"]:
        return jsonify({"ok": False, "error": "loading"}), 503

    with _state["lock"]:
        idx = int(request.args.get("idx", _state["idx"]))
        idx = max(0, min(idx, _state["n"] - 1))

        if _state.get("demo") and _state.get("demo_days"):
            day = _state["demo_days"][idx]
            return jsonify(
                {
                    "ok": True,
                    "date": day["date"],
                    "idx": idx,
                    "n": _state["n"],
                    "model": _state.get("demo_model") or "demo",
                    "lat": [float(x) for x in _state["lat"]],
                    "lon": [float(x) for x in _state["lon"]],
                    "surface": day["surface"],
                    "pred": day.get("pred") or {},
                    "true": day.get("true") or {},
                    "advance_sec": ADVANCE_SEC,
                }
            )

        date = str(_state["times"][idx])[:10]
        surface_maps = {}
        for c in CHANNELS:
            surface_maps[c] = _grid_to_list(_state["surface"][c].isel(time=idx).values)

        pred_maps = _predict(idx)

        true_maps = {}
        if _state["target"] is not None:
            th = _state["target"].thetao.isel(time=idx).values  # depth, lat, lon
            depths = _state["depths"]
            for d in PRED_DEPTHS_M:
                di = int(np.argmin(np.abs(depths - d)))
                key = f"{int(depths[di])}m"
                true_maps[key] = _grid_to_list(th[di])

        return jsonify(
            {
                "ok": True,
                "date": date,
                "idx": idx,
                "n": _state["n"],
                "model": RUN_TAG,
                "lat": [float(x) for x in _state["lat"]],
                "lon": [float(x) for x in _state["lon"]],
                "surface": surface_maps,
                "pred": pred_maps,
                "true": true_maps,
                "advance_sec": ADVANCE_SEC,
            }
        )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ready": _state["ready"],
            "error": _state["error"],
            "model": RUN_TAG,
            "n": _state["n"],
            "idx": _state["idx"],
        }
    )


@app.route("/")
def index():
    index_path = WEB / "index.html"
    if not index_path.is_file():
        return f"index.html not found at {index_path}", 500
    return send_from_directory(WEB, "index.html")


@app.route("/<path:filename>")
def public_file(filename):
    return send_from_directory(WEB, filename)


def start_background() -> None:
    if getattr(start_background, "_started", False):
        return
    start_background._started = True
    threading.Thread(target=_init, daemon=True).start()
    threading.Thread(target=_advance_loop, daemon=True).start()


print(f"WEB={WEB}  index={'yes' if (WEB / 'index.html').is_file() else 'NO'}", flush=True)
if (DATA / "surface.nc").is_file():
    start_background()
else:
    _load_demo()
    threading.Thread(target=_advance_loop, daemon=True).start()


def main():
    port = int(os.environ.get("PORT", "8765"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    print(f"Open http://{host}:{port}  -> Live Map tab", flush=True)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
