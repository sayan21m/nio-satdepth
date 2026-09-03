#!/usr/bin/env python3
"""Run ViT inference for one day from train_daily_JFM_2015_2024."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xarray as xr
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "train_daily_JFM_2015_2024"
CKPT_DIR = ROOT / "ml" / "checkpoints"
RUN_TAG = "jfm_2015_2024"
CHANNELS = ["sst", "sss", "sla", "adt", "uo", "vo", "u10", "v10"]


class PatchEmbed(layers.Layer):
    def __init__(self, patch_size, embed_dim, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid",
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
            num_heads=num_heads, key_dim=max(embed_dim // num_heads, 1), dropout=dropout,
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
        self, in_ch=8, out_ch=15, patch_size=10, embed_dim=128, depth=4,
        num_heads=4, mlp_dim=256, dropout=0.1, l2=1e-4, max_tokens=512, **kwargs,
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


def main(date: str) -> None:
    norm_path = CKPT_DIR / f"vit_{RUN_TAG}_norm.npz"
    weights_path = CKPT_DIR / f"vit_{RUN_TAG}_best.weights.h5"
    norm = np.load(norm_path)
    x_mean, x_std = norm["x_mean"], norm["x_std"]
    y_mean, y_std = norm["y_mean"], norm["y_std"]
    depths = norm["depths"]
    l2 = float(norm["l2"])

    surface = xr.open_dataset(DATA / "surface.nc", engine="h5netcdf")
    target = xr.open_dataset(DATA / "target.nc", engine="h5netcdf")

    times = surface.time.values.astype("datetime64[D]")
    matches = np.where(times.astype(str) == date)[0]
    if matches.size == 0:
        march = [str(times[i])[:10] for i in range(len(times)) if str(times[i]).startswith("2024-03")]
        raise SystemExit(f"Date {date} not in cube. March 2024: {march}")
    day_idx = int(matches[0])

    x = np.stack([surface[name].values[day_idx] for name in CHANNELS], axis=-1).astype(np.float32)
    y = np.moveaxis(target.thetao.values[day_idx].astype(np.float32), 0, -1)
    x_ok = np.all(np.isfinite(x), axis=-1)
    y_ok = np.isfinite(y)
    mask = (y_ok & x_ok[..., None]).astype(np.float32)

    x_n = np.nan_to_num((x - x_mean) / x_std, nan=0.0).astype(np.float32)

    model = SmallViT(in_ch=8, out_ch=15, l2=l2)
    _ = model(x_n[None], training=False)
    model.load_weights(weights_path)

    pred_n = model(x_n[None], training=False).numpy()[0]
    pred_c = pred_n * y_std + y_mean
    true_c = y

    diff2 = np.where(mask, (pred_c - true_c) ** 2, 0.0)
    rmse_by_depth = np.sqrt(diff2.sum(axis=(0, 1)) / np.maximum(mask.sum(axis=(0, 1)), 1.0))
    overall_rmse = float(np.sqrt(diff2.sum() / np.maximum(mask.sum(), 1.0)))

    out_dir = ROOT / "outputs" / "vit" / f"predict_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lon = surface.longitude.values
    lat = surface.latitude.values

    print(f"ViT prediction — {date}  (index {day_idx})")
    print(f"Weights: {weights_path.name}")
    print("RMSE (°C) by depth:")
    for d, r in zip(depths, rmse_by_depth):
        print(f"  {float(d):6.0f} m  {r:.3f} °C")
    print(f"overall RMSE: {overall_rmse:.3f} °C")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, d in enumerate([0.0, 100.0]):
        di = int(np.where(depths == d)[0][0])
        truth = np.where(mask[..., di], true_c[..., di], np.nan)
        pred = np.where(mask[..., di], pred_c[..., di], np.nan)
        err = pred - truth
        for ax, field, title in zip(
            axes[row],
            [truth, pred, err],
            [f"true @ {d:.0f} m", f"pred @ {d:.0f} m", f"pred−true @ {d:.0f} m"],
        ):
            if "pred−true" in title:
                vmax = np.nanpercentile(np.abs(err), 95)
                im = ax.pcolormesh(lon, lat, field, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            else:
                im = ax.pcolormesh(lon, lat, field, shading="auto", cmap="turbo")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"ViT — {date}")
    fig.savefig(out_dir / "maps_surface_100m.png", dpi=140)
    plt.close(fig)

    lat0, lon0 = 15.0, 65.0
    j = int(np.argmin(np.abs(lat - lat0)))
    i = int(np.argmin(np.abs(lon - lon0)))
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(true_c[j, i, :], depths, marker="o", label="true")
    ax.plot(pred_c[j, i, :], depths, marker="s", label="ViT pred")
    ax.invert_yaxis()
    ax.set_xlabel("thetao (°C)")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"Profile @ {lat[j]:.2f}°N, {lon[i]:.2f}°E — {date}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "profile_15N_65E.png", dpi=140)
    plt.close(fig)

    np.savez(
        out_dir / f"pred_{date}.npz",
        date=date,
        pred=pred_c.astype(np.float32),
        true=true_c.astype(np.float32),
        mask=mask,
        depths=depths,
        lat=lat,
        lon=lon,
        rmse_by_depth=rmse_by_depth.astype(np.float32),
        overall_rmse=np.float32(overall_rmse),
    )
    print(f"Saved → {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2024-03-15", help="YYYY-MM-DD in JFM cube")
    main(p.parse_args().date)
