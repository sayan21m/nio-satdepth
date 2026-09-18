#!/usr/bin/env python3
"""ConvGRU RNN — 3-day satellite surface lags → thetao (15 depths).

Satellite-only (no GLORYS θ lag). Fair temporal baseline vs ViT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DATA = ROOT / "data" / "processed" / "train_daily_2015_2024"
RUN_TAG = "full_2015_2024_sat_rnn"
CKPT_DIR = ROOT / "ml" / "checkpoints"
OUT = ROOT / "outputs" / "rnn_sat_lag"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SURFACE_CHANNELS = ["sst", "sss", "sla", "adt", "uo", "vo", "u10", "v10"]
LAG_OFFSETS = [2, 1, 0]
MAX_LAG = max(LAG_OFFSETS)
N_IN = len(SURFACE_CHANNELS)
L2 = 1e-4
WIDTH = 32
BATCH = 2
EPOCHS = 40
PATIENCE = 6
LR = 5e-4


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hidden: int, k: int = 3):
        super().__init__()
        pad = k // 2
        self.hidden = hidden
        self.gates = nn.Conv2d(in_ch + hidden, 2 * hidden, k, padding=pad)
        self.cand = nn.Conv2d(in_ch + hidden, hidden, k, padding=pad)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        z, r = self.gates(torch.cat([x, h], dim=1)).chunk(2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        h_t = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_t


class ConvGRU2d(nn.Module):
    def __init__(self, in_ch: int, hidden: int, k: int = 3):
        super().__init__()
        self.cell = ConvGRUCell(in_ch, hidden, k)
        self.hidden = hidden

    def forward(self, x, h=None):
        # x: (B, T, C, H, W)
        hs = []
        for ti in range(x.shape[1]):
            h = self.cell(x[:, ti], h)
            hs.append(h)
        return torch.stack(hs, dim=1), h


class SatLagRNN(nn.Module):
    """(B, T, 8, H, W) → (B, 15, H, W) via ConvGRU."""

    def __init__(self, in_ch: int = 8, out_ch: int = 15, hidden: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gru0 = ConvGRU2d(hidden, hidden)
        self.bn0 = nn.BatchNorm2d(hidden)
        self.gru1 = ConvGRU2d(hidden, hidden)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.refine = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, out_ch, 1)

    def _bn_seq(self, bn, x):
        b, t, c, h, w = x.shape
        return bn(x.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)

    def forward(self, x):
        b, t, c, h, w = x.shape
        e = self.enc(x.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w)
        e, _ = self.gru0(e)
        e = self._bn_seq(self.bn0, e)
        _, h_last = self.gru1(e)
        h_last = torch.relu(self.refine(h_last))
        return self.head(h_last)


def channel_mean_std(arr: np.ndarray):
    c = arr.shape[-1]
    mean = np.zeros(c, dtype=np.float32)
    std = np.ones(c, dtype=np.float32)
    for i in range(c):
        vals = arr[..., i][np.isfinite(arr[..., i])]
        if vals.size:
            mean[i] = vals.mean()
            s = vals.std()
            std[i] = s if s > 1e-6 else 1.0
    return mean, std


def valid_lag_indices(indices, times):
    ok = []
    for t in indices:
        if t < MAX_LAG:
            continue
        good = True
        for lag in LAG_OFFSETS:
            if lag > 0:
                delta = (times[t] - times[t - lag]) / np.timedelta64(1, "D")
                if int(delta) != lag:
                    good = False
                    break
        if good:
            ok.append(int(t))
    return np.array(ok, dtype=np.int32)


def masked_mse(y_pred, y_true, mask):
    return ((y_pred - y_true) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def ridge_loss(model, l2):
    if l2 <= 0:
        return torch.zeros((), device=DEVICE)
    return l2 * sum(p.square().sum() for p in model.parameters())


def main():
    print("device:", DEVICE)
    print("DATA:", DATA)

    surface = xr.open_dataset(DATA / "surface.nc", engine="h5netcdf")
    target = xr.open_dataset(DATA / "target.nc", engine="h5netcdf")
    depth_vals = target.depth.values.astype(np.float32)
    n_out = len(depth_vals)
    times = surface.time.values.astype("datetime64[D]")
    n_days = len(times)
    n_train = int(n_days * 0.8)
    train_idx = valid_lag_indices(np.arange(0, n_train), times)
    val_idx = valid_lag_indices(np.arange(n_train, n_days), times)
    print(f"days={n_days} train={len(train_idx)} val={len(val_idx)}")

    x_surface = np.stack([surface[name].values for name in SURFACE_CHANNELS], axis=-1).astype(np.float32)
    y_all = np.moveaxis(target.thetao.values.astype(np.float32), 1, -1)
    surf_mean, surf_std = channel_mean_std(x_surface[train_idx])
    y_mean, y_std = channel_mean_std(y_all[train_idx])
    h, w = x_surface.shape[1], x_surface.shape[2]

    def make_example(t: int):
        frames = np.stack([x_surface[t - lag] for lag in LAG_OFFSETS], axis=0).astype(np.float32)
        y = y_all[t].copy()
        x_ok = np.all(np.isfinite(frames), axis=(0, 3))
        y_ok = np.isfinite(y)
        mask = (y_ok & x_ok[..., None]).astype(np.float32)
        frames = np.nan_to_num((frames - surf_mean) / surf_std, nan=0.0).astype(np.float32)
        y = np.nan_to_num((y - y_mean) / y_std, nan=0.0).astype(np.float32)
        frames = np.moveaxis(frames, -1, 1)
        y = np.moveaxis(y, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        return frames, y, mask

    class DS(Dataset):
        def __init__(self, idx):
            self.idx = np.asarray(idx, dtype=np.int32)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            x, y, m = make_example(int(self.idx[i]))
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)

    train_loader = DataLoader(DS(train_idx), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(DS(val_idx), batch_size=BATCH, shuffle=False, num_workers=0)

    model = SatLagRNN(N_IN, n_out, hidden=WIDTH).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    best_path = CKPT_DIR / f"rnn_sat_lag_{RUN_TAG}_w{WIDTH}_best.pt"
    norm_path = CKPT_DIR / f"rnn_sat_lag_{RUN_TAG}_w{WIDTH}_norm.npz"
    np.savez(
        norm_path,
        surf_mean=surf_mean,
        surf_std=surf_std,
        y_mean=y_mean,
        y_std=y_std,
        surface_channels=np.array(SURFACE_CHANNELS),
        all_depths=depth_vals,
        lag_offsets=np.array(LAG_OFFSETS, dtype=np.int32),
        width=np.int32(WIDTH),
        l2=np.float32(L2),
    )

    history = {"loss": [], "val_loss": []}
    best_val = float("inf")
    wait = 0

    def run_epoch(loader, training: bool):
        model.train(training)
        tot = n = 0.0
        for x, y, m in loader:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            if training:
                opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = masked_mse(pred, y, m) + ridge_loss(model, L2)
            if training:
                loss.backward()
                opt.step()
            bs = x.shape[0]
            tot += float(loss.detach()) * bs
            n += bs
        return tot / max(n, 1.0)

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(train_loader, True)
        with torch.no_grad():
            va = run_epoch(val_loader, False)
        history["loss"].append(tr)
        history["val_loss"].append(va)
        star = ""
        if va < best_val - 1e-5:
            best_val = va
            wait = 0
            star = " *best*"
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        print(f"epoch {epoch:02d}  train={tr:.4f}  val={va:.4f}  wait={wait}/{PATIENCE}{star}", flush=True)
        if wait >= PATIENCE:
            print("early stop", flush=True)
            break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))

    # RMSE in °C
    sse = np.zeros(n_out, dtype=np.float64)
    count = np.zeros(n_out, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for x, y_n, m in val_loader:
            pred_n = model(x.to(DEVICE)).cpu().numpy()
            y_n = y_n.numpy()
            m = m.numpy()
            pred = pred_n * y_std[:, None, None] + y_mean[:, None, None]
            y = y_n * y_std[:, None, None] + y_mean[:, None, None]
            diff2 = np.where(m, (pred - y) ** 2, 0.0)
            sse += diff2.sum(axis=(0, 2, 3))
            count += m.sum(axis=(0, 2, 3))

    rmse = np.sqrt(sse / np.maximum(count, 1.0))
    overall = float(np.sqrt(sse.sum() / count.sum()))
    print("\nValidation RMSE (°C) by depth:")
    for d, r in zip(depth_vals, rmse):
        print(f"  {float(d):6.0f} m  {r:.3f} °C")
    print(f"mean RMSE: {rmse.mean():.3f} °C")
    print(f"overall RMSE: {overall:.3f} °C")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_title("ConvGRU sat-lag RNN — loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "01_loss_curve.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rmse, depth_vals, marker="o")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (°C)")
    ax.set_ylabel("depth (m)")
    ax.set_title("ConvGRU sat-lag validation RMSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "02_rmse_by_depth.png", dpi=140)
    plt.close()

    np.savez(OUT / "metrics.npz", depths=depth_vals, rmse=rmse, overall=overall, mean=float(rmse.mean()))
    print("saved:", best_path)
    print("figures:", OUT)
    surface.close()
    target.close()


if __name__ == "__main__":
    main()
