# SIH PS 26066 — Satellite Embedding Subsurface Temperature

Reconstruct **depth-wise subsurface ocean temperature** over the North Indian Ocean from daily surface satellite observations using a deep learning pipeline.

**Problem focus:** learn a mapping from surface fields → temperature profiles at standard depths, using satellite embeddings / DL models.

---

## Domain & resolution

| Setting | Value |
|---------|--------|
| Region | `5°N–30°N`, `45°E–105°E` |
| Spatial grid | **0.25° × 0.25°** |
| Temporal | **Daily** |
| Target depths (m) | `0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000` |

---

## What you train on

| Role | Variables | File |
|------|-----------|------|
| **Inputs (X)** | `sst`, `sss`, `sla`, `adt`, `uo`, `vo`, `u10`, `v10` | `data/processed/train_daily_2019Q1/surface.nc` |
| **Labels (Y)** | `thetao` (15 depths) | `data/processed/train_daily_2019Q1/target.nc` |

Current ready split: **2019-01-01 → 2019-03-31** (90 daily samples).

| Variable | Meaning |
|----------|---------|
| `sst` | Sea surface temperature (°C) |
| `sss` | Sea surface salinity |
| `sla` | Sea level anomaly (m) |
| `adt` | Absolute dynamic topography (m) |
| `uo`, `vo` | Surface currents |
| `u10`, `v10` | Surface wind components (daily mean) |
| `thetao` | Potential temperature vs depth (°C) |

---

## Project layout

```
SIH_PS_26066/
├── README.md                 ← you are here
├── requirements.txt
├── src/
│   ├── download_pipeline.py  ← data CLI (add / download)
│   └── extend_dataset.py     ← download + harmonize engine
├── ml/                       ← training / models / checkpoints
│   ├── configs/
│   ├── scripts/
│   ├── models/
│   └── checkpoints/
├── data/
│   ├── README.md
│   └── processed/
│       ├── train_daily_2019Q1/   ← use for training
│       │   ├── surface.nc
│       │   └── target.nc
│       └── chunks/               ← daily chunk cache (resume/skip)
├── outputs/                  ← figures / analysis artifacts
└── scripts/                  ← optional data helpers
```

---

## Setup

### 1. Python environment

```bash
cd SIH_PS_26066
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Copernicus Marine account

1. Create a free account: [https://marine.copernicus.eu/](https://marine.copernicus.eu/)
2. Log in once with the toolbox:

```bash
copernicusmarine login
```

Enter your username (email) and password when prompted.  
Credentials are stored under `~/.copernicusmarine/` (not in this repo).

---

## Use the existing dataset

```python
import xarray as xr

surface = xr.open_dataset("data/processed/train_daily_2019Q1/surface.nc")
target  = xr.open_dataset("data/processed/train_daily_2019Q1/target.nc")

print(surface)  # time × lat × lon × features
print(target)   # time × depth × lat × lon
```

Supervised setup:

- **X** = surface fields on each day / grid cell (or patches for CNN/ViT)
- **Y** = `thetao` profile at the same location / day

---

## Add more data to the dataset

All new data goes through **`src/download_pipeline.py`**.

### Preview chunks (no download)

```bash
python src/download_pipeline.py \
  --start-date 2019-04-01 \
  --end-date 2019-06-30 \
  --dry-run
```

### Download + preprocess + merge

```bash
python src/download_pipeline.py \
  --start-date 2019-04-01 \
  --end-date 2019-06-30 \
  --chunk quarter \
  --out-name train_daily_2019Q2
```

This will:

1. Download multi-source CMEMS products for the range  
2. Harmonize to **0.25° daily**  
3. Write pieces under `data/processed/chunks/`  
4. Merge into `data/processed/<out-name>/surface.nc` and `target.nc`  
5. Delete temporary raw downloads after each chunk  

Already-finished chunks (e.g. Jan–Mar 2019) are **skipped** automatically.

### Merge several periods into one train set

```bash
# After Q1 and Q2 chunks exist:
python src/download_pipeline.py \
  --start-date 2019-01-01 \
  --end-date 2019-06-30 \
  --merge-only \
  --out-name train_daily_2019H1
```

### CLI options

| Flag | Meaning |
|------|---------|
| `--start-date YYYY-MM-DD` | Inclusive start (**required**) |
| `--end-date YYYY-MM-DD` | Inclusive end (**required**) |
| `--chunk quarter\|month\|all` | Split size (default: `quarter`) |
| `--out-name NAME` | Folder under `data/processed/` |
| `--dry-run` | Print planned chunks only |
| `--merge-only` | Merge existing chunks; no download |
| `--no-merge` | Keep chunks only; skip final merge |

### Recommended ranges for this SIH PoC

| Split | Suggested range | Notes |
|-------|-----------------|--------|
| Train | e.g. 2019–2020 | Expand with the pipeline as disk/time allow |
| Test / holdout | e.g. recent month | GLORYS MY labels lag; ~June 2026 was available earlier, **July 2026 may not be** |

---

## Data sources (CMEMS)

| Field | Product family |
|-------|----------------|
| Subsurface T (labels) | GLORYS (`GLOBAL_MULTIYEAR_PHY` / `cmems_mod_glo_phy_my_…`) |
| SST | OSTIA L4 |
| SSS | Multi-observation surface salinity |
| SLA / ADT | DUACS altimetry L4 |
| Currents | Multi-observation surface currents |
| Winds | Scatterometer + model L4 (MY / NRT by date) |

If a product is missing at the requested resolution, the pipeline regrids as needed (allowed by the problem statement).

---

## Folder meanings

| Path | Meaning |
|------|---------|
| `data/processed/train_daily_*/` | Merged **daily** cubes for ML (**use these**) |
| `data/processed/chunks/` | **Daily** date-range pieces for resume/skip |
| `data/raw_tmp/` | Temporary raw downloads (auto-deleted per chunk) |

Both `train_daily_*` and `chunks` are **daily** — not yearly averages.

---

## Expected solution path (SIH)

1. **Preprocessing** — done via `download_pipeline.py`  
2. **Satellite embedding** — CNN / ViT / autoencoder / GNN / attention on surface fields  
3. **Reconstruction head** — map embeddings → `thetao` at standard depths  
4. **Validation** — RMSE, correlation, bias vs held-out days / independent ARGO (INCOIS LAS)  
5. **PoC demo** — Bay of Bengal / Arabian Sea maps and profiles  

---

## Tips & troubleshooting

- **Disk / time:** each quarter of NIO GLORYS+winds is large; prefer `--chunk quarter` or `month`.  
- **Login errors:** run `copernicusmarine login` again.  
- **Failed mid-download:** re-run the same command; completed chunks in `chunks/` are skipped.  
- **OOM during harmonize:** already mitigated with batched regridding; close other heavy apps if needed.  
- **Do not commit** `.nc` files or credentials (see `.gitignore`).

---

## Quick start checklist

1. `pip install -r requirements.txt`  
2. `copernicusmarine login`  
3. Open `data/processed/train_daily_2019Q1/surface.nc` + `target.nc`  
4. Build / train your embedding + reconstruction model  
5. Add more dates with `python src/download_pipeline.py --start-date … --end-date …`
