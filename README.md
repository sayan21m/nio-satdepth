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
| **Inputs (X)** | `sst`, `sss`, `sla`, `adt`, `uo`, `vo`, `u10`, `v10` | `data/processed/train_daily_JFM_2015_2024/surface.nc` |
| **Labels (Y)** | `thetao` (15 depths) | `data/processed/train_daily_JFM_2015_2024/target.nc` |

**Current set:** Jan–Mar (JFM) **2015–2024** — **903** daily samples (same season, multi-year).

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
├── README.md
├── requirements.txt
├── src/                      ← data download only
│   ├── download_pipeline.py
│   └── extend_dataset.py
├── ml/
│   ├── configs/              ← EDA + train ViT / ConvLSTM
│   └── checkpoints/
├── data/
│   └── processed/
│       └── train_daily_JFM_2015_2024/
├── outputs/
└── web/                      ← demo page (both models)
```

Training models live **inside** the notebooks (no `ml/models/*.py` for now).

### Demo webpage

```bash
cd web
python3 -m http.server 8080
# open http://localhost:8080
```

ViT vs ConvLSTM, RMSE table, and result galleries.

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
2. Log in once:

```bash
copernicusmarine login
```

Credentials go under `~/.copernicusmarine/` (not in this repo).

---

## Use the existing dataset

```python
import xarray as xr

surface = xr.open_dataset("data/processed/train_daily_JFM_2015_2024/surface.nc")
target  = xr.open_dataset("data/processed/train_daily_JFM_2015_2024/target.nc")

print(surface)  # time × lat × lon
print(target)   # time × depth × lat × lon
```

- **X** = surface fields each day  
- **Y** = `thetao` at 15 depths  

Notebooks: `ml/configs/eda.ipynb`, `train_vit.ipynb`, `train_convlstm_lag.ipynb`.

---

## Add more data (JFM 2021–2025)

```bash
# Example: 2021 Jan–Mar
python src/download_pipeline.py \
  --start-date 2021-01-01 \
  --end-date 2021-03-31 \
  --chunk quarter \
  --out-name train_daily_2021JFM
```

Repeat for 2022…2025, then merge folders into `train_daily_JFM_2019_2025`.

| Flag | Meaning |
|------|---------|
| `--start-date YYYY-MM-DD` | Inclusive start |
| `--end-date YYYY-MM-DD` | Inclusive end |
| `--chunk quarter\|month\|all` | Split size (default: `quarter`) |
| `--out-name NAME` | Folder under `data/processed/` |
| `--dry-run` | Print planned chunks only |
| `--merge-only` | Merge existing chunks; no download |
| `--no-merge` | Keep chunks only; skip final merge |

---

## Data sources (CMEMS)

| Field | Product family |
|-------|----------------|
| Subsurface T (labels) | GLORYS |
| SST | OSTIA L4 |
| SSS | Multi-observation surface salinity |
| SLA / ADT | DUACS altimetry L4 |
| Currents | Multi-observation surface currents |
| Winds | Scatterometer + model L4 |

---

## Expected solution path (SIH)

1. **Preprocessing** — `download_pipeline.py`  
2. **Embedding** — ViT / ConvLSTM in notebooks  
3. **Reconstruction** — surface → `thetao` at depths  
4. **Validation** — RMSE (°C), correlation, % within 1 °C  
5. **PoC demo** — Bay of Bengal / Arabian Sea maps and profiles (`web/`)  

---

## Quick start

1. `pip install -r requirements.txt`  
2. `copernicusmarine login`  
3. Open `data/processed/train_daily_JFM_2015_2024/`  
4. Train via `ml/configs/train_vit.ipynb` or `train_convlstm_lag.ipynb`  
5. Add JFM years with `src/download_pipeline.py`  

Do not commit `.nc` files or credentials (see `.gitignore`).
