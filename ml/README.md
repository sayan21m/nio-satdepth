# ML workspace

Notebook-first training. Model code lives **inside** the notebooks.

```
ml/
├── configs/
│   ├── eda.ipynb                 # EDA
│   ├── train_vit.ipynb           # Vision Transformer (TF) — SIH baseline
│   └── train_convlstm_lag.ipynb  # ConvLSTM (PyTorch/MPS) — 3-day lags + shallow θ
└── checkpoints/                  # saved weights + norm stats
```

## Data

- Inputs: `data/processed/train_daily_JFM_2015_2024/surface.nc`
- Labels: `data/processed/train_daily_JFM_2015_2024/target.nc`

Season-matched **Jan–Mar** days for **2015–2024** (903 days).

## Models

| Notebook | Framework | Role |
|----------|-----------|------|
| `train_vit.ipynb` | TensorFlow | Surface → 15 depths (satellite-only) |
| `train_convlstm_lag.ipynb` | PyTorch / MPS | 3-day surface + lag shallow θ; use **closed-loop** §6 for daily ops |

## Apple GPU

- **ConvLSTM** uses **PyTorch MPS**. Install `torch` in the kernel env.
- **ViT** uses TensorFlow. `tensorflow-metal` is optional and version-sensitive.

## Demo site

Open `web/` in a browser (local server):

```bash
cd web && python3 -m http.server 8080
```
