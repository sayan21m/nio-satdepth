# ML workspace

Notebook-first training. Model code lives **inside** the notebooks.

```
ml/
├── configs/
│   ├── eda.ipynb              # EDA
│   ├── train_vit.ipynb        # Vision Transformer — kept SIH model
│   └── compare_models.ipynb   # RMSE leaderboard (kept vs dropped)
└── checkpoints/               # saved weights + norm stats
```

## Data

- Inputs: `data/processed/train_daily_2015_2024/surface.nc`
- Labels: `data/processed/train_daily_2015_2024/target.nc`

Full-year **Jan–Dec** days for **2015–2024** (3653 days).

## Models

| Notebook | Framework | Role | Overall RMSE |
|----------|-----------|------|-------------:|
| `train_vit.ipynb` | TensorFlow | Surface → 15 depths (satellite-only) | **0.671 °C** |
| `compare_models.ipynb` | — | Leaderboard of all tried architectures | — |

Worse architectures (ViT + $L_{\mathrm{grad}}$, thermocline-first, structured embedding, depth-conditioned, probabilistic ViT, TC-band ViT) were trained, compared, and removed. See `compare_models.ipynb`.

## Demo site

```bash
cd web && python3 -m http.server 8080
```
