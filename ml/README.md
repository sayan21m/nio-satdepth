# ML workspace

Put training, models, and experiment configs here.

```
ml/
├── configs/       # hyperparameters, experiment YAML/JSON
├── scripts/       # train / eval / inference scripts
├── models/        # model definitions (CNN, ViT, etc.)
└── checkpoints/   # saved weights (.pt / .ckpt)
```

## Data paths (do not move data here)

- Inputs: `data/processed/train_daily_2019Q1/surface.nc`
- Labels: `data/processed/train_daily_2019Q1/target.nc`

## Suggested next scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Train embedding + reconstruction model |
| `scripts/eval.py` | RMSE / correlation / bias on holdout |
| `scripts/infer.py` | Run on new surface fields → depth profiles |

Example train entry (to implement):

```bash
python ml/scripts/train.py \
  --surface data/processed/train_daily_2019Q1/surface.nc \
  --target data/processed/train_daily_2019Q1/target.nc \
  --config ml/configs/default.yaml \
  --out ml/checkpoints/
```
