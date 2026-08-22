# SIH PS 26066 — data

Region: North Indian Ocean `5°N–30°N`, `45°E–105°E` · **0.25° daily**

## Ready for training (Jan–Mar 2019, daily)

| Path | Role |
|------|------|
| `processed/train_daily_2019Q1/surface.nc` | Inputs X — `sst,sss,sla,adt,uo,vo,u10,v10` |
| `processed/train_daily_2019Q1/target.nc` | Labels Y — `thetao` (15 depths) |

## Folder meanings

| Folder | Meaning |
|--------|---------|
| `processed/train_daily_2019Q1/` | Merged **daily** train split (use this) |
| `processed/chunks/` | **Daily** chunk cache (quarter/month pieces for resume/skip) |

Both are daily — nothing here is yearly averages.

## Download more with date range

```bash
python src/download_pipeline.py --start-date 2019-04-01 --end-date 2019-06-30 --dry-run

python src/download_pipeline.py \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --chunk quarter \
  --out-name train_daily_2019Q2
```

Flags: `--chunk quarter|month|all`, `--merge-only`, `--no-merge`, `--dry-run`
