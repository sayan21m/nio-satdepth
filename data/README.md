# SIH PS 26066 — data

Region: North Indian Ocean `5°N–30°N`, `45°E–105°E` · **0.25° daily**

## Ready for training

| Path | Range | Days |
|------|-------|------|
| **`processed/train_daily_JFM_2015_2024/`** | Jan–Mar 2015–2024 | **903** |

| File | Role |
|------|------|
| `surface.nc` | Inputs X — `sst,sss,sla,adt,uo,vo,u10,v10` |
| `target.nc` | Labels Y — `thetao` (15 depths) |

`raw_tmp/` is created on download if needed; you can delete it after merging into `processed/`.

## Download more years

```bash
python src/download_pipeline.py \
  --start-date YYYY-01-01 --end-date YYYY-03-31 \
  --chunk quarter --out-name train_daily_YYYYJFM
```

Then merge into the main train cube if needed.
