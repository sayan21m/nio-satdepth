#!/usr/bin/env bash
# Keep Jan–Mar 2019 running. After Q1+Q2 (through Jun) are processed, stop the
# long preset and merge 2019-01 → 2019-06 only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/cap_to_2019H1.log"
CHUNKS="$ROOT/data/processed/chunks"
Q1="$CHUNKS/surface_2019-01-01_2019-03-31.nc"
Q2="$CHUNKS/surface_2019-04-01_2019-06-30.nc"

mkdir -p logs
echo "[$(date -Iseconds)] Cap plan: keep Q1 running → allow Q2 → stop before Q3 → merge H1." | tee -a "$LOG"

echo "[$(date -Iseconds)] Waiting for Q1 2019-01→03…" | tee -a "$LOG"
while [[ ! -f "$Q1" ]]; do sleep 30; done
echo "[$(date -Iseconds)] Q1 done." | tee -a "$LOG"

echo "[$(date -Iseconds)] Waiting for Q2 2019-04→06 (from current job or pipeline)…" | tee -a "$LOG"
# If preset dies early, kick pipeline for Jan–Jun (skips Q1).
if [[ ! -f "$Q2" ]] && ! pgrep -f 'extend_dataset.py --preset' >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] Preset not running; starting range pipeline for H1." | tee -a "$LOG"
  PYTHONUNBUFFERED=1 python src/download_pipeline.py \
    --start-date 2019-01-01 --end-date 2019-06-30 \
    --chunk quarter --out-name train_2019H1 2>&1 | tee -a "$LOG"
  echo "[$(date -Iseconds)] Cap-to-2019H1 complete." | tee -a "$LOG"
  exit 0
fi

while [[ ! -f "$Q2" ]]; do
  # If preset crashed mid-Q2, finish with pipeline
  if ! pgrep -f 'extend_dataset.py --preset' >/dev/null 2>&1 \
     && ! pgrep -f 'copernicusmarine subset' >/dev/null 2>&1 \
     && ! pgrep -f 'download_pipeline.py' >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] Jobs idle without Q2; launching pipeline." | tee -a "$LOG"
    PYTHONUNBUFFERED=1 python src/download_pipeline.py \
      --start-date 2019-01-01 --end-date 2019-06-30 \
      --chunk quarter --out-name train_2019H1 2>&1 | tee -a "$LOG"
    echo "[$(date -Iseconds)] Cap-to-2019H1 complete." | tee -a "$LOG"
    exit 0
  fi
  sleep 30
done
echo "[$(date -Iseconds)] Q2 done. Stopping preset before Q3+…" | tee -a "$LOG"

# Wait for any in-flight subset to finish cleanly, then kill orchestrator
while pgrep -f 'copernicusmarine subset' >/dev/null 2>&1; do sleep 15; done
pkill -f 'extend_dataset.py --preset' 2>/dev/null || true
sleep 2

PYTHONUNBUFFERED=1 python src/download_pipeline.py \
  --start-date 2019-01-01 --end-date 2019-06-30 \
  --merge-only --out-name train_2019H1 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] Cap-to-2019H1 complete → data/processed/train_2019H1/" | tee -a "$LOG"
