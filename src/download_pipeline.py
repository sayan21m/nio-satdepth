#!/usr/bin/env python3
"""Configurable NIO CMEMS download + harmonize pipeline.

Select any date range; downloads in quarterly chunks by default, regrids to
0.25° daily, writes processed cubes, deletes raw.

Examples
--------
  # Jan–Jun 2019 only
  python src/download_pipeline.py --start-date 2019-01-01 --end-date 2019-06-30

  # Custom range, monthly chunks
  python src/download_pipeline.py --start-date 2020-01-01 --end-date 2020-03-31 --chunk month

  # Skip download; merge existing daily chunks in range
  python src/download_pipeline.py --start-date 2019-01-01 --end-date 2019-06-30 --merge-only

  # List planned chunks without downloading
  python src/download_pipeline.py --start-date 2019-01-01 --end-date 2019-06-30 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Reuse download / harmonize / merge from extend_dataset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extend_dataset import (  # noqa: E402
    CHUNKS,
    FINAL,
    process_chunks,
    merge_chunks,
    _merge_files,
)


def parse_day(s: str) -> date:
    return date.fromisoformat(s)


def month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


def quarter_bounds(year: int, q: int) -> tuple[date, date]:
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    sm, sd = starts[q]
    em, ed = ends[q]
    return date(year, sm, sd), date(year, em, ed)


def iter_chunks(start: date, end: date, mode: str) -> list[tuple[date, date]]:
    if start > end:
        raise ValueError(f"start-date {start} is after end-date {end}")
    chunks: list[tuple[date, date]] = []

    if mode == "all":
        return [(start, end)]

    if mode == "month":
        cur = date(start.year, start.month, 1)
        while cur <= end:
            a = max(cur, start)
            b = min(month_end(cur), end)
            if a <= b:
                chunks.append((a, b))
            if cur.month == 12:
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = date(cur.year, cur.month + 1, 1)
        return chunks

    # quarterly (default)
    y = start.year
    while y <= end.year:
        for q in (1, 2, 3, 4):
            a, b = quarter_bounds(y, q)
            a = max(a, start)
            b = min(b, end)
            if a <= b:
                chunks.append((a, b))
        y += 1
    return chunks


def merge_range(start: date, end: date, out_name: str) -> None:
    """Merge daily chunks whose chunk-start falls inside [start, end]."""
    surf = sorted(CHUNKS.glob("surface_*.nc"))
    tgt = sorted(CHUNKS.glob("target_*.nc"))

    def keep(p: Path) -> bool:
        tag = p.stem.replace("surface_", "").replace("target_", "")
        chunk_start = date.fromisoformat(tag.split("_")[0])
        return start <= chunk_start <= end

    sf = [p for p in surf if keep(p)]
    tf = [p for p in tgt if keep(p)]
    if not sf:
        raise SystemExit(
            f"No processed chunks in {CHUNKS} for {start} → {end}. "
            "Run download first."
        )
    out_dir = FINAL / out_name
    _merge_files(sf, tf, out_dir, out_name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download & harmonize NIO satellite/ocean data for a chosen date range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--start-date",
        type=parse_day,
        required=True,
        help="Inclusive start (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        type=parse_day,
        required=True,
        help="Inclusive end (YYYY-MM-DD)",
    )
    p.add_argument(
        "--chunk",
        choices=["quarter", "month", "all"],
        default="quarter",
        help="How to split the range for downloads (default: quarter)",
    )
    p.add_argument(
        "--out-name",
        default=None,
        help="Output folder name under data/processed/ (default: range tag)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned chunks and exit",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing daily chunks in this range",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Download/harmonize chunks but do not merge into one cube",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    start, end = args.start_date, args.end_date
    chunks = iter_chunks(start, end, args.chunk)
    out_name = args.out_name or f"{start.isoformat()}_{end.isoformat()}"

    print(f"Range: {start} → {end}", flush=True)
    print(f"Chunk mode: {args.chunk}  ({len(chunks)} chunk(s))", flush=True)
    for a, b in chunks:
        tag = f"{a.isoformat()}_{b.isoformat()}"
        exists = (CHUNKS / f"surface_{tag}.nc").exists()
        print(f"  - {tag}{'  [exists]' if exists else ''}", flush=True)

    if args.dry_run:
        print("Dry-run only. Exiting.", flush=True)
        return

    if args.merge_only:
        merge_range(start, end, out_name)
        print("Merge done.", flush=True)
        return

    CHUNKS.mkdir(parents=True, exist_ok=True)
    process_chunks(chunks)

    if not args.no_merge:
        merge_range(start, end, out_name)
        print(f"Done. Cubes in data/processed/{out_name}/", flush=True)
    else:
        print("Done (chunks only, no merge).", flush=True)


if __name__ == "__main__":
    main()
