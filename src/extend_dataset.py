"""Download + harmonize NIO cubes year-by-year (2019 → present).

Keeps only processed 0.25° daily NetCDFs; deletes raw after each year.
Usage:
  python src/extend_dataset.py                 # 2019-01-01 → today (capped by product)
  python src/extend_dataset.py --start 2019 --end 2021
  python src/extend_dataset.py --years 2019 2020
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_tmp"
CHUNKS = ROOT / "data" / "processed" / "chunks"
FINAL = ROOT / "data" / "processed"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RES = 0.25
STANDARD_DEPTHS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=np.float32,
)

# Practical caps: MY reanalyses lag real time; winds switch MY→NRT ~2024-06
GLORYS_END = date(2026, 6, 15)
SST_REP_END = date(2025, 12, 18)  # OSTIA REP lag; override if download fails
WIND_MY_END = date(2024, 6, 30)
WIND_NRT_START = date(2024, 7, 1)


def target_grid():
    lat = np.arange(LAT_MIN + RES / 2, LAT_MAX, RES, dtype=np.float32)
    lon = np.arange(LON_MIN + RES / 2, LON_MAX, RES, dtype=np.float32)
    return (
        xr.DataArray(lat, dims="latitude", name="latitude"),
        xr.DataArray(lon, dims="longitude", name="longitude"),
    )


def run_subset(args: list[str]) -> None:
    cmd = ["copernicusmarine", "subset", *args, "--overwrite"]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download_range(t0: date, t1: date, tag: str) -> Path:
    """Download one period into RAW/tag/. Returns folder."""
    folder = RAW / tag
    if folder.exists():
        shutil.rmtree(folder)
    for sub in ("glorys", "sst", "sss", "sla", "currents", "winds"):
        (folder / sub).mkdir(parents=True, exist_ok=True)

    t0s, t1s = f"{t0}T00:00:00", f"{t1}T23:59:59"
    bbox = ["-x", str(LON_MIN), "-X", str(LON_MAX), "-y", str(LAT_MIN), "-Y", str(LAT_MAX)]
    time = ["-t", t0s, "-T", t1s]

    # GLORYS temperature
    run_subset(
        [
            "-i",
            "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            "-v",
            "thetao",
            *bbox,
            *time,
            "-z",
            "0",
            "-Z",
            "1000",
            "-o",
            str(folder / "glorys"),
            "-f",
            f"glorys_{tag}.nc",
        ]
    )

    # SST — prefer REP; if period past REP, try NRT OSTIA id
    sst_id = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
    if t0 > SST_REP_END:
        sst_id = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
    try:
        run_subset(
            [
                "-i",
                sst_id,
                "-v",
                "analysed_sst",
                *bbox,
                *time,
                "-o",
                str(folder / "sst"),
                "-f",
                f"sst_{tag}.nc",
            ]
        )
    except subprocess.CalledProcessError:
        # fallback NRT
        run_subset(
            [
                "-i",
                "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
                "-v",
                "analysed_sst",
                *bbox,
                *time,
                "-o",
                str(folder / "sst"),
                "-f",
                f"sst_{tag}.nc",
            ]
        )

    run_subset(
        [
            "-i",
            "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            "-v",
            "sla",
            "-v",
            "adt",
            *bbox,
            *time,
            "-o",
            str(folder / "sla"),
            "-f",
            f"sla_{tag}.nc",
        ]
    )

    run_subset(
        [
            "-i",
            "cmems_obs-mob_glo_phy_my_0.125deg_P1D-m",
            "-v",
            "so",
            *bbox,
            *time,
            "-z",
            "0",
            "-Z",
            "1",
            "-o",
            str(folder / "sss"),
            "-f",
            f"sss_{tag}.nc",
        ]
    )

    run_subset(
        [
            "-i",
            "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
            "-v",
            "uo",
            "-v",
            "vo",
            *bbox,
            *time,
            "-o",
            str(folder / "currents"),
            "-f",
            f"cur_{tag}.nc",
        ]
    )

    # Winds: MY until 2023, NRT from mid-2024; gap 2024-01..2024-06 filled with MY if possible
    if t1 <= WIND_MY_END:
        wind_id = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"
    elif t0 >= WIND_NRT_START:
        wind_id = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
    else:
        # straddling / early 2024 — try MY first for the chunk
        wind_id = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"

    try:
        run_subset(
            [
                "-i",
                wind_id,
                "-v",
                "eastward_wind",
                "-v",
                "northward_wind",
                *bbox,
                *time,
                "-o",
                str(folder / "winds"),
                "-f",
                f"wind_{tag}.nc",
            ]
        )
    except subprocess.CalledProcessError:
        run_subset(
            [
                "-i",
                "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
                "-v",
                "eastward_wind",
                "-v",
                "northward_wind",
                *bbox,
                *time,
                "-o",
                str(folder / "winds"),
                "-f",
                f"wind_{tag}.nc",
            ]
        )

    return folder


def open_nc(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="h5netcdf")


def regrid_xy(da: xr.DataArray, lat, lon) -> xr.DataArray:
    rename = {}
    if "lat" in da.dims:
        rename["lat"] = "latitude"
    if "lon" in da.dims:
        rename["lon"] = "longitude"
    if rename:
        da = da.rename(rename)
    return da.interp(latitude=lat, longitude=lon, method="linear")


def find_one(folder: Path, pattern: str) -> Path:
    hits = list(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No {pattern} in {folder}")
    return hits[0]


def _load_regrid(path: Path, var: str, lat, lon, name: str) -> xr.DataArray:
    """Open → regrid → load → close source (keeps peak RAM low)."""
    print(f"  regrid {name}…", flush=True)
    ds = open_nc(path)
    da = ds[var]
    if "depth" in da.dims:
        da = da.sel(depth=0.0, method="nearest") if da.sizes["depth"] > 1 else da.squeeze(
            "depth", drop=True
        )
    out = regrid_xy(da, lat, lon).astype("float32").load()
    out.name = name
    ds.close()
    return out


def harmonize_folder(folder: Path, tag: str) -> tuple[Path, Path]:
    """Memory-safe harmonize: one variable / time-batch at a time."""
    import gc

    lat, lon = target_grid()
    CHUNKS.mkdir(parents=True, exist_ok=True)

    sst = _load_regrid(find_one(folder / "sst", "*.nc"), "analysed_sst", lat, lon, "sst")
    sst = (sst - 273.15).astype("float32")
    sst.name = "sst"

    sss = _load_regrid(find_one(folder / "sss", "*.nc"), "so", lat, lon, "sss")
    sla = _load_regrid(find_one(folder / "sla", "*.nc"), "sla", lat, lon, "sla")
    adt = _load_regrid(find_one(folder / "sla", "*.nc"), "adt", lat, lon, "adt")
    uo = _load_regrid(find_one(folder / "currents", "*.nc"), "uo", lat, lon, "uo")
    vo = _load_regrid(find_one(folder / "currents", "*.nc"), "vo", lat, lon, "vo")

    # Winds: daily-mean on native grid first, then regrid (avoid hourly×interp)
    print("  winds daily-mean + regrid…", flush=True)
    wind = open_nc(find_one(folder / "winds", "*.nc"))
    u_day = wind["eastward_wind"].resample(time="1D").mean().astype("float32").load()
    v_day = wind["northward_wind"].resample(time="1D").mean().astype("float32").load()
    wind.close()
    del wind
    gc.collect()
    u10 = regrid_xy(u_day, lat, lon).astype("float32").load()
    v10 = regrid_xy(v_day, lat, lon).astype("float32").load()
    u10.name, v10.name = "u10", "v10"
    del u_day, v_day
    gc.collect()

    surface = xr.merge([sst, sss, sla, adt, uo, vo, u10, v10], join="inner").sortby("time")
    del sst, sss, sla, adt, uo, vo, u10, v10
    gc.collect()

    # GLORYS: vertical interp + horizontal regrid in time batches
    print("  glorys depth+xy regrid (batched)…", flush=True)
    gpath = find_one(folder / "glorys", "*.nc")
    glorys_ds = open_nc(gpath)
    glorys = glorys_ds["thetao"]
    zmin = float(glorys.depth.min())
    z_for = np.clip(STANDARD_DEPTHS, zmin, float(glorys.depth.max()))
    n_time = glorys.sizes["time"]
    batch = 10
    parts = []
    for i in range(0, n_time, batch):
        sl = glorys.isel(time=slice(i, min(i + batch, n_time)))
        t = sl.interp(depth=z_for, method="linear")
        t = t.assign_coords(depth=("depth", STANDARD_DEPTHS))
        t = regrid_xy(t, lat, lon).astype("float32").load()
        parts.append(t)
        print(f"    glorys times {i}:{min(i + batch, n_time)}/{n_time}", flush=True)
        gc.collect()
    glorys_ds.close()
    temp = xr.concat(parts, dim="time")
    temp.name = "thetao"
    del parts
    gc.collect()
    target = temp.to_dataset()
    del temp
    gc.collect()

    # Align time to calendar day
    surface = surface.assign_coords(time=surface.time.dt.floor("D"))
    target = target.assign_coords(time=target.time.dt.floor("D"))
    times = np.intersect1d(surface.time.values, target.time.values)
    surface = surface.sel(time=times)
    target = target.sel(time=times)
    ocean = surface["sst"].notnull().any("time")
    surface = surface.where(ocean)
    target = target.where(ocean)

    surf_path = CHUNKS / f"surface_{tag}.nc"
    tgt_path = CHUNKS / f"target_{tag}.nc"
    print(f"  writing {surf_path.name}…", flush=True)
    surface.to_netcdf(
        surf_path,
        encoding={v: {"zlib": True, "complevel": 4} for v in surface.data_vars},
        engine="h5netcdf",
    )
    print(f"  writing {tgt_path.name}…", flush=True)
    target.to_netcdf(
        tgt_path,
        encoding={v: {"zlib": True, "complevel": 4} for v in target.data_vars},
        engine="h5netcdf",
    )
    surface.close()
    target.close()
    print(f"Wrote {surf_path} and {tgt_path}", flush=True)
    return surf_path, tgt_path


def _merge_files(surf_files: list[Path], tgt_files: list[Path], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging {len(surf_files)} chunks → {out_dir} ({prefix})", flush=True)
    surf = xr.open_mfdataset(surf_files, combine="by_coords", engine="h5netcdf")
    tgt = xr.open_mfdataset(tgt_files, combine="by_coords", engine="h5netcdf")
    surf = surf.assign_coords(time=surf.time.dt.floor("D"))
    tgt = tgt.assign_coords(time=tgt.time.dt.floor("D"))
    times = np.intersect1d(surf.time.values, tgt.time.values)
    surf = surf.sel(time=times).sortby("time").load()
    tgt = tgt.sel(time=times).sortby("time").load()

    t0 = str(surf.time.values[0])[:10]
    t1 = str(surf.time.values[-1])[:10]
    surf_out = out_dir / f"surface_{prefix}_{t0}_{t1}.nc"
    tgt_out = out_dir / f"target_{prefix}_{t0}_{t1}.nc"
    enc_s = {v: {"zlib": True, "complevel": 4} for v in surf.data_vars}
    enc_t = {v: {"zlib": True, "complevel": 4} for v in tgt.data_vars}
    surf.to_netcdf(surf_out, encoding=enc_s, engine="h5netcdf")
    tgt.to_netcdf(tgt_out, encoding=enc_t, engine="h5netcdf")
    # canonical names inside split folder
    shutil.copy(surf_out, out_dir / "surface.nc")
    shutil.copy(tgt_out, out_dir / "target.nc")
    print(f"  surface: {surf_out.name} {dict(surf.sizes)}", flush=True)
    print(f"  target:  {tgt_out.name} {dict(tgt.sizes)}", flush=True)
    surf.close()
    tgt.close()


def merge_chunks(split: str = "all") -> None:
    """Merge daily chunk files. split='train'|'test'|'all'."""
    surf_files = sorted(CHUNKS.glob("surface_*.nc"))
    tgt_files = sorted(CHUNKS.glob("target_*.nc"))
    if not surf_files:
        raise RuntimeError("No processed chunk files to merge")

    def in_range(path: Path, start: str, end: str) -> bool:
        # surface_YYYY-MM-DD_YYYY-MM-DD.nc
        name = path.stem.replace("surface_", "").replace("target_", "")
        chunk_start = name.split("_")[0]
        return start <= chunk_start <= end

    if split == "train":
        sf = [p for p in surf_files if in_range(p, "2019-01-01", "2020-12-31")]
        tf = [p for p in tgt_files if in_range(p, "2019-01-01", "2020-12-31")]
        _merge_files(sf, tf, FINAL / "train", "train")
    elif split == "test":
        sf = [p for p in surf_files if in_range(p, "2026-06-01", "2026-07-31")]
        tf = [p for p in tgt_files if in_range(p, "2026-06-01", "2026-07-31")]
        _merge_files(sf, tf, FINAL / "test", "test")
    else:
        _merge_files(surf_files, tgt_files, FINAL, "all")
        # also build train/test if those chunks exist
        try:
            merge_chunks("train")
        except Exception as e:
            print("train merge skipped:", e, flush=True)
        try:
            merge_chunks("test")
        except Exception as e:
            print("test merge skipped:", e, flush=True)


def year_bounds(year: int, global_end: date) -> tuple[date, date]:
    t0 = date(year, 1, 1)
    t1 = date(year, 12, 31)
    if t1 > global_end:
        t1 = global_end
    if t0 > global_end:
        raise ValueError(f"Year {year} starts after global end {global_end}")
    return t0, t1


def process_chunks(chunks: list[tuple[date, date]]) -> None:
    for a, b in chunks:
        tag = f"{a.isoformat()}_{b.isoformat()}"
        surf_y = CHUNKS / f"surface_{tag}.nc"
        if surf_y.exists():
            print(f"Skip existing {surf_y.name}", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)
        folder = download_range(a, b, tag)
        harmonize_folder(folder, tag)
        shutil.rmtree(folder, ignore_errors=True)
        print(f"Removed raw {folder}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2019)
    parser.add_argument("--end", type=int, default=None, help="inclusive calendar year")
    parser.add_argument("--years", type=int, nargs="*", default=None)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--merge-split", choices=["train", "test", "all"], default="all")
    parser.add_argument(
        "--quarterly",
        action="store_true",
        help="Download in 3-month chunks (safer for winds/GLORYS size)",
    )
    parser.add_argument(
        "--preset",
        choices=["train2019_2020_test2026jun", "none"],
        default="none",
        help="SIH PoC split: train 2019-2020, test 2026-06 (Jul unavailable in GLORYS MY)",
    )
    args = parser.parse_args()

    if args.merge_only:
        merge_chunks(args.merge_split)
        return

    RAW.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(parents=True, exist_ok=True)

    if args.preset == "train2019_2020_test2026jun":
        # Train: full 2019 + 2020. Test: June 2026 (GLORYS MY has no July 2026 yet).
        print("Preset: TRAIN 2019-01-01→2020-12-31 | TEST 2026-06-01→2026-06-30", flush=True)
        train_chunks: list[tuple[date, date]] = []
        for year in (2019, 2020):
            for a, b in [
                (date(year, 1, 1), date(year, 3, 31)),
                (date(year, 4, 1), date(year, 6, 30)),
                (date(year, 7, 1), date(year, 9, 30)),
                (date(year, 10, 1), date(year, 12, 31)),
            ]:
                train_chunks.append((a, b))
        process_chunks(train_chunks)
        process_chunks([(date(2026, 6, 1), date(2026, 6, 30))])
        merge_chunks("train")
        merge_chunks("test")
        print("ALL DONE (train 2019-2020 + test 2026-06)", flush=True)
        return

    today = date.today()
    global_end = min(today, GLORYS_END)
    end_year = args.end or global_end.year
    years = args.years or list(range(args.start, end_year + 1))

    print(f"Extending dataset years={years} global_end={global_end}", flush=True)

    chunks: list[tuple[date, date]] = []
    for year in years:
        t0, t1 = year_bounds(year, global_end)
        if args.quarterly or (t1 - t0).days > 100:
            quarters = [
                (date(year, 1, 1), date(year, 3, 31)),
                (date(year, 4, 1), date(year, 6, 30)),
                (date(year, 7, 1), date(year, 9, 30)),
                (date(year, 10, 1), date(year, 12, 31)),
            ]
            for a, b in quarters:
                a = max(a, t0)
                b = min(b, t1)
                if a <= b:
                    chunks.append((a, b))
        else:
            chunks.append((t0, t1))

    process_chunks(chunks)
    merge_chunks(args.merge_split)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("Download failed:", e, file=sys.stderr)
        sys.exit(e.returncode)
