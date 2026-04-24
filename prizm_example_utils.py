from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
from cegalprizm.pythontool.seismic import SeismicCube

def _parse_range(stats: Dict[str, Any], key: str) -> Tuple[int, int]:
    vals = [int(x) for x in str(stats[key]).split() if x.lstrip("-").isdigit()]
    if len(vals) < 2:
        raise ValueError(f"Could not parse '{key}' from stats: {stats.get(key)}")
    return vals[0], vals[1]

def chunk_extent(cube: SeismicCube, extent: List[int], mode: str) -> List[int]:
    """
    Convert extents between index and seismic coordinates.
    mode='ijk_grid': [i0,i1,j0,j1] -> [il0,il1,xl0,xl1]
    mode='seismic' : [il0,il1,xl0,xl1] -> [i0,i1,j0,j1]
    """
    if mode not in ("ijk_grid", "seismic"):
        raise ValueError(f"mode must be 'ijk_grid' or 'seismic', got {mode!r}")

    i0, i1, j0, j1 = extent
    ni, nj, _ = cube.extent
    r = cube.retrieve_stats()
    il0, il1 = _parse_range(r, "Inline range")
    xl0, xl1 = _parse_range(r, "Crossline range")

    if mode == "ijk_grid":
        i1, j1 = min(i1, ni - 1), min(j1, nj - 1)
        if not (0 <= i0 <= i1 < ni and 0 <= j0 <= j1 < nj):
            raise ValueError(f"IJK extent out of bounds: {(i0, i1, j0, j1)} for {(ni, nj)}")
        return [il0 + i0, il0 + i1, xl0 + j0, xl0 + j1]

    if not (il0 <= i0 <= i1 <= il1 and xl0 <= j0 <= j1 <= xl1):
        raise ValueError(f"Seismic extent out of bounds: {(i0, i1, j0, j1)} for IL[{il0},{il1}] XL[{xl0},{xl1}]")
    return [i0 - il0, i1 - il0, j0 - xl0, j1 - xl0]

def chunk_boundaries(
    extent: Tuple[int, int, int],
    chunk_shape: Tuple[int, int],
) -> List[Tuple[int, int, int, int, int, int]]:
    """Return inclusive chunk bounds: (i0,i1,j0,j1,k0,k1)."""
    ni, nj, nk = extent
    ci, cj = chunk_shape

    def ranges(n: int, c: int) -> List[Tuple[int, int]]:
        out = []
        start = 0
        while start < n:
            end = min(start + c - 1, n - 1)
            out.append((start, end))
            start = end + 1
        return out

    return [
        (i0, i1, j0, j1, 0, nk - 1)
        for i0, i1 in ranges(ni, ci)
        for j0, j1 in ranges(nj, cj)
    ]

def seismic_cubes_as_xarray(ptp, seismic_cube_name: str) -> xr.DataArray:
    """Read full cube into DataArray(inline, xline, twt) with i/j coords."""
    s = ptp.seismic_cubes.get_by_name(seismic_cube_name)
    ni, nj, nk = s.extent
    r = s.retrieve_stats()

    il0, _ = _parse_range(r, "Inline range")
    xl0, _ = _parse_range(r, "Crossline range")

    z0 = abs(float(r["Time Max"])) if "Time Max" in r and "Time Min" in r else abs(float(r["Depth Max"]))
    dz = float(r["Sample interval"])

    data = s.chunk(irange=(0, ni - 1), jrange=(0, nj - 1), krange=(0, nk - 1)).as_array()

    return xr.DataArray(
        data,
        name="seismic",
        dims=("inline", "xline", "twt"),
        coords={
            "inline": np.arange(il0, il0 + data.shape[0]),
            "xline": np.arange(xl0, xl0 + data.shape[1]),
            "twt": z0 + dz * np.arange(data.shape[2]),
            "i": ("inline", np.arange(data.shape[0], dtype=np.int32)),
            "j": ("xline", np.arange(data.shape[1], dtype=np.int32)),
        },
    )

def live_trace_map(
    seismic_da: xr.DataArray,
    twt_range: Optional[Tuple[float, float]] = None,
    amp_range: Optional[Tuple[float, float]] = None,
):
    """
    2D live map (inline,xline): 1=live, 0=dead.
    Dead if constant over twt and/or fully inside amp_range.
    """
    da = seismic_da if twt_range is None else seismic_da.sel(twt=slice(*twt_range))
    nt = da.sizes["twt"]
    if nt == 0:
        raise ValueError("Selected twt_range produced empty data.")

    first = da.isel(twt=0)
    constant_dead = xr.full_like(first, True, dtype=bool)
    in_range_dead = xr.full_like(first, True, dtype=bool) if amp_range is not None else None

    for k in range(nt):
        s = da.isel(twt=k)
        constant_dead &= (s == first)
        if amp_range is not None:
            a0, a1 = amp_range
            in_range_dead &= (s >= a0) & (s <= a1)

    dead = constant_dead | in_range_dead if amp_range is not None else constant_dead
    live_map = (~dead).astype(np.uint8).rename("live_trace").assign_coords(i=seismic_da["i"], j=seismic_da["j"])\n
    n_total = live_map.size
    n_live = int(live_map.sum().item())
    stats = {
        "n_total_traces": n_total,
        "n_live_traces": n_live,
        "n_dead_traces": n_total - n_live,
        "dead_pct": 100.0 * (n_total - n_live) / n_total,
        "twt_first": float(da.twt.values[0]),
        "twt_last": float(da.twt.values[-1]),
    }
    return live_map, stats

def live_map_to_df(live_map: xr.DataArray) -> pd.DataFrame:
    """2D map -> DataFrame[i, j, cvalues]."""
    lm = live_map.rename("cvalues").transpose("inline", "xline")
    i2d, j2d = xr.broadcast(lm["i"], lm["j"])
    return pd.DataFrame(
        {
            "i": i2d.values.ravel(),
            "j": j2d.values.ravel(),
            "cvalues": lm.values.ravel().astype(np.uint8),
        }
    )

def add_xy(df: pd.DataFrame, seismic_volume) -> pd.DataFrame:
    """Append x,y from ijk positions (k=0)."""
    is_list = df["i"].astype(int).tolist()
    js_list = df["j"].astype(int).tolist()
    zs_list = [0] * len(is_list)
    x, y, _ = seismic_volume.ijks_to_positions((is_list, js_list, zs_list))
    out = df.copy()
    out["x"] = np.asarray(x, dtype=float)
    out["y"] = np.asarray(y, dtype=float)
    return out[["i", "j", "x", "y", "cvalues"]]

def create_livetrace_df(df: pd.DataFrame, value: int = 1) -> pd.DataFrame:
    """Build Poly/Vert table from rows where cvalues==value.
    Groups by i and creates open polylines along j."""
    sel = df[df["cvalues"] == value].sort_values(["i", "j"])
    rows = []
    poly_id = 1

    for _, g in sel.groupby("i", sort=True):
        if len(g) < 2:
            continue
        for v, r in enumerate(g.itertuples(index=False), start=1):
            rows.append(
                {"Poly": poly_id, "Vert": v, "X": float(r.x), "Y": float(r.y), "Z": 0.0, "Closed": False}
            )
        poly_id += 1

    out = pd.DataFrame(rows, columns=["Poly", "Vert", "X", "Y", "Z", "Closed"])
    if out.empty:
        raise ValueError("No valid polylines: need at least 2 points per line.")
    return out.astype({"Poly": int, "Vert": int, "X": float, "Y": float, "Z": float, "Closed": bool})