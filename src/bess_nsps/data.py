"""
data.py — data loading utilities for the DP optimizer

This module provides loaders for time-series profiles coming from CSV or MATLAB (.mat) files. Profiles typically contain load demand or other power data as a function of time. The functions here standardise those profiles into numpy arrays that the optimisation code can work with.


It supports:
- CSV files with columns [date, time, pow_kw].
- MATLAB files (both v7.x via SciPy and v7.3+ HDF5 via h5py) - Actually NOT WORKING properly.
- Optional metadata for windowing (time or index bounds).
"""

import os
import os
from typing import Tuple, Any, Dict, Iterable, Optional
import numpy as np
import pandas as pd


# ------------------------------ small utils ------------------------------

def _is_num_array(x) -> bool:
    # Check if x can be converted into a non-empty numeric numpy array
    try:
        a = np.asarray(x)
        return np.issubdtype(a.dtype, np.number) and a.size > 0
    except Exception:
        return False


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    # Recursively extract numeric arrays from dict-like MATLAB structures. Returns a flat mapping name -> np.ndarray

    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        # MATLAB struct as np.void
        if hasattr(v, "dtype") and getattr(v.dtype, "names", None):
            for field in v.dtype.names:
                out.update(_flatten_dict({field: v[field]}, prefix=name))
        elif isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=name))
        else:
            try:
                a = np.asarray(v).squeeze()
                if _is_num_array(a):
                    out[name] = a
            except Exception:
                pass
    return out


# ------------------------------ CSV loader -------------------------------

def _load_csv_profile(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # CSV loader for files with columns:
    #   - date : 'YYYY-MM-DD'
    #   - time : 'HH:MM:SS'
    #   - pow_kw : power in kW
    # Returns (pow_load[kW], t_plot[min from first sample]).

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # Check CSV file
    if "date" not in cols or "time" not in cols:
        raise KeyError(f"{os.path.basename(path)} must have 'date' and 'time' columns")
    if "pow_kw" not in cols:
        raise KeyError(f"{os.path.basename(path)} must have 'pow_kw' column (kW)")

    dcol = cols["date"]; tcol = cols["time"]; pcol = cols["pow_kw"]

    # Build datetime index and minutes since start
    dt = pd.to_datetime(df[dcol].astype(str) + " " + df[tcol].astype(str), errors="raise", utc=False)
    t_min = (dt - dt.iloc[0]).dt.total_seconds() / 60.0
    p_kw = pd.to_numeric(df[pcol], errors="coerce")

    # Filter out invalid rows
    mask = ~(t_min.isna() | p_kw.isna())
    t = t_min[mask].to_numpy(dtype=float)
    p = p_kw[mask].to_numpy(dtype=float)

    # Ensure increasing time
    order = np.argsort(t)
    return p[order], t[order]


def csv_dt_bounds_to_minutes(path: str, dt_start: str, dt_end: str) -> Tuple[float, float]:
    # For a CSV profile, convert datetime bounds into minutes relative to first row of a CSV profile.
    
    df = pd.read_csv(path, nrows=1)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "time" not in cols:
        raise KeyError(f"{os.path.basename(path)} needs 'date' and 'time' to use --dt-start/--dt-end")

    base_dt = pd.to_datetime(df[cols["date"]].iloc[0] + " " + df[cols["time"]].iloc[0], utc=False)

    def _parse(s: str):
        s = s.strip().replace("T", " ")
        parts = s.split()
        if len(parts) == 2 and "-" in parts[0]:
            # full datetime
            return pd.to_datetime(s, utc=False)
        # time-of-day only → anchor to base date
        return pd.to_datetime(str(base_dt.date()) + " " + s, utc=False)

    dt0 = _parse(dt_start)
    dt1 = _parse(dt_end)

    t0min = (dt0 - base_dt).total_seconds() / 60.0
    t1min = (dt1 - base_dt).total_seconds() / 60.0
    return float(t0min), float(t1min)


# ------------------------------ MAT loader -------------------------------

def _try_scipy_load(path: str) -> Dict[str, Any]:
    # Try loading .mat file via scipy.io.loadmat.
    from scipy.io import loadmat
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def _try_h5py_load(path: str) -> Dict[str, Any]:
    # Try loading v7.3 MAT file via h5py.
    import h5py
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    arr = np.array(obj)
                    out[name] = arr
                except Exception:
                    pass
        f.visititems(visit)
    return out


def get_mat_window(path: str) -> Optional[Dict[str, Any]]:
    # Optional metadata reader: looks for index or time window metadata inside a MAT file.

    try:
        from scipy.io import loadmat
        md = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        return None

    def pick(k): return md.get(k, None)

    # Index windows (convert 1-based MATLAB -> 0-based Python if needed)
    for a, b in [("idx_start", "idx_end"), ("i0", "i1")]:
        i0, i1 = pick(a), pick(b)
        if i0 is not None and i1 is not None:
            i0 = int(np.asarray(i0).ravel()[0])
            i1 = int(np.asarray(i1).ravel()[0])
            if i0 >= 1 and i1 >= 1:
                i0 -= 1; i1 -= 1
            return {"i0": max(0, i0), "i1": max(0, i1)}

    # Time windows in minutes
    for a, b in [("t_start_min", "t_end_min"), ("t0", "t1")]:
        t0, t1 = pick(a), pick(b)
        if t0 is not None and t1 is not None:
            t0 = float(np.asarray(t0).ravel()[0])
            t1 = float(np.asarray(t1).ravel()[0])
            return {"t0": t0, "t1": t1}

    return None


def load_profile(path: str, step_min: float = 1.0, max_samples: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    # Load a power profile from either CSV or MAT file, returning (p[t], t[t]).

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_csv_profile(path)

    # --- .mat path ---
    md: Dict[str, Any] = {}
    tried = []

    # 1) Try SciPy first
    try:
        md = _try_scipy_load(path); tried.append("scipy")
    except Exception:
        md = {}

    # 2) If empty or suspicious (v7.3), try h5py
    if not md:
        try:
            md = _try_h5py_load(path); tried.append("h5py")
        except Exception:
            pass

    # Build a flat map of arrays: name -> np.ndarray
    flat = _flatten_dict(md)

    # Preferred aliases
    p_aliases = [
        "pow_load","P_load","P_req","P_required","Pdem","Pload","pow","P","P_ship","P_meas"
    ]
    t_aliases = [
        "t_plot","t","time_min","time","Time","minutes","mins","min","T","ts","timestamp"
    ]

    def pick_by_alias(aliases: Iterable[str]) -> Optional[np.ndarray]:
        # exact key
        for k in aliases:
            if k in md and _is_num_array(md[k]):
                return np.asarray(md[k]).astype(float).ravel()
        # flattened keys (nested)
        for k in aliases:
            for fk in flat:
                if fk.split(".")[-1] == k or fk == k:
                    a = np.asarray(flat[fk]).squeeze()
                    if _is_num_array(a):
                        return a.astype(float).ravel()
        return None

    pow_load = pick_by_alias(p_aliases)
    t_plot = pick_by_alias(t_aliases)

    # Heuristic fallback: take the largest 1-D numeric as pow_load
    if pow_load is None:
        candidates = []
        for k, a in flat.items():
            a = np.asarray(a).squeeze()
            if _is_num_array(a) and a.ndim == 1 and a.size > 10:
                candidates.append((k, a))
        if not candidates:
            raise KeyError(f"Could not find a numeric load vector in {os.path.basename(path)} (tried {tried})")
        candidates.sort(key=lambda kv: kv[1].size, reverse=True)
        pow_load = candidates[0][1].astype(float).ravel()

    # Time vector: try to match length, else generate simple index
    if t_plot is None or t_plot.size != pow_load.size:
        t_match = None
        for k, a in flat.items():
            a = np.asarray(a).squeeze()
            if _is_num_array(a) and a.ndim == 1 and a.size == pow_load.size:
                t_match = a.astype(float).ravel()
                break
        if t_match is None:
            t_plot = np.arange(pow_load.size, dtype=float)
        else:
            t_plot = t_match

    return pow_load.astype(float).ravel(), t_plot.astype(float).ravel()


def choose_profiles(names, data_dir: str = "data"):
    # Resolve profile names into file paths inside a data directory.
    
    paths = []
    for n in names:
        p = os.path.join(data_dir, n)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Profile not found: {p}")
        paths.append(p)
    return paths