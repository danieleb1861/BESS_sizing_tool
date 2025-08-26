import numpy as np
from typing import Tuple

def slice_by_indices(p_kw, t_min, i0: int, i1: int):
    i0 = max(0, int(i0)); i1 = int(i1)
    return np.asarray(p_kw)[i0:i1], np.asarray(t_min)[i0:i1]

def slice_by_time(p_kw, t_min, t0_min: float, t1_min: float):
    t = np.asarray(t_min, float).ravel()
    mask = (t >= float(t0_min)) & (t <= float(t1_min))
    return np.asarray(p_kw)[mask], t[mask]

def slice_by_datetime(dt, p_kw, t0, t1):
    dt = np.asarray(dt).astype('datetime64[ns]').ravel()
    mask = (dt >= np.datetime64(t0)) & (dt <= np.datetime64(t1))
    return np.asarray(p_kw)[mask], dt[mask]

def _window_score_ramp(p, w):
    dp = np.abs(np.diff(p, prepend=p[0]))
    # sum|dp| in sliding window of width w (samples)
    c = np.cumsum(dp)
    out = c[w:] - c[:-w]
    # align length: put zeros at edges
    padL = w//2; padR = len(p) - len(out) - padL
    return np.pad(out, (padL, padR), mode='constant')

def _window_score_std(p, w):
    # simple rolling std using cumulative sums
    p = np.asarray(p, float)
    c1 = np.cumsum(p); c2 = np.cumsum(p*p)
    c1 = np.pad(c1, (1,0)); c2 = np.pad(c2, (1,0))
    s1 = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    var = np.maximum(s2/w - (s1/w)**2, 0.0)
    out = np.sqrt(var)
    padL = w//2; padR = len(p) - len(out) - padL
    return np.pad(out, (padL, padR), mode='constant')

def find_manoeuvre_window(p_kw, t_min, window_min: float, method: str = "ramp") -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick the most 'interesting' window of length window_min based on:
      - ramp: maximize sum |dp/dt|
      - std:  maximize rolling std
    """
    t = np.asarray(t_min, float).ravel()
    p = np.asarray(p_kw, float).ravel()
    if len(t) < 2 or window_min <= 0:
        return p, t
    # assume ~1-minute spacing; convert minutes to samples
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    w = max(2, int(round(window_min / max(dt, 1e-9))))
    w = min(w, len(p)-1) if len(p) > 2 else 2
    if method == "std":
        score = _window_score_std(p, w)
    else:
        score = _window_score_ramp(p, w)
    k = int(np.argmax(score))
    half = w//2
    i0 = max(0, k - half)
    i1 = min(len(p), i0 + w)
    return p[i0:i1], t[i0:i1]