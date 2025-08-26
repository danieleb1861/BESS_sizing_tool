import numpy as np

def get_default_interval(basename: str, manov: int, dt_native):
    """
    Return (t0, t1) as datetime64 for known files, mirroring your MATLAB choices.
    We anchor only by time-of-day on the first day of the series.
    """
    if dt_native is None or len(dt_native) == 0:
        return None
    day0 = dt_native[0].astype('datetime64[D]')

    def hhmmss(h, m, s):  # build datetime64 from day0 + time
        return day0 + np.timedelta64(int(h), 'h') + np.timedelta64(int(m), 'm') + np.timedelta64(int(s), 's')

    b = basename.upper()
    # IND-OMN examples you shared
    if b == "IND-OMN.MAT" and manov == 1:
        # NO. 3 ramp up: 15:13:20 → 15:15:30
        return hhmmss(15,13,20), hhmmss(15,15,30)
    if b == "IND-OMN.MAT" and manov == 2:
        # NO. 1 multiple ramps + switch: 19:26:51 → 19:29:22
        return hhmmss(19,26,51), hhmmss(19,29,22)

    # OMN-PAK examples you shared (manov 1 first)
    if b == "OMN-PAK.MAT" and manov == 1:
        # NO. 1 ramp up + multiple ramps: 14:54:35 → 15:02:10
        return hhmmss(14,54,35), hhmmss(15,2,10)
    if b == "OMN-PAK.MAT" and manov == 2:
        # Not critical example: 13:18:20 → 13:26:14
        return hhmmss(13,18,20), hhmmss(13,26,14)

    # Add other files here as needed…

    return None