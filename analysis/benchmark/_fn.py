def band(x, cuts, pts):
    """Score by band; NaN -> 0 (absent observation cannot trigger)."""
    out = np.zeros(len(x))
    v = x.values if hasattr(x, 'values') else x
    for (lo, hi), p in zip(cuts, pts):
        out = np.where((v >= lo) & (v < hi), p, out)
    return np.nan_to_num(out)


def news2(d):
    """NEWS2, five components, computed on the worst value in the window."""
    s = np.zeros(len(d))
    # respiratory rate (worst = highest, and lowest for bradypnoea)
    rr = d['resp_rate_max']
    s += band(rr, [(-1, 9), (9, 12), (12, 21), (21, 25), (25, 1e9)], [3, 1, 0, 2, 3])
    rrl = d['resp_rate_min']
    s += np.where(rrl.fillna(99) <= 8, 3, 0)
    # SpO2 (worst = lowest)
    sp = d['spo2_min']
    s += band(sp, [(-1, 92), (92, 94), (94, 96), (96, 1e9)], [3, 2, 1, 0])
    # systolic BP (worst = lowest; very high also scores)
    sb = d['sbp_nbp_min']
    s += band(sb, [(-1, 91), (91, 101), (101, 111), (111, 220), (220, 1e9)], [3, 2, 1, 0, 3])
    # pulse
    hr = d['heart_rate_max']
    s += band(hr, [(-1, 41), (41, 51), (51, 91), (91, 111), (111, 131), (131, 1e9)],
              [3, 1, 0, 1, 2, 3])
    hrl = d['heart_rate_min']
    s += np.where(hrl.fillna(99) <= 40, 3, 0)
    # temperature
    t = d['temp_c_mean'].copy()
    tf = d['temp_f_mean']
    t = t.fillna((tf - 32) * 5 / 9)
    s += band(t, [(-1, 35.01), (35.01, 36.01), (36.01, 38.01), (38.01, 39.01), (39.01, 1e9)],
              [3, 1, 0, 1, 2])
    return s


