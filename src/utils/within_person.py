"""Within-person transforms that use only each person's earlier observations."""

import numpy as np


def past_centre(X: np.ndarray, users: np.ndarray, times: np.ndarray, mode: str = "past_mean") -> np.ndarray:
    """Expresses each row of ``X`` relative to the same user's *earlier* rows.

    ``past_mean``: x - mean(earlier rows). ``past_z``: that deviation divided by the
    SD of the earlier rows. Rows without enough history (no earlier row for the mean,
    fewer than two for the SD) and features whose earlier SD is zero get 0. Rows keep
    their input order; ties in time keep their input order too.
    """
    if mode not in ("past_mean", "past_z"):
        raise ValueError(f"unknown mode {mode!r}")
    X = np.asarray(X, dtype=np.float64)
    order = np.lexsort((np.asarray(times), np.asarray(users)))
    Xs, us = X[order], np.asarray(users)[order]
    out_sorted = np.zeros_like(Xs)
    starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(us)]):
        block = Xs[a:b]
        n_prev = np.arange(b - a)[:, None].astype(np.float64)
        prev_sum = np.cumsum(block, axis=0) - block
        prev_mean = np.divide(prev_sum, n_prev, out=block.copy(), where=n_prev > 0)
        dev = block - prev_mean  # first row: 0
        if mode == "past_z":
            prev_sq = np.cumsum(block**2, axis=0) - block**2
            var = np.divide(prev_sq, n_prev, out=np.zeros_like(block), where=n_prev > 0) - prev_mean**2
            sd = np.sqrt(np.clip(var, 0.0, None))
            ok = (n_prev >= 2) & (sd > 1e-8)
            dev = np.divide(dev, sd, out=np.zeros_like(dev), where=ok)
        out_sorted[a:b] = dev
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out
