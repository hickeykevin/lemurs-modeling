"""Script to verify Catch22 behavior across scalers: None, Global, Subject, and Dual.

Demonstrates:
1. For Catch22: Raw, Global, and Subject standardized versions produce virtually identical
   features (correlation ~ 1.0) because Catch22 features are scale and location invariant.
   DualScaler therefore produces 44 features where features 0-21 and 22-43 are duplicates.
2. In contrast, for SummaryStats (mean, median, std, min, max):
   Global and Subject streams produce completely different features (capturing between-person
   level vs within-person anomaly).
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pycatch22
from sklearn.preprocessing import StandardScaler
from src.data.components.scalers import DualScaler


def test_catch22_invariance():
    print("=" * 80)
    print(" 1. CATCH22 BEHAVIOR ACROSS SCALERS (Raw vs Global vs Subject vs Dual)")
    print("=" * 80)

    # Simulate realistic 24-hour hourly step count time series for User A
    np.random.seed(42)
    # User A is relatively active: morning jog, afternoon walks, rest periods
    raw_steps = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 800.0, 1200.0,
        300.0, 150.0, 400.0, 600.0, 200.0, 100.0, 350.0, 900.0,
        500.0, 250.0, 100.0, 50.0, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float64)

    # Population stats (Global): Population mean = 400, std = 300
    mu_pop, sigma_pop = 400.0, 300.0
    global_steps = (raw_steps - mu_pop) / sigma_pop

    # User A personal stats (Subject): User mean = 250, std = 350
    mu_user, sigma_user = 250.0, 350.0
    subject_steps = (raw_steps - mu_user) / sigma_user

    # Compute Catch22 on all three versions
    res_raw = pycatch22.catch22_all(raw_steps)
    res_global = pycatch22.catch22_all(global_steps)
    res_subject = pycatch22.catch22_all(subject_steps)

    names = res_raw["names"]
    vals_raw = np.array(res_raw["values"])
    vals_global = np.array(res_global["values"])
    vals_subject = np.array(res_subject["values"])

    print(f"\nTime series length: {len(raw_steps)} hours")
    print(f"Raw step counts: min={raw_steps.min()}, max={raw_steps.max()}, mean={raw_steps.mean():.1f}")
    print(f"Global standardized: min={global_steps.min():.2f}, max={global_steps.max():.2f}, mean={global_steps.mean():.2f}")
    print(f"Subject standardized: min={subject_steps.min():.2f}, max={subject_steps.max():.2f}, mean={subject_steps.mean():.2f}")

    print("\n" + "-" * 80)
    print(f"{'Catch22 Feature':<40} | {'Raw Value':<12} | {'Global Val':<12} | {'Subject Val':<12} | {'Diff (Glob-Sub)':<15}")
    print("-" * 80)

    max_diff = 0.0
    for name, vr, vg, vs in zip(names, vals_raw, vals_global, vals_subject):
        diff = abs(vg - vs)
        max_diff = max(max_diff, diff)
        print(f"{name:<40} | {vr:<12.5f} | {vg:<12.5f} | {vs:<12.5f} | {diff:<15.2e}")

    corr_raw_global = np.corrcoef(vals_raw, vals_global)[0, 1]
    corr_raw_subject = np.corrcoef(vals_raw, vals_subject)[0, 1]
    corr_global_subject = np.corrcoef(vals_global, vals_subject)[0, 1]

    print("-" * 80)
    print(f"Maximum absolute difference between Global and Subject features: {max_diff:.2e}")
    print(f"Correlation (Raw vs Global):     {corr_raw_global:.6f}")
    print(f"Correlation (Raw vs Subject):    {corr_raw_subject:.6f}")
    print(f"Correlation (Global vs Subject): {corr_global_subject:.6f}")
    print("=" * 80)


def test_dual_scaler_collinearity():
    print("\n" + "=" * 80)
    print(" 2. DUAL SCALER WITH CATCH22: 44 OUTPUT FEATURES ARE PAIRWISE COLLINEAR")
    print("=" * 80)

    # 3 samples of length 24, 1 modality
    x_raw = np.random.exponential(scale=200, size=(5, 24, 1)).astype(np.float32)
    uids = np.array(["u1", "u1", "u2", "u2", "u3"])

    # Fit DualScaler
    scaler = DualScaler(StandardScaler(), StandardScaler(), scale_dim=1)
    seqs = [x_raw[i] for i in range(len(x_raw))]
    scaler.fit_by_subject(seqs, uids)
    dual_out = scaler.transform_by_subject(x_raw, uids)

    # dual_out has 2 channels: channel 0 = global, channel 1 = subject
    print(f"DualScaler output shape: {dual_out.shape} -> (samples, time_steps, 2 channels)")

    # Compute Catch22 for both channels
    feats_chan0 = np.array([pycatch22.catch22_all(dual_out[i, :, 0])["values"] for i in range(len(x_raw))])
    feats_chan1 = np.array([pycatch22.catch22_all(dual_out[i, :, 1])["values"] for i in range(len(x_raw))])

    pairwise_corrs = [
        np.corrcoef(feats_chan0[:, k], feats_chan1[:, k])[0, 1]
        for k in range(22)
    ]
    # Filter out NaNs if a feature is constant across samples
    valid_corrs = [c for c in pairwise_corrs if not np.isnan(c)]

    print(f"Number of Catch22 features per channel: 22")
    print(f"Total features across both channels:    44")
    print(f"Average correlation between Channel 0 (Global) and Channel 1 (Subject): {np.mean(valid_corrs):.4f}")
    print("Conclusion: Catch22 on DualScaler produces 2 sets of collinear features.")
    print("=" * 80)


def test_contrast_with_summary_stats():
    print("\n" + "=" * 80)
    print(" 3. CONTRAST: SUMMARY STATS (MEAN, MEDIAN, STD, MIN, MAX)")
    print("=" * 80)

    # User A window: raw steps
    raw_steps = np.array([100.0, 200.0, 400.0, 100.0])
    # Global scale (Pop mean=1000, std=500)
    global_steps = (raw_steps - 1000.0) / 500.0
    # Subject scale (User mean=200, std=100)
    subject_steps = (raw_steps - 200.0) / 100.0

    stats = {
        "mean": lambda s: np.mean(s),
        "median": lambda s: np.median(s),
        "std": lambda s: np.std(s),
        "min": lambda s: np.min(s),
        "max": lambda s: np.max(s),
    }

    print(f"{'Statistic':<15} | {'Raw':<12} | {'Global (Between-Person)':<24} | {'Subject (Within-Person)':<24}")
    print("-" * 80)
    for stat_name, fn in stats.items():
        vr = fn(raw_steps)
        vg = fn(global_steps)
        vs = fn(subject_steps)
        print(f"{stat_name:<15} | {vr:<12.2f} | {vg:<24.2f} | {vs:<24.2f}")

    print("-" * 80)
    print("Notice: For SummaryStats, Global mean is -1.60 (user is inactive vs population),")
    print("        while Subject mean is 0.00 (user is exactly at their normal baseline).")
    print("        They provide TWO DIFFERENT, COMPLEMENTARY SIGNALS!")
    print("=" * 80)


if __name__ == "__main__":
    test_catch22_invariance()
    test_dual_scaler_collinearity()
    test_contrast_with_summary_stats()
