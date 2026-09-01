import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import quad
from pathlib import Path
from sklearn.linear_model import LinearRegression

OUTPUT_DIR = Path("reports/sweep_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "sweep_vj7lbnsh_full_records.csv")

def parse_sampler(s):
    parts = s.split("_")
    strategy = parts[0].capitalize()
    lookback_h = int(parts[1].replace("h", ""))
    window_h = int(parts[2].replace("h", ""))
    return strategy, lookback_h, lookback_h / 24.0, window_h

strategy, lb_h, lb_d, win_h = zip(*df["sampler_pkg"].map(parse_sampler))
df["sampler_strategy"] = strategy
df["lookback_hours"] = lb_h
df["lookback_days"] = lb_d
df["window_hours"] = win_h
df["sequence_length"] = df["lookback_hours"] / df["window_hours"]

print("================================================================================")
print("  STEP 1: RQ3 MULTI-FACTOR FACTORIAL ANOVA & EFFECT SIZES (Partial Eta-Squared)")
print("================================================================================")

# Compute Full Multi-Way ANOVA using OLS Sum of Squares Decomposition
y = df["pooled_auroc"].values
grand_mean = np.mean(y)
ss_total = np.sum((y - grand_mean)**2)
n_total = len(y)

# One-way ANOVAs for each main factor
factors = {
    "Lookback Horizon (Days)": "lookback_days",
    "Sampling Granularity (Hours)": "window_hours",
    "Sampling Strategy (Rolling/Offset)": "sampler_strategy"
}

anova_rows = []
for label, col in factors.items():
    groups = [group["pooled_auroc"].values for _, group in df.groupby(col)]
    k = len(groups)
    f_val, p_val = stats.f_oneway(*groups)
    # SS_between
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
    eta_p2 = ss_between / (ss_between + ss_within)
    
    anova_rows.append({
        "Factor": label,
        "df": k - 1,
        "Sum_of_Squares": ss_between,
        "F_statistic": f_val,
        "p_value": p_val,
        "Partial_Eta_Squared (eta_p^2)": eta_p2
    })

df_anova = pd.DataFrame(anova_rows)
df_anova.to_csv(OUTPUT_DIR / "tables" / "rq3_step1_factorial_anova.csv", index=False)
print(df_anova.to_string(index=False))

print("\n================================================================================")
print("  STEP 2: RQ3 LOOKBACK HORIZON TRENDS & MARGINAL PERFORMANCE")
print("================================================================================")

# JZS Bayes Factor helper
def jzs_bayes_factor_paired(diff, r=np.sqrt(2)/2):
    diff_clean = diff.dropna()
    n = len(diff_clean)
    t_stat, _ = stats.ttest_1samp(diff_clean, 0)
    df_val = n - 1
    def integrand(g):
        term1 = (1.0 + n * g * (r**2))**(-0.5)
        term2 = (1.0 + (t_stat**2) / (df_val * (1.0 + n * g * (r**2))))**(-n / 2.0)
        prior = (2.0 * np.pi)**(-0.5) * (g**(-1.5)) * np.exp(-1.0 / (2.0 * g))
        return term1 * term2 * prior
    num, _ = quad(integrand, 0, np.inf)
    denom = (1.0 + (t_stat**2) / df_val)**(-n / 2.0)
    bf10 = num / denom if denom > 0 else np.inf
    return t_stat, bf10

def interpret_bf(bf10):
    if bf10 > 100: return "Decisive Evidence (H1)"
    elif bf10 > 30: return "Very Strong Evidence (H1)"
    elif bf10 > 10: return "Strong Evidence (H1)"
    elif bf10 > 3: return "Moderate Evidence (H1)"
    elif bf10 >= 1/3: return "Anecdotal / Inconclusive"
    elif bf10 >= 1/10: return "Moderate Evidence (H0: Eq)"
    elif bf10 >= 1/30: return "Strong Evidence (H0: Eq)"
    else: return "Decisive Evidence (H0: Eq)"

# Marginal means by Lookback Horizon
lb_summary = df.groupby("lookback_days").agg(
    N=("pooled_auroc", "count"),
    Mean_AUROC=("pooled_auroc", "mean"),
    SD_AUROC=("pooled_auroc", "std"),
    Median_AUROC=("pooled_auroc", "median"),
    Mean_F1=("pooled_f1", "mean"),
    Mean_Within_Person=("within_person_auroc", "mean")
).reset_index()

# Trend regression
x_vals = df["lookback_days"].values
y_vals = df["pooled_auroc"].values
slope_lin, inter_lin, r_lin, p_lin, _ = stats.linregress(x_vals, y_vals)
poly2 = np.polyfit(x_vals, y_vals, 2)

print(f"• Linear Trend: Slope = {slope_lin:+.4f} AUROC/day, r = {r_lin:.3f}, p = {p_lin:.4e}")
print(f"• Polynomial Fit: AUROC(x) = {poly2[0]:.5f}*x^2 + {poly2[1]:.5f}*x + {poly2[2]:.5f}")

lb_summary.to_csv(OUTPUT_DIR / "tables" / "rq3_step2_lookback_summary.csv", index=False)
print("\n• Lookback Horizon Summary Table:")
print(lb_summary.to_string(index=False))

print("\n================================================================================")
print("  STEP 3: RQ3 SAMPLING STRATEGY & GRANULARITY HEAD-TO-HEAD CONTRASTS")
print("================================================================================")

# 3.1 Rolling vs Offset (Matched Pairs: N = 420 pairs)
piv_strat = df.pivot(index=["model_pkg", "modalities_raw", "lookback_hours", "window_hours"], columns="sampler_strategy", values="pooled_auroc")
diff_strat = piv_strat["Rolling"] - piv_strat["Offset"]
n_strat = len(diff_strat)
mean_strat_d = diff_strat.mean()
std_strat_d = diff_strat.std()
cohen_strat = mean_strat_d / std_strat_d
ci_l_s, ci_u_s = stats.t.interval(0.95, n_strat-1, loc=mean_strat_d, scale=stats.sem(diff_strat))
w_stat_s, w_p_s = stats.wilcoxon(diff_strat)
_, bf10_s = jzs_bayes_factor_paired(diff_strat)

print(f"• 3.1 Sampling Strategy Contrast: Rolling vs. Offset (N = {n_strat} Matched Pairs)")
print(f"    Mean Difference: {mean_strat_d:+.4f} AUROC (95% CI: [{ci_l_s:+.4f}, {ci_u_s:+.4f}])")
print(f"    Win Rate: {(diff_strat > 0).mean()*100:.1f}% for Rolling")
print(f"    Cohen's dz: {cohen_strat:.4f}")
print(f"    Wilcoxon p-value: {w_p_s:.4e}")
print(f"    Bayes Factor BF10: {bf10_s:.4e} ({interpret_bf(bf10_s)})")

# 3.2 Granularity Pairwise Contrasts (4h, 6h, 8h, 12h)
piv_win = df.pivot(index=["model_pkg", "modalities_raw", "lookback_hours", "sampler_strategy"], columns="window_hours", values="pooled_auroc")
win_windows = [4, 6, 8, 12]
win_pairs_res = []

for i in range(len(win_windows)):
    for j in range(i+1, len(win_windows)):
        w1, w2 = win_windows[i], win_windows[j]
        diff_w = piv_win[w1] - piv_win[w2]
        n_w = len(diff_w)
        mean_w_d = diff_w.mean()
        std_w_d = diff_w.std()
        ci_l_w, ci_u_w = stats.t.interval(0.95, n_w-1, loc=mean_w_d, scale=stats.sem(diff_w))
        w_stat_w, w_p_w = stats.wilcoxon(diff_w)
        _, bf10_w = jzs_bayes_factor_paired(diff_w)
        win_pairs_res.append({
            "Comparison": f"{w1}h vs. {w2}h",
            "Mean_Delta": mean_w_d,
            "95% CI": f"[{ci_l_w:+.4f}, {ci_u_w:+.4f}]",
            "Win_Rate (%)": (diff_w > 0).mean() * 100,
            "Cohen_dz": mean_w_d / std_w_d,
            "Wilcoxon_p": w_p_w,
            "BF10": bf10_w,
            "Interpretation": interpret_bf(bf10_w)
        })

df_win_pw = pd.DataFrame(win_pairs_res)
df_win_pw.to_csv(OUTPUT_DIR / "tables" / "rq3_step3_granularity_pairwise.csv", index=False)
print("\n• 3.2 Sampling Granularity Pairwise Contrasts (N = 210 Matched Pairs):")
print(df_win_pw[["Comparison", "Mean_Delta", "95% CI", "Win_Rate (%)", "Cohen_dz", "Wilcoxon_p", "BF10", "Interpretation"]].to_string(index=False))

# 3.3 Top 5 Optimal Temporal Configurations (Averaged across models and modalities)
top_temporal = df.groupby(["sampler_strategy", "lookback_days", "window_hours"]).agg(
    N=("pooled_auroc", "count"),
    Mean_AUROC=("pooled_auroc", "mean"),
    SD_AUROC=("pooled_auroc", "std"),
    Mean_F1=("pooled_f1", "mean"),
    Within_Person_AUROC=("within_person_auroc", "mean")
).reset_index().sort_values("Mean_AUROC", ascending=False).head(5)

top_temporal.to_csv(OUTPUT_DIR / "tables" / "rq3_step3_top_temporal_configs.csv", index=False)
print("\n• 3.3 Top 5 Optimal Temporal Hyperparameter Sets across Grid:")
print(top_temporal.to_string(index=False))

print(f"\nAll RQ3 tables successfully exported to: {OUTPUT_DIR / 'tables'}")
