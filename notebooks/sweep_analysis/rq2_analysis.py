import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import quad
from pathlib import Path

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

model_map = {"default": "LSTM", "transformer": "Transformer", "flaml_xgboost": "FLAML XGBoost"}
df["model_name"] = df["model_pkg"].map(model_map)

# Pivot table: rows are the 280 matched experimental conditions (40 samplers x 7 modalities), columns are the 3 models
piv_m = df.pivot(index=["sampler_pkg", "modalities_raw"], columns="model_name", values="pooled_auroc")
models = ["LSTM", "Transformer", "FLAML XGBoost"]
piv_m = piv_m[models]

print("================================================================================")
print("  STEP 1: RQ2 OMNIBUS MODEL ARCHITECTURE TESTS (N = 280 Matched Blocks)")
print("================================================================================")

# 1.1 Friedman Test
f_stat, f_pval = stats.friedmanchisquare(piv_m["LSTM"], piv_m["Transformer"], piv_m["FLAML XGBoost"])
# Kendall's W = chi2 / (N * (k - 1))
k = 3
n_blocks = len(piv_m)
kendall_w = f_stat / (n_blocks * (k - 1))

print(f"• Non-Parametric Friedman Test:")
print(f"    Chi2(2) = {f_stat:.4f}, p-value = {f_pval:.4e}")
print(f"    Kendall's W (Effect Size) = {kendall_w:.4f} (Strong Concordance)")

# 1.2 Model Summary Statistics across all 280 conditions
model_summary = []
for m in models:
    vals = piv_m[m]
    ci_l, ci_u = stats.t.interval(0.95, len(vals)-1, loc=vals.mean(), scale=stats.sem(vals))
    model_summary.append({
        "Model": m,
        "Mean_AUROC": vals.mean(),
        "95% CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
        "SD": vals.std(),
        "Median_AUROC": vals.median(),
        "Min_AUROC": vals.min(),
        "Max_AUROC": vals.max()
    })
df_ms = pd.DataFrame(model_summary)
df_ms.to_csv(OUTPUT_DIR / "tables" / "rq2_step1_model_summary.csv", index=False)
print("\n• Global Model Summary across 280 Conditions:")
print(df_ms.to_string(index=False))

print("\n================================================================================")
print("  STEP 2: RQ2 PAIRWISE HEAD-TO-HEAD CONTRASTS (Wilcoxon + JZS Bayes Factors)")
print("================================================================================")

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

rq2_pairs = [
    ("LSTM", "FLAML XGBoost", "LSTM vs. FLAML XGBoost"),
    ("Transformer", "FLAML XGBoost", "Transformer vs. FLAML XGBoost"),
    ("Transformer", "LSTM", "Transformer vs. LSTM")
]

rq2_pairwise_res = []
for m1, m2, label in rq2_pairs:
    diff = piv_m[m1] - piv_m[m2]
    n = len(diff)
    mean_d = diff.mean()
    std_d = diff.std()
    cohen_dz = mean_d / std_d
    ci_l, ci_u = stats.t.interval(0.95, n - 1, loc=mean_d, scale=stats.sem(diff))
    
    w_stat, w_pval = stats.wilcoxon(diff)
    t_stat, bf10 = jzs_bayes_factor_paired(diff)
    win_rate = (diff > 0).mean() * 100
    
    rq2_pairwise_res.append({
        "Comparison": label,
        "N_Pairs": n,
        "Mean_Delta": mean_d,
        "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
        "Win_Rate (%)": win_rate,
        "Cohen_dz": cohen_dz,
        "t_stat": t_stat,
        "Wilcoxon_p": w_pval,
        "BF10": bf10,
        "BF01": 1.0 / bf10 if bf10 > 0 else np.nan,
        "Interpretation": interpret_bf(bf10)
    })

df_rq2_pw = pd.DataFrame(rq2_pairwise_res)
df_rq2_pw.to_csv(OUTPUT_DIR / "tables" / "rq2_step2_pairwise_contrasts.csv", index=False)
print("\n• Pairwise Head-to-Head Table:")
print(df_rq2_pw[["Comparison", "Mean_Delta", "95% CI", "Win_Rate (%)", "Cohen_dz", "Wilcoxon_p", "BF10", "Interpretation"]].to_string(index=False))

print("\n================================================================================")
print("  STEP 3: RQ2 HORIZON-DEPENDENT ADVANTAGE (Moderation / Interaction Analysis)")
print("================================================================================")

# Join lookback days onto paired table
lb_map = df[["sampler_pkg", "lookback_days"]].drop_duplicates().set_index("sampler_pkg")["lookback_days"]
piv_m_reset = piv_m.reset_index()
piv_m_reset["lookback_days"] = piv_m_reset["sampler_pkg"].map(lb_map)
piv_m_reset["Delta_Sequence_minus_Tree"] = ((piv_m_reset["LSTM"] + piv_m_reset["Transformer"]) / 2.0) - piv_m_reset["FLAML XGBoost"]

horizon_grp = piv_m_reset.groupby("lookback_days").agg(
    N=("Delta_Sequence_minus_Tree", "count"),
    Mean_Advantage=("Delta_Sequence_minus_Tree", "mean"),
    SD_Advantage=("Delta_Sequence_minus_Tree", "std"),
    LSTM_Mean=("LSTM", "mean"),
    Transformer_Mean=("Transformer", "mean"),
    XGB_Mean=("FLAML XGBoost", "mean")
).reset_index()

# Trend test: Linear regression of sequence advantage vs lookback days
slope, intercept, r_val, p_val, std_err = stats.linregress(piv_m_reset["lookback_days"], piv_m_reset["Delta_Sequence_minus_Tree"])
print(f"• Linear Trend of Sequence Advantage (Delta AUROC) vs Lookback Horizon:")
print(f"    Slope: {slope:+.4f} AUROC per Day, r = {r_val:.3f}, p-value = {p_val:.4e}")

# ANOVA across horizons
f_horiz, p_horiz = stats.f_oneway(*[group["Delta_Sequence_minus_Tree"].values for _, group in piv_m_reset.groupby("lookback_days")])
print(f"• One-Way ANOVA of Sequence Advantage across Lookback Horizons (1d to 5d):")
print(f"    F(4, 275) = {f_horiz:.4f}, p-value = {p_horiz:.4e}")

horizon_grp["Linear_Trend_p"] = p_val
horizon_grp.to_csv(OUTPUT_DIR / "tables" / "rq2_step3_horizon_moderation.csv", index=False)
print("\n• Sequence Model Advantage by Lookback Horizon:")
print(horizon_grp[["lookback_days", "N", "Mean_Advantage", "LSTM_Mean", "Transformer_Mean", "XGB_Mean"]].to_string(index=False))

print(f"\nAll RQ2 statistical tests exported to: {OUTPUT_DIR / 'tables'}")
