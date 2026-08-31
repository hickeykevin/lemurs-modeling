import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import quad
from pathlib import Path

# Setup paths
OUTPUT_DIR = Path("reports/sweep_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

# Load full 840-run dataset
df = pd.read_csv(OUTPUT_DIR / "sweep_vj7lbnsh_full_records.csv")

def format_modalities(m_str):
    m_list = json.loads(m_str) if isinstance(m_str, str) else m_str
    m_set = set(m_list)
    if m_set == {"step"}:
        return "Step Only"
    elif m_set == {"calorie"}:
        return "Calorie Only"
    elif m_set == {"distance"}:
        return "Distance Only"
    elif m_set == {"step", "calorie"}:
        return "Step + Calorie"
    elif m_set == {"step", "distance"}:
        return "Step + Distance"
    elif m_set == {"calorie", "distance"}:
        return "Calorie + Distance"
    elif m_set == {"step", "calorie", "distance"}:
        return "All 3 (Step+Cal+Dist)"
    return "Other"

df["modality_label"] = df["modalities_raw"].map(format_modalities)

# Pivot so rows are the 120 matched blocks (40 samplers x 3 models) and columns are the 7 modalities
piv = df.pivot(index=["sampler_pkg", "model_pkg"], columns="modality_label", values="pooled_auroc")
mod_order = [
    "Step Only", "Calorie Only", "Distance Only",
    "Step + Calorie", "Step + Distance", "Calorie + Distance",
    "All 3 (Step+Cal+Dist)"
]
piv = piv[mod_order]

print("================================================================================")
print("  STEP 1: OMNIBUS MODALITY TESTS (N = 120 Matched Experimental Blocks)")
print("================================================================================")

# 1.1 Friedman Omnibus Test
f_stat, f_pval = stats.friedmanchisquare(*[piv[col] for col in mod_order])
print(f"• Non-Parametric Friedman Test:")
print(f"    Chi2(6) = {f_stat:.4f}, p-value = {f_pval:.4e}")
if f_pval < 0.001:
    print("    --> Significant omnibus difference among modalities across matched blocks (p < 0.001).")

# 1.2 Baseline Predictiveness vs Chance (0.50)
baseline_res = []
for m in mod_order:
    vals = piv[m]
    t_stat, p_val = stats.ttest_1samp(vals, 0.50)
    w_stat, w_val = stats.wilcoxon(vals - 0.50)
    mean_v = vals.mean()
    ci_l, ci_u = stats.t.interval(0.95, len(vals)-1, loc=mean_v, scale=stats.sem(vals))
    baseline_res.append({
        "Modality": m,
        "Mean_AUROC": mean_v,
        "95% CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
        "SD": vals.std(),
        "t_stat": t_stat,
        "p_val_vs_chance": p_val
    })
df_baseline = pd.DataFrame(baseline_res)
df_baseline.to_csv(OUTPUT_DIR / "tables" / "rq1_step1_baseline_predictiveness.csv", index=False)
print("\n• Baseline Predictiveness against Chance (0.50):")
print(df_baseline[["Modality", "Mean_AUROC", "95% CI", "t_stat", "p_val_vs_chance"]].to_string(index=False))

print("\n================================================================================")
print("  STEP 2: POST-HOC PAIRWISE COMPARISONS (Wilcoxon + BH-FDR + JZS Bayes Factor)")
print("================================================================================")

def jzs_bayes_factor_paired(diff, r=np.sqrt(2)/2):
    """
    Compute JZS Bayes Factor (BF10) for paired t-test (Rouder et al., 2009).
    r = sqrt(2)/2 = 0.707 (standard default medium scale Cauchy prior on effect size delta).
    """
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
    if bf10 > 100:
        return "Decisive Evidence (H1)"
    elif bf10 > 30:
        return "Very Strong Evidence (H1)"
    elif bf10 > 10:
        return "Strong Evidence (H1)"
    elif bf10 > 3:
        return "Moderate Evidence (H1)"
    elif bf10 >= 1/3:
        return "Anecdotal / Inconclusive"
    elif bf10 >= 1/10:
        return "Moderate Evidence (H0: Eq)"
    elif bf10 >= 1/30:
        return "Strong Evidence (H0: Eq)"
    else:
        return "Decisive Evidence (H0: Eq)"

pairwise_results = []
for i in range(len(mod_order)):
    for j in range(i + 1, len(mod_order)):
        m1 = mod_order[i]
        m2 = mod_order[j]
        diff = piv[m1] - piv[m2]
        n = len(diff)
        mean_d = diff.mean()
        std_d = diff.std()
        cohen_dz = mean_d / std_d
        ci_l, ci_u = stats.t.interval(0.95, n - 1, loc=mean_d, scale=stats.sem(diff))
        
        w_stat, w_pval = stats.wilcoxon(diff)
        t_stat, bf10 = jzs_bayes_factor_paired(diff)
        
        pairwise_results.append({
            "Comparison": f"{m1} vs. {m2}",
            "Mean_Delta": mean_d,
            "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
            "Cohen_dz": cohen_dz,
            "Wilcoxon_W": w_stat,
            "Wilcoxon_p": w_pval,
            "t_stat": t_stat,
            "BF10": bf10,
            "Interpretation": interpret_bf(bf10)
        })

df_pw = pd.DataFrame(pairwise_results)

# Benjamini-Hochberg FDR correction
sorted_indices = df_pw["Wilcoxon_p"].argsort()
p_vals_sorted = df_pw["Wilcoxon_p"].iloc[sorted_indices].values
m_tests = len(df_pw)
q_values = np.zeros(m_tests)
q_values[-1] = p_vals_sorted[-1]
for k in range(m_tests - 2, -1, -1):
    q_values[k] = min(q_values[k+1], p_vals_sorted[k] * m_tests / (k + 1))
q_values = np.minimum(q_values, 1.0)
df_pw.loc[df_pw.index[sorted_indices], "BH_FDR_q"] = q_values

df_pw.to_csv(OUTPUT_DIR / "tables" / "rq1_step2_pairwise_contrasts.csv", index=False)

# Display Key Pairwise Comparisons
print("\n• Key Post-Hoc Contrasts Table (Selected Illustrative Pairs):")
display_cols = ["Comparison", "Mean_Delta", "95% CI", "Cohen_dz", "Wilcoxon_p", "BH_FDR_q", "BF10", "Interpretation"]
print(df_pw[display_cols].to_string(index=False))

print("\n================================================================================")
print("  STEP 3: MULTIMODAL SYNERGY VS. REDUNDANCY ANALYSIS (Marginal Delta)")
print("================================================================================")

synergy_records = []

# 3.1 Pairwise combinations vs best of their constituent single modalities
pairs_dict = {
    "Step + Distance": ("Step Only", "Distance Only"),
    "Step + Calorie": ("Step Only", "Calorie Only"),
    "Calorie + Distance": ("Calorie Only", "Distance Only")
}

for pair_name, (sing1, sing2) in pairs_dict.items():
    pair_scores = piv[pair_name]
    best_single = np.maximum(piv[sing1], piv[sing2])
    marginal_delta = pair_scores - best_single
    
    mean_gain = marginal_delta.mean()
    std_gain = marginal_delta.std()
    cohen_dz = mean_gain / std_gain
    ci_l, ci_u = stats.t.interval(0.95, len(marginal_delta)-1, loc=mean_gain, scale=stats.sem(marginal_delta))
    w_stat, w_p = stats.wilcoxon(marginal_delta)
    _, bf10 = jzs_bayes_factor_paired(marginal_delta)
    
    synergy_records.append({
        "Level": "Single -> Pair",
        "Transition": f"Max({sing1}, {sing2}) -> {pair_name}",
        "Marginal_Mean_Delta": mean_gain,
        "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
        "Cohen_dz": cohen_dz,
        "Wilcoxon_p": w_p,
        "BF10": bf10,
        "Synergy_Verdict": "Synergistic (Gain > 0)" if (mean_gain > 0 and w_p < 0.05) else "Redundant / Diminishing"
    })

# 3.2 Trivariate combination vs best of the 3 pairs
best_pair_scores = np.maximum(np.maximum(piv["Step + Distance"], piv["Step + Calorie"]), piv["Calorie + Distance"])
trivariate_delta = piv["All 3 (Step+Cal+Dist)"] - best_pair_scores

mean_gain = trivariate_delta.mean()
std_gain = trivariate_delta.std()
cohen_dz = mean_gain / std_gain
ci_l, ci_u = stats.t.interval(0.95, len(trivariate_delta)-1, loc=mean_gain, scale=stats.sem(trivariate_delta))
w_stat, w_p = stats.wilcoxon(trivariate_delta)
_, bf10 = jzs_bayes_factor_paired(trivariate_delta)

synergy_records.append({
    "Level": "Pair -> Trio",
    "Transition": "Max(All Pairs) -> All 3 Modalities",
    "Marginal_Mean_Delta": mean_gain,
    "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
    "Cohen_dz": cohen_dz,
    "Wilcoxon_p": w_p,
    "BF10": bf10,
    "Synergy_Verdict": "Synergistic (Gain > 0)" if (mean_gain > 0 and w_p < 0.05) else "Negative Synergy / Noise Penalty"
})

df_synergy = pd.DataFrame(synergy_records)
df_synergy.to_csv(OUTPUT_DIR / "tables" / "rq1_step3_synergy_analysis.csv", index=False)
print("\n• Stepwise Marginal Gains & Synergy Table:")
print(df_synergy.to_string(index=False))

# Export LaTeX formatted table for paper inclusion
latex_pw = df_pw.head(8).to_latex(
    index=False,
    float_format="%.4f",
    caption="Post-hoc pairwise contrasts for RQ1 showing Frequentist and Bayesian JZS Bayes Factors across 120 matched conditions.",
    label="tab:rq1_pairwise_contrasts"
)
with open(OUTPUT_DIR / "tables" / "rq1_table_paper.tex", "w") as f:
    f.write(latex_pw)

print(f"\nAll RQ1 test tables successfully exported to: {OUTPUT_DIR / 'tables'}")
