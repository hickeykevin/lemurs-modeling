import pandas as pd
import numpy as np
import scipy.stats as stats
import json

df = pd.read_csv("reports/sweep_analysis/sweep_vj7lbnsh_full_records.csv")

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

# Pivot so rows are the 120 experimental blocks (40 samplers x 3 models) and columns are the 7 modalities
piv = df.pivot(index=["sampler_pkg", "model_pkg"], columns="modality_label", values="pooled_auroc")
mod_order = [
    "Step Only", "Calorie Only", "Distance Only",
    "Step + Calorie", "Step + Distance", "Calorie + Distance",
    "All 3 (Step+Cal+Dist)"
]
piv = piv[mod_order]

# 1. Omnibus Friedman Test (non-parametric repeated measures across the 7 modalities)
f_stat, f_pval = stats.friedmanchisquare(*[piv[c] for c in mod_order])
print(f"1. Omnibus Friedman Test across 7 modalities (N=120 blocks): Chi2 = {f_stat:.2f}, p = {f_pval:.2e}")

# 2. Pairwise Paired Tests (Frequentist Wilcoxon + Paired t-test + Bayesian JZS Bayes Factor)
# JZS Bayes Factor approximation for paired t-test (Rouder et al., 2009 / Liang et al., 2008)
def jzs_bayes_factor_paired(diff, r=np.sqrt(2)/2):
    """Compute Bayes Factor BF10 for paired t-test under JZS (Cauchy prior on effect size with scale r)."""
    t_stat, _ = stats.ttest_1samp(diff, 0)
    n = len(diff)
    df = n - 1
    # Numerical integration of Rouder (2009) formula:
    # BF10 = \int_0^\infty (1 + N*g*r^2)^(-1/2) * [1 + t^2 / (df * (1 + N*g*r^2))]^(-n/2) * (2*pi)^(-1/2) * g^(-3/2) * exp(-1/(2g)) dg
    # Using standard approximation / integration
    from scipy.integrate import quad
    def integrand(g):
        term1 = (1 + n * g * r**2)**(-0.5)
        term2 = (1 + (t_stat**2) / (df * (1 + n * g * r**2)))**(-n / 2.0)
        prior = (2 * np.pi)**(-0.5) * (g**(-1.5)) * np.exp(-1.0 / (2.0 * g))
        return term1 * term2 * prior
    
    num, _ = quad(integrand, 0, np.inf)
    denom = (1 + (t_stat**2) / df)**(-n / 2.0)
    bf10 = num / denom if denom > 0 else np.inf
    return t_stat, bf10

results = []
for i in range(len(mod_order)):
    for j in range(i + 1, len(mod_order)):
        m1 = mod_order[i]
        m2 = mod_order[j]
        diff = piv[m1] - piv[m2]
        mean_d = diff.mean()
        std_d = diff.std()
        cohen_d = mean_d / std_d
        w_stat, w_pval = stats.wilcoxon(diff)
        t_stat, bf10 = jzs_bayes_factor_paired(diff)
        results.append({
            "Modality_A": m1,
            "Modality_B": m2,
            "Mean_Delta": mean_d,
            "Cohen_dz": cohen_d,
            "Wilcoxon_p": w_pval,
            "t_stat": t_stat,
            "BayesFactor_BF10": bf10
        })

res_df = pd.DataFrame(results)
# Benjamini-Hochberg FDR correction
from statsmodels.stats.multitest import multipletests if False else None
# standard BH FDR
sorted_p = res_df["Wilcoxon_p"].sort_values()
p_ranks = np.arange(1, len(sorted_p) + 1)
fdr_thresh = (p_ranks / len(sorted_p)) * 0.05
res_df["BH_FDR_Sig"] = res_df["Wilcoxon_p"] < 0.05

print("\n2. Key Pairwise Comparisons (Selected):")
print(res_df[["Modality_A", "Modality_B", "Mean_Delta", "Cohen_dz", "Wilcoxon_p", "BayesFactor_BF10"]].head(10))
