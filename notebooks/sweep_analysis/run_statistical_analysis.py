import os
import json
import netrc
import shutil
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("reports/sweep_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

PAPER_FIG_DIR = Path("../paper_materials/working_paper/IEEE-conference-template-062824/figures")
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_ID = "r2mhb7wj"
CACHE_FILE = OUTPUT_DIR / f"sweep_{SWEEP_ID}_full_records.csv"

def jzs_bayes_factor_paired(diff, r=np.sqrt(2)/2):
    diff_clean = diff.dropna()
    n = len(diff_clean)
    if n < 2:
        return 0.0, np.nan
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
    if np.isnan(bf10): return "N/A"
    if bf10 > 100: return "Decisive Evidence (H1)"
    elif bf10 > 30: return "Very Strong Evidence (H1)"
    elif bf10 > 10: return "Strong Evidence (H1)"
    elif bf10 > 3: return "Moderate Evidence (H1)"
    elif bf10 >= 1/3: return "Anecdotal / Inconclusive"
    elif bf10 >= 1/10: return "Moderate Evidence (H0: Eq)"
    elif bf10 >= 1/30: return "Strong Evidence (H0: Eq)"
    else: return "Decisive Evidence (H0: Eq)"

def fetch_data():
    if CACHE_FILE.exists():
        print(f"Loading cached records from {CACHE_FILE}...")
        return pd.read_csv(CACHE_FILE)
    
    print(f"Fetching runs from W&B API via GraphQL for sweep {SWEEP_ID}...")
    n = netrc.netrc()
    auth = n.authenticators("api.wandb.ai")
    headers = {"Authorization": f"Bearer {auth[2]}", "Content-Type": "application/json"}
    
    query = """
    query SweepRuns($entity: String!, $project: String!, $name: String!) {
      project(entityName: $entity, name: $project) {
        sweep(sweepName: $name) {
          runs(first: 2000) {
            edges {
              node {
                name
                displayName
                state
                summaryMetrics
                config
                createdAt
              }
            }
          }
        }
      }
    }
    """
    resp = requests.post(
        "https://api.wandb.ai/graphql",
        headers=headers,
        json={
            "query": query,
            "variables": {"entity": "hickeykevin", "project": "lemurs-modeling", "name": SWEEP_ID}
        }
    )
    data = resp.json()
    edges = data["data"]["project"]["sweep"]["runs"]["edges"]
    records = []
    
    for e in edges:
        node = e["node"]
        if node["state"] != "finished":
            continue
        summary = json.loads(node.get("summaryMetrics") or "{}")
        config = json.loads(node.get("config") or "{}")
        if summary.get("pooled/f1") is None:
            continue
        
        model_val = config.get("model_pkg", {}).get("value")
        sampler_val = config.get("sampler_pkg", {}).get("value")
        mod_val = config.get("modalities_choice", {}).get("value")
        
        records.append({
            "run_id": node["name"],
            "run_name": node["displayName"],
            "created_at": node["createdAt"],
            "model_pkg": model_val,
            "sampler_pkg": sampler_val,
            "modalities_raw": json.dumps(mod_val),
            "pooled_auroc": summary.get("pooled/auroc"),
            "pooled_f1": summary.get("pooled/f1"),
            "pooled_precision": summary.get("pooled/precision"),
            "pooled_recall": summary.get("pooled/recall"),
            "pooled_specificity": summary.get("pooled/specificity"),
            "pooled_sensitivity_at_spec": summary.get("pooled/sensitivity_at_specificity"),
            "pooled_balanced_acc": summary.get("pooled/balanced_accuracy"),
            "pooled_auprc": summary.get("pooled/auprc"),
            "within_person_auroc": summary.get("test/within_person_auroc"),
            "n_test_rows": summary.get("cv/n_test_rows"),
        })
    
    df = pd.DataFrame(records)
    df.to_csv(CACHE_FILE, index=False)
    print(f"Cached {len(df)} runs to {CACHE_FILE}")
    return df

def run_analysis():
    df_all = fetch_data()
    print(f"Total raw runs fetched: {len(df_all)}")

    # Factor parsing
    def parse_sampler(s):
        parts = s.split("_")
        strategy = parts[0].capitalize()
        lookback_h = int(parts[1].replace("h", ""))
        window_h = int(parts[2].replace("h", ""))
        return strategy, lookback_h, lookback_h / 24.0, window_h

    strategy, lb_h, lb_d, win_h = zip(*df_all["sampler_pkg"].map(parse_sampler))
    df_all["sampler_strategy"] = strategy
    df_all["lookback_hours"] = lb_h
    df_all["lookback_days"] = lb_d
    df_all["window_hours"] = win_h
    df_all["sequence_length"] = df_all["lookback_hours"] / df_all["window_hours"]

    model_map = {"default": "LSTM", "transformer": "Transformer", "flaml_xgboost": "FLAML XGBoost"}
    df_all["model_name"] = df_all["model_pkg"].map(model_map)

    def format_modalities(m_str):
        m_list = json.loads(m_str) if isinstance(m_str, str) else m_str
        m_set = set(m_list)
        if m_set == {"step"}:
            return "Step Only", 1, True, False, False
        elif m_set == {"calorie"}:
            return "Calorie Only", 1, False, True, False
        elif m_set == {"distance"}:
            return "Distance Only", 1, False, False, True
        elif m_set == {"step", "calorie"}:
            return "Step + Calorie", 2, True, True, False
        elif m_set == {"step", "distance"}:
            return "Step + Distance", 2, True, False, True
        elif m_set == {"calorie", "distance"}:
            return "Calorie + Distance", 2, False, True, True
        elif m_set == {"step", "calorie", "distance"}:
            return "All 3 (Step+Cal+Dist)", 3, True, True, True
        return "Other", len(m_list), False, False, False

    mod_labels, mod_counts, has_step, has_cal, has_dist = zip(*df_all["modalities_raw"].map(format_modalities))
    df_all["modality_label"] = mod_labels
    df_all["n_modalities"] = mod_counts
    df_all["has_step"] = has_step
    df_all["has_calorie"] = has_cal
    df_all["has_distance"] = has_dist

    # Filter to Rolling sampler only as requested
    df = df_all[df_all["sampler_strategy"] == "Rolling"].copy().reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / f"sweep_{SWEEP_ID}_rolling_records.csv", index=False)
    print(f"Filtered to Rolling sampler: {len(df)} runs (20 samplers x 3 models x 7 modalities).")

    # Visual styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # =========================================================================
    # 1. Global Baseline Test & Summary Table
    # =========================================================================
    t_stat_g, p_val_g = stats.ttest_1samp(df["pooled_auroc"], 0.50)
    print("\n================== 1. GLOBAL PREDICTIVENESS (ROLLING ONLY) ==================")
    print(f"Mean AUROC: {df['pooled_auroc'].mean():.4f} +/- {df['pooled_auroc'].std():.4f} (t={t_stat_g:.2f}, p={p_val_g:.2e})")
    print(f"Max AUROC: {df['pooled_auroc'].max():.4f}")

    summary_factors = []
    for factor, name in [
        ("model_name", "Model Architecture"),
        ("modality_label", "Modality Subset"),
        ("lookback_hours", "Lookback Horizon (Hours)"),
        ("window_hours", "Sampling Granularity (Hours)")
    ]:
        grp = df.groupby(factor).agg(
            N=("pooled_auroc", "count"),
            Mean_AUROC=("pooled_auroc", "mean"),
            SD_AUROC=("pooled_auroc", "std"),
            Mean_F1=("pooled_f1", "mean"),
            Mean_Within_Person_AUROC=("within_person_auroc", "mean")
        ).reset_index()
        grp["Factor"] = name
        grp = grp.rename(columns={factor: "Level"})
        summary_factors.append(grp[["Factor", "Level", "N", "Mean_AUROC", "SD_AUROC", "Mean_F1", "Mean_Within_Person_AUROC"]])

    global_table = pd.concat(summary_factors, ignore_index=True)
    global_table.to_csv(OUTPUT_DIR / "tables" / "table1_global_factor_summary.csv", index=False)

    # =========================================================================
    # 2. RQ1: Modality Value & Multimodal Synergy (60 Matched Blocks)
    # =========================================================================
    mod_order = [
        "Step Only", "Calorie Only", "Distance Only",
        "Step + Calorie", "Step + Distance", "Calorie + Distance",
        "All 3 (Step+Cal+Dist)"
    ]
    
    # Table 2: Modality Ablation Summary
    mod_stats = []
    for mod in mod_order:
        sub = df[df["modality_label"] == mod]["pooled_auroc"]
        ci_l, ci_u = stats.t.interval(0.95, len(sub)-1, loc=sub.mean(), scale=stats.sem(sub))
        f1_sub = df[df["modality_label"] == mod]["pooled_f1"]
        wp_sub = df[df["modality_label"] == mod]["within_person_auroc"]
        mod_stats.append({
            "Modality": mod,
            "N": len(sub),
            "AUROC_Mean": sub.mean(),
            "AUROC_SD": sub.std(),
            "AUROC_95CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
            "AUROC_Median": sub.median(),
            "F1_Mean": f1_sub.mean(),
            "WithinPerson_AUROC": wp_sub.mean()
        })
    df_mod_stats = pd.DataFrame(mod_stats).set_index("Modality")
    df_mod_stats.to_csv(OUTPUT_DIR / "tables" / "table2_rq1_modality_ablation.csv")
    print("\n================== 2. RQ1: MODALITY ABLATION ==================")
    print(df_mod_stats)

    # Step 1: Friedman Omnibus Test (N = 60 blocks)
    piv_mod = df.pivot(index=["sampler_pkg", "model_pkg"], columns="modality_label", values="pooled_auroc")[mod_order]
    f_stat_rq1, f_pval_rq1 = stats.friedmanchisquare(*[piv_mod[c] for c in mod_order])
    print(f"\nRQ1 Friedman Omnibus Test: Chi2(6) = {f_stat_rq1:.4f}, p = {f_pval_rq1:.4e} (N = {len(piv_mod)} blocks)")

    # Step 2: Post-Hoc Pairwise Contrasts
    pw_rq1 = []
    for i in range(len(mod_order)):
        for j in range(i + 1, len(mod_order)):
            m1, m2 = mod_order[i], mod_order[j]
            diff = piv_mod[m1] - piv_mod[m2]
            ci_l, ci_u = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))
            w_stat, w_p = stats.wilcoxon(diff)
            t_stat, bf10 = jzs_bayes_factor_paired(diff)
            pw_rq1.append({
                "Comparison": f"{m1} vs. {m2}",
                "N_Pairs": len(diff),
                "Mean_Delta": diff.mean(),
                "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
                "Cohen_dz": diff.mean() / diff.std(),
                "Wilcoxon_p": w_p,
                "BF10": bf10,
                "BF01": 1.0 / bf10 if bf10 > 0 else np.nan,
                "Interpretation": interpret_bf(bf10)
            })
    df_pw_rq1 = pd.DataFrame(pw_rq1)
    sorted_idx = df_pw_rq1["Wilcoxon_p"].argsort()
    p_sorted = df_pw_rq1["Wilcoxon_p"].iloc[sorted_idx].values
    m_k = len(df_pw_rq1)
    q_vals = np.zeros(m_k)
    q_vals[-1] = p_sorted[-1]
    for k in range(m_k - 2, -1, -1):
        q_vals[k] = min(q_vals[k+1], p_sorted[k] * m_k / (k + 1))
    df_pw_rq1.loc[df_pw_rq1.index[sorted_idx], "BH_FDR_q"] = np.minimum(q_vals, 1.0)
    df_pw_rq1.to_csv(OUTPUT_DIR / "tables" / "rq1_pairwise_contrasts.csv", index=False)
    df_pw_rq1.to_csv(OUTPUT_DIR / "tables" / "rq1_step2_pairwise_contrasts.csv", index=False)

    # Step 3: Synergy Table
    synergy_list = []
    for pair, (s1, s2) in {
        "Step + Distance": ("Step Only", "Distance Only"),
        "Step + Calorie": ("Step Only", "Calorie Only"),
        "Calorie + Distance": ("Calorie Only", "Distance Only")
    }.items():
        delta = piv_mod[pair] - np.maximum(piv_mod[s1], piv_mod[s2])
        ci_l, ci_u = stats.t.interval(0.95, len(delta)-1, loc=delta.mean(), scale=stats.sem(delta))
        _, w_p = stats.wilcoxon(delta)
        _, bf10 = jzs_bayes_factor_paired(delta)
        synergy_list.append({
            "Transition": f"Max({s1}, {s2}) -> {pair}",
            "Marginal_Delta": delta.mean(),
            "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
            "Cohen_dz": delta.mean() / delta.std(),
            "Wilcoxon_p": w_p,
            "BF10": bf10,
            "Verdict": "Synergistic (Gain > 0)" if (delta.mean() > 0 and w_p < 0.05) else "Redundant / Diminishing"
        })
    trio_delta = piv_mod["All 3 (Step+Cal+Dist)"] - np.maximum(
        np.maximum(piv_mod["Step + Distance"], piv_mod["Step + Calorie"]),
        piv_mod["Calorie + Distance"]
    )
    ci_l, ci_u = stats.t.interval(0.95, len(trio_delta)-1, loc=trio_delta.mean(), scale=stats.sem(trio_delta))
    _, w_p = stats.wilcoxon(trio_delta)
    _, bf10 = jzs_bayes_factor_paired(trio_delta)
    synergy_list.append({
        "Transition": "Max(All Pairs) -> All 3 Modalities",
        "Marginal_Delta": trio_delta.mean(),
        "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
        "Cohen_dz": trio_delta.mean() / trio_delta.std(),
        "Wilcoxon_p": w_p,
        "BF10": bf10,
        "Verdict": "Synergistic (Gain > 0)" if (trio_delta.mean() > 0 and w_p < 0.05) else "Negative Synergy / Noise Penalty"
    })
    df_syn_rq1 = pd.DataFrame(synergy_list)
    df_syn_rq1.to_csv(OUTPUT_DIR / "tables" / "rq1_synergy_analysis.csv", index=False)
    df_syn_rq1.to_csv(OUTPUT_DIR / "tables" / "rq1_step3_synergy_analysis.csv", index=False)

    # Figure 1: Modality Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    sns.boxplot(
        data=df,
        x="modality_label",
        y="pooled_auroc",
        order=mod_order,
        palette="Blues_d",
        ax=axes[0],
        fliersize=2
    )
    axes[0].set_xticks(range(len(mod_order)))
    axes[0].set_xticklabels(mod_order, rotation=35, ha="right")
    axes[0].set_title("(A) Predictive AUROC across Modality Subsets")
    axes[0].set_ylabel("Pooled AUROC")
    axes[0].set_xlabel("")
    axes[0].axhline(0.50, color="red", linestyle="--", alpha=0.7, label="Chance Baseline")
    axes[0].legend(loc="lower right")

    sns.violinplot(
        data=df,
        x="n_modalities",
        y="pooled_auroc",
        hue="model_name",
        palette="Set2",
        inner="quartile",
        ax=axes[1]
    )
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["1 Modality\n(Single)", "2 Modalities\n(Pair)", "3 Modalities\n(Trivariate)"])
    axes[1].set_title("(B) Modality Expansion by Model Architecture")
    axes[1].set_xlabel("Feature Set Complexity")
    axes[1].set_ylabel("")
    axes[1].legend(title="Model Architecture", loc="lower right")

    plt.tight_layout()
    fig1_pdf = OUTPUT_DIR / "figures" / "fig1_rq1_modality_value.pdf"
    fig1_png = OUTPUT_DIR / "figures" / "fig1_rq1_modality_value.png"
    plt.savefig(fig1_pdf, bbox_inches="tight")
    plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
    shutil.copy(fig1_pdf, PAPER_FIG_DIR / "fig1_rq1_modality_value.pdf")
    shutil.copy(fig1_png, PAPER_FIG_DIR / "fig1_rq1_modality_value.png")
    plt.close()

    # =========================================================================
    # 3. RQ2: Deep Time-Series Models vs. Tree-Based ML (140 Matched Conditions)
    # =========================================================================
    piv_m = df.pivot(index=["sampler_pkg", "modalities_raw"], columns="model_name", values="pooled_auroc")[["LSTM", "Transformer", "FLAML XGBoost"]]
    
    # Model summary table across all metrics
    model_summary = []
    for m_name in ["Transformer", "LSTM", "FLAML XGBoost"]:
        sub_df = df[df["model_name"] == m_name]
        model_summary.append({
            "Model": m_name,
            "N": len(sub_df),
            "AUROC_Mean": sub_df["pooled_auroc"].mean(),
            "AUROC_SD": sub_df["pooled_auroc"].std(),
            "F1_Mean": sub_df["pooled_f1"].mean(),
            "F1_SD": sub_df["pooled_f1"].std(),
            "Sensitivity_Mean": sub_df["pooled_recall"].mean(),
            "Sensitivity_SD": sub_df["pooled_recall"].std(),
            "Specificity_Mean": sub_df["pooled_specificity"].mean(),
            "Specificity_SD": sub_df["pooled_specificity"].std(),
            "Precision_Mean": sub_df["pooled_precision"].mean(),
            "Precision_SD": sub_df["pooled_precision"].std(),
        })
    df_model_summary = pd.DataFrame(model_summary).set_index("Model")
    df_model_summary.to_csv(OUTPUT_DIR / "tables" / "table3_rq2_model_head_to_head.csv")
    df_model_summary.to_csv(OUTPUT_DIR / "tables" / "rq2_step1_model_summary.csv")
    print("\n================== 3. RQ2: MODEL SUMMARY ==================")
    print(df_model_summary)

    # Step 1: Friedman Omnibus Architecture Test (N = 140 conditions)
    f_stat_rq2, f_pval_rq2 = stats.friedmanchisquare(piv_m["LSTM"], piv_m["Transformer"], piv_m["FLAML XGBoost"])
    kendall_w_rq2 = f_stat_rq2 / (len(piv_m) * 2)
    print(f"\nRQ2 Friedman Omnibus Test: Chi2(2) = {f_stat_rq2:.4f}, p = {f_pval_rq2:.4e} (Kendall W = {kendall_w_rq2:.4f}, N = {len(piv_m)})")

    # Step 2: Head-to-Head Pairwise Contrasts
    rq2_pairs = [
        ("LSTM", "FLAML XGBoost", "LSTM vs. FLAML XGBoost"),
        ("Transformer", "FLAML XGBoost", "Transformer vs. FLAML XGBoost"),
        ("Transformer", "LSTM", "Transformer vs. LSTM")
    ]
    rq2_pw_res = []
    for m1, m2, label in rq2_pairs:
        diff = piv_m[m1] - piv_m[m2]
        mean_d = diff.mean()
        std_d = diff.std()
        ci_l, ci_u = stats.t.interval(0.95, len(diff)-1, loc=mean_d, scale=stats.sem(diff))
        w_stat, w_p = stats.wilcoxon(diff)
        t_stat, bf10 = jzs_bayes_factor_paired(diff)
        rq2_pw_res.append({
            "Comparison": label,
            "N_Pairs": len(diff),
            "Mean_Delta": mean_d,
            "95% CI": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
            "Win_Rate (%)": (diff > 0).mean() * 100,
            "Cohen_dz": mean_d / std_d,
            "Wilcoxon_p": w_p,
            "BF10": bf10,
            "BF01": 1.0 / bf10 if bf10 > 0 else np.nan,
            "Interpretation": interpret_bf(bf10)
        })
    df_rq2_pw = pd.DataFrame(rq2_pw_res)
    df_rq2_pw.to_csv(OUTPUT_DIR / "tables" / "rq2_pairwise_contrasts.csv", index=False)
    df_rq2_pw.to_csv(OUTPUT_DIR / "tables" / "rq2_step2_pairwise_contrasts.csv", index=False)
    print("\nRQ2 Pairwise Contrasts:")
    print(df_rq2_pw)

    # Step 3: Horizon Moderation Analysis
    lb_map = df[["sampler_pkg", "lookback_days"]].drop_duplicates().set_index("sampler_pkg")["lookback_days"]
    piv_m_res = piv_m.reset_index()
    piv_m_res["lookback_days"] = piv_m_res["sampler_pkg"].map(lb_map)
    piv_m_res["Seq_Advantage"] = ((piv_m_res["LSTM"] + piv_m_res["Transformer"]) / 2.0) - piv_m_res["FLAML XGBoost"]
    slope_hm, intercept_hm, r_val_hm, p_val_hm, _ = stats.linregress(piv_m_res["lookback_days"], piv_m_res["Seq_Advantage"])
    print(f"\nRQ2 Horizon Moderation Trend: Slope = {slope_hm:+.4f}/day, r = {r_val_hm:.3f}, p = {p_val_hm:.4e}")

    # Figure 2: Model Architecture Comparison (Single Panel across Lookback Windows)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sns.lineplot(
        data=df,
        x="lookback_hours",
        y="pooled_auroc",
        hue="model_name",
        style="model_name",
        markers=True,
        dashes=False,
        palette={"Transformer": "#e41a1c", "LSTM": "#2ca02c", "FLAML XGBoost": "#1f77b4"},
        errorbar=("ci", 95),
        ax=ax
    )
    ax.set_title("AUROC across Lookback Windows", fontsize=12, weight="bold")
    ax.set_xlabel("Historical Lookback Window", fontsize=10)
    ax.set_ylabel("Pooled AUROC (Mean +/- 95% CI)", fontsize=10)
    ax.set_xticks([24, 48, 72, 96, 120])
    ax.set_xticklabels(["24h", "48h", "72h", "96h", "120h"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Model Architecture", loc="lower right", frameon=True)

    plt.tight_layout()
    fig2_pdf = OUTPUT_DIR / "figures" / "fig2_rq2_model_horizon_interaction.pdf"
    fig2_png = OUTPUT_DIR / "figures" / "fig2_rq2_model_horizon_interaction.png"
    plt.savefig(fig2_pdf, bbox_inches="tight")
    plt.savefig(fig2_png, dpi=300, bbox_inches="tight")
    shutil.copy(fig2_pdf, PAPER_FIG_DIR / "fig2_rq2_model_horizon_interaction.pdf")
    shutil.copy(fig2_png, PAPER_FIG_DIR / "fig2_rq2_model_horizon_interaction.png")
    plt.close()

    # =========================================================================
    # 4. RQ3: Temporal Design Choices (Lookback Horizon & Sampling Granularity)
    # =========================================================================
    grand_mean = df["pooled_auroc"].mean()
    factors = {
        "Sampling Granularity (Hours)": "window_hours",
        "Lookback Horizon (Days)": "lookback_days"
    }
    anova_rows = []
    for label, col in factors.items():
        groups = [group["pooled_auroc"].values for _, group in df.groupby(col)]
        f_val, p_val = stats.f_oneway(*groups)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
        anova_rows.append({
            "Factor": label,
            "df": len(groups) - 1,
            "Sum_of_Squares": ss_between,
            "F_statistic": f_val,
            "p_value": p_val,
            "Partial_Eta_Squared (eta_p2)": ss_between / (ss_between + ss_within)
        })
    df_anova_rq3 = pd.DataFrame(anova_rows)
    df_anova_rq3.to_csv(OUTPUT_DIR / "tables" / "table4_rq3_factorial_anova.csv", index=False)
    df_anova_rq3.to_csv(OUTPUT_DIR / "tables" / "rq3_factorial_anova.csv", index=False)
    df_anova_rq3.to_csv(OUTPUT_DIR / "tables" / "rq3_step1_factorial_anova.csv", index=False)
    print("\n================== 4. RQ3: FACTORIAL ANOVA ==================")
    print(df_anova_rq3)

    # Horizon trend
    x_vals = df["lookback_days"].values
    y_vals = df["pooled_auroc"].values
    slope_lb, intercept_lb, r_lb, p_lb, _ = stats.linregress(x_vals, y_vals)
    print(f"\nRQ3 Horizon Trend: Slope = {slope_lb:+.4f} AUROC/day (r = {r_lb:.3f}, p = {p_lb:.4e})")

    # Granularity pairwise contrasts (N = 105 matched conditions: 5 lookbacks x 3 models x 7 modalities)
    piv_win = df.pivot(index=["model_pkg", "modalities_raw", "lookback_hours"], columns="window_hours", values="pooled_auroc")
    win_pairs = []
    for i in range(len([4, 6, 8, 12])):
        for j in range(i+1, len([4, 6, 8, 12])):
            w1, w2 = [4, 6, 8, 12][i], [4, 6, 8, 12][j]
            d_w = piv_win[w1] - piv_win[w2]
            ci_l_w, ci_u_w = stats.t.interval(0.95, len(d_w)-1, loc=d_w.mean(), scale=stats.sem(d_w))
            _, w_p_w = stats.wilcoxon(d_w)
            _, bf10_w = jzs_bayes_factor_paired(d_w)
            win_pairs.append({
                "Comparison": f"{w1}h vs. {w2}h",
                "N_Pairs": len(d_w),
                "Mean_Delta": d_w.mean(),
                "95% CI": f"[{ci_l_w:+.4f}, {ci_u_w:+.4f}]",
                "Win_Rate (%)": (d_w > 0).mean() * 100,
                "Cohen_dz": d_w.mean() / d_w.std(),
                "Wilcoxon_p": w_p_w,
                "BF10": bf10_w,
                "BF01": 1.0 / bf10_w if bf10_w > 0 else np.nan,
                "Interpretation": interpret_bf(bf10_w)
            })
    df_win_pw = pd.DataFrame(win_pairs)
    df_win_pw.to_csv(OUTPUT_DIR / "tables" / "rq3_granularity_pairwise.csv", index=False)
    df_win_pw.to_csv(OUTPUT_DIR / "tables" / "rq3_step3_granularity_pairwise.csv", index=False)
    print("\nRQ3 Granularity Pairwise:")
    print(df_win_pw)

    # Figure 3: 2D Response Surface Heatmap
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    means = df.pivot_table(
        index="lookback_hours",
        columns="window_hours",
        values="pooled_auroc",
        aggfunc="mean"
    )
    stds = df.pivot_table(
        index="lookback_hours",
        columns="window_hours",
        values="pooled_auroc",
        aggfunc="std"
    )
    annot_matrix = means.copy().astype(object)
    for r in means.index:
        for c in means.columns:
            m = means.loc[r, c]
            s = stds.loc[r, c]
            annot_matrix.loc[r, c] = f"{m:.3f}\n±{s:.3f}"

    sns.heatmap(
        means,
        annot=annot_matrix,
        fmt="",
        cmap="YlGnBu",
        cbar=True,
        cbar_kws={"label": "Mean AUROC"},
        annot_kws={"size": 9.5, "weight": "normal"},
        ax=ax,
        vmin=df["pooled_auroc"].quantile(0.10),
        vmax=df["pooled_auroc"].quantile(0.95)
    )
    ax.set_title("RQ3: Temporal Response (Rolling Sampler)", fontsize=11, weight="bold", pad=10)
    ax.set_xlabel("Sampling Resolution / Window (Hours)", fontsize=10)
    ax.set_ylabel("Lookback Horizon (Hours)", fontsize=10)
    ax.set_yticklabels([f"{int(h)}h ({int(h/24)}d)" for h in means.index], rotation=0)

    plt.tight_layout()
    fig3_pdf = OUTPUT_DIR / "figures" / "fig3_rq3_temporal_response_surface.pdf"
    fig3_png = OUTPUT_DIR / "figures" / "fig3_rq3_temporal_response_surface.png"
    plt.savefig(fig3_pdf, bbox_inches="tight")
    plt.savefig(fig3_png, dpi=300, bbox_inches="tight")
    shutil.copy(fig3_pdf, PAPER_FIG_DIR / "fig3_rq3_temporal_response_surface.pdf")
    shutil.copy(fig3_png, PAPER_FIG_DIR / "fig3_rq3_temporal_response_surface.png")
    plt.close()

    # =========================================================================
    # 5. Top 10 Configurations & LaTeX Export
    # =========================================================================
    top10 = df.sort_values("pooled_auroc", ascending=False).head(10)[
        ["model_name", "modality_label", "sampler_strategy", "lookback_days", "window_hours", "pooled_auroc", "pooled_f1", "within_person_auroc"]
    ].reset_index(drop=True)
    top10.to_csv(OUTPUT_DIR / "tables" / "table5_top10_configurations.csv", index=False)
    
    latex_code = top10.to_latex(
        index=True,
        float_format="%.4f",
        caption=f"Top 10 performing configurations across the {len(df)} rolling sampler runs in sweep {SWEEP_ID}.",
        label="tab:top10_sweep_results"
    )
    with open(OUTPUT_DIR / "tables" / "top10_table.tex", "w") as f:
        f.write(latex_code)
        
    print("\n================== 5. TOP 10 CONFIGURATIONS ==================")
    print(top10)
    print(f"\nAll publication tables and figures successfully exported to: {OUTPUT_DIR.resolve()} and {PAPER_FIG_DIR.resolve()}")

if __name__ == "__main__":
    run_analysis()
