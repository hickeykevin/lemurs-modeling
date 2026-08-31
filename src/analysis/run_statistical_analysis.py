import os
import json
import netrc
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("reports/sweep_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

CACHE_FILE = OUTPUT_DIR / "sweep_vj7lbnsh_full_records.csv"

def fetch_data():
    if CACHE_FILE.exists():
        print(f"Loading cached records from {CACHE_FILE}...")
        return pd.read_csv(CACHE_FILE)
    
    print("Fetching 840 runs from W&B API via GraphQL...")
    n = netrc.netrc()
    auth = n.authenticators("api.wandb.ai")
    headers = {"Authorization": f"Bearer {auth[2]}", "Content-Type": "application/json"}
    
    query = """
    query SweepRuns($entity: String!, $project: String!, $name: String!) {
      project(entityName: $entity, name: $project) {
        sweep(sweepName: $name) {
          runs(first: 1500) {
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
            "variables": {"entity": "hickeykevin", "project": "lemurs-modeling", "name": "vj7lbnsh"}
        }
    )
    edges = resp.json()["data"]["project"]["sweep"]["runs"]["edges"]
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
    df = fetch_data()
    print(f"Loaded {len(df)} runs.")

    # Parsing factors
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
    df["feature_density"] = df["lookback_hours"] / df["window_hours"]

    model_map = {"default": "LSTM", "transformer": "Transformer", "flaml_xgboost": "FLAML XGBoost"}
    df["model_name"] = df["model_pkg"].map(model_map)

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

    mod_labels, mod_counts, has_step, has_cal, has_dist = zip(*df["modalities_raw"].map(format_modalities))
    df["modality_label"] = mod_labels
    df["n_modalities"] = mod_counts
    df["has_step"] = has_step
    df["has_calorie"] = has_cal
    df["has_distance"] = has_dist

    # 1. Global Baseline Test
    t_stat, p_val = stats.ttest_1samp(df["pooled_auroc"], 0.50)
    print("\n================== 1. GLOBAL PREDICTIVENESS ==================")
    print(f"Mean AUROC: {df['pooled_auroc'].mean():.4f} +/- {df['pooled_auroc'].std():.4f} (t={t_stat:.2f}, p={p_val:.2e})")
    print(f"Max AUROC: {df['pooled_auroc'].max():.4f}")

    # Global Factor summary table
    summary_factors = []
    for factor, name in [
        ("model_name", "Model Architecture"),
        ("modality_label", "Modality Subset"),
        ("lookback_hours", "Lookback Horizon (Hours)"),
        ("window_hours", "Sampling Granularity (Hours)"),
        ("sampler_strategy", "Sampling Strategy")
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

    # 2. RQ1: Modality Ablation Table & OLS Regression
    mod_order = [
        "Step Only", "Calorie Only", "Distance Only",
        "Step + Calorie", "Step + Distance", "Calorie + Distance",
        "All 3 (Step+Cal+Dist)"
    ]
    mod_stats = df.groupby("modality_label").agg(
        N=("pooled_auroc", "count"),
        AUROC_Mean=("pooled_auroc", "mean"),
        AUROC_SD=("pooled_auroc", "std"),
        AUROC_Median=("pooled_auroc", "median"),
        F1_Mean=("pooled_f1", "mean"),
        WithinPerson_AUROC=("within_person_auroc", "mean")
    ).reindex(mod_order)
    mod_stats.to_csv(OUTPUT_DIR / "tables" / "table2_rq1_modality_ablation.csv")
    print("\n================== 2. RQ1: MODALITY ABLATION ==================")
    print(mod_stats)

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
    axes[1].set_xticklabels(["1 Modality\n(Single)", "2 Modalities\n(Pair)", "3 Modalities\n(Trivariate)"])
    axes[1].set_title("(B) Modality Expansion by Model Architecture")
    axes[1].set_xlabel("Feature Set Complexity")
    axes[1].set_ylabel("")
    axes[1].legend(title="Model Architecture", loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "fig1_rq1_modality_value.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "figures" / "fig1_rq1_modality_value.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. RQ2: Head-to-head Model comparisons (N=280 matched pairs)
    paired_df = df.pivot(index=["sampler_pkg", "modalities_raw"], columns="model_name", values="pooled_auroc").reset_index()
    paired_df["Delta_LSTM_minus_XGB"] = paired_df["LSTM"] - paired_df["FLAML XGBoost"]
    paired_df["Delta_Transformer_minus_XGB"] = paired_df["Transformer"] - paired_df["FLAML XGBoost"]
    paired_df["Delta_Transformer_minus_LSTM"] = paired_df["Transformer"] - paired_df["LSTM"]

    def run_paired_test(diff_series, name):
        diff = diff_series.dropna()
        n = len(diff)
        mean_d = diff.mean()
        std_d = diff.std()
        cohens_d = mean_d / std_d
        t_stat, t_pval = stats.ttest_1samp(diff, 0)
        w_stat, w_pval = stats.wilcoxon(diff)
        ci_l, ci_u = stats.t.interval(0.95, n-1, loc=mean_d, scale=stats.sem(diff))
        win_rate = (diff > 0).mean() * 100
        return {
            "Comparison": name,
            "N_Pairs": n,
            "Mean_Delta": mean_d,
            "95% CI": f"[{ci_l:.4f}, {ci_u:.4f}]",
            "Win_Rate (%)": win_rate,
            "Cohens_d": cohens_d,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "Wilcoxon_W": w_stat,
            "Wilcoxon_pval": w_pval
        }

    comp_table = pd.DataFrame([
        run_paired_test(paired_df["Delta_LSTM_minus_XGB"], "LSTM vs. FLAML XGBoost"),
        run_paired_test(paired_df["Delta_Transformer_minus_XGB"], "Transformer vs. FLAML XGBoost"),
        run_paired_test(paired_df["Delta_Transformer_minus_LSTM"], "Transformer vs. LSTM")
    ])
    print("\n================== 3. RQ2: MODEL HEAD-TO-HEAD ==================")
    print(comp_table)
    comp_table.to_csv(OUTPUT_DIR / "tables" / "table3_rq2_model_head_to_head.csv", index=False)

    # Figure 2: Model x Horizon Interaction
    lb_mapping = df[["sampler_pkg", "lookback_days"]].drop_duplicates().set_index("sampler_pkg")["lookback_days"]
    paired_df["lookback_days"] = paired_df["sampler_pkg"].map(lb_mapping)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(
        data=df,
        x="lookback_days",
        y="pooled_auroc",
        hue="model_name",
        style="model_name",
        markers=True,
        dashes=False,
        palette="Set1",
        errorbar=("ci", 95),
        ax=axes[0]
    )
    axes[0].set_title("(A) Model Discrimination across Lookback Horizons")
    axes[0].set_xlabel("Historical Lookback Window (Days)")
    axes[0].set_ylabel("Pooled AUROC (Mean +/- 95% CI)")
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].legend(title="Model", loc="lower right")

    delta_melt = paired_df.melt(
        id_vars=["lookback_days"],
        value_vars=["Delta_LSTM_minus_XGB", "Delta_Transformer_minus_XGB"],
        var_name="Comparison",
        value_name="AUROC_Advantage"
    )
    delta_melt["Comparison"] = delta_melt["Comparison"].map({
        "Delta_LSTM_minus_XGB": "LSTM Advantage over XGBoost",
        "Delta_Transformer_minus_XGB": "Transformer Advantage over XGBoost"
    })
    sns.lineplot(
        data=delta_melt,
        x="lookback_days",
        y="AUROC_Advantage",
        hue="Comparison",
        style="Comparison",
        markers=True,
        palette=["#2ca02c", "#d62728"],
        ax=axes[1]
    )
    axes[1].axhline(0, color="black", linestyle=":", alpha=0.8)
    axes[1].set_title("(B) Time-Series Advantage Over Classical ML by Horizon")
    axes[1].set_xlabel("Historical Lookback Window (Days)")
    axes[1].set_ylabel("Delta AUROC (Sequence - Tree)")
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "fig2_rq2_model_horizon_interaction.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "figures" / "fig2_rq2_model_horizon_interaction.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. RQ3: 2D Response Surface Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, (strategy_name, ax) in enumerate(zip(["Rolling", "Offset"], axes)):
        strat_df = df[df["sampler_strategy"] == strategy_name]
        pivot_map = strat_df.pivot_table(
            index="lookback_hours",
            columns="window_hours",
            values="pooled_auroc",
            aggfunc="mean"
        )
        sns.heatmap(
            pivot_map,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar=(idx == 1),
            ax=ax,
            vmin=df["pooled_auroc"].quantile(0.10),
            vmax=df["pooled_auroc"].quantile(0.95)
        )
        ax.set_title(f"Strategy: {strategy_name} Sampling")
        ax.set_xlabel("Resample Frequency / Window (Hours)")
        ax.set_ylabel("Lookback Horizon (Hours)" if idx == 0 else "")
        ax.set_yticklabels([f"{int(h)}h ({int(h/24)}d)" for h in pivot_map.index], rotation=0)

    plt.suptitle("Figure 3: AUROC Response Surface by Temporal Granularity and Horizon", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "fig3_rq3_temporal_response_surface.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "figures" / "fig3_rq3_temporal_response_surface.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Top 10 Configurations & LaTeX Export
    top10 = df.sort_values("pooled_auroc", ascending=False).head(10)[
        ["model_name", "modality_label", "sampler_strategy", "lookback_days", "window_hours", "pooled_auroc", "pooled_f1", "within_person_auroc"]
    ].reset_index(drop=True)
    top10.to_csv(OUTPUT_DIR / "tables" / "table5_top10_configurations.csv", index=False)
    
    latex_code = top10.to_latex(
        index=True,
        float_format="%.4f",
        caption="Top 10 performing configurations across the 840-run passive sensing hyperparameter grid.",
        label="tab:top10_sweep_results"
    )
    with open(OUTPUT_DIR / "tables" / "top10_table.tex", "w") as f:
        f.write(latex_code)
        
    print("\n================== 4. TOP 10 CONFIGURATIONS ==================")
    print(top10)
    print(f"\nAll publication tables and figures successfully exported to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    run_analysis()
