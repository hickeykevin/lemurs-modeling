#!/usr/bin/env python3
"""Interactive Streamlit Dashboard for Wearable Sensor Modality & Model Synergy Analysis.

Supports:
- Dynamic loading of any sweep CSV file (via file path or drag-and-drop upload)
- Metric selection including Pooled Balanced Accuracy, AUROC, F1, Precision, Recall, Specificity
- RQ1: Modality hierarchy & model synergy evaluation
- RQ2: Featurizer representation matrix (Raw vs SummaryStats vs Catch22)
- RQ3: Temporal resolution surface (Lookback x Resample) with decoupled FLAML models + featurizers
- Tabular data filtering and CSV download
"""

from pathlib import Path
import io
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Wearable Modality Synergy Dashboard",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark modern styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        background-color: #1e293b;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_csv(source) -> pd.DataFrame:
    """Load CSV from file path or uploaded file buffer and compute derived metrics."""
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    # Clean string columns
    str_cols = ["model_pkg", "sampler_pkg", "modalities_choice", "featurizer_pkg", "sweep_id", "experiment"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Standardize featurizer names
    if "featurizer_pkg" in df.columns:
        df["featurizer_pkg"] = df["featurizer_pkg"].replace({
            "none": "raw",
            "none (raw sequences)": "raw",
            "None": "raw",
            "nan": "raw",
        }).fillna("raw")

    # Sweep 9dvq8uiy was pure RQ1 (featurizer was null in Hydra, meaning raw flattened sequences)
    if "sweep_id" in df.columns and "featurizer_pkg" in df.columns:
        mask_9dvq = (df["sweep_id"] == "9dvq8uiy")
        df.loc[mask_9dvq, "featurizer_pkg"] = "raw"

    # Ensure numeric columns
    numeric_cols = [
        "pooled_auroc", "pooled_f1", "pooled_precision", "pooled_recall",
        "pooled_specificity", "pooled_sens_at_spec_0.9", "within_person_auroc_mean",
        "lookback_hours", "resample_freq_hours", "num_modalities"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute Pooled Balanced Accuracy: (Recall + Specificity) / 2
    if "pooled_balanced_accuracy" not in df.columns:
        if "pooled_recall" in df.columns and "pooled_specificity" in df.columns:
            df["pooled_balanced_accuracy"] = (df["pooled_recall"] + df["pooled_specificity"]) / 2.0
    else:
        df["pooled_balanced_accuracy"] = pd.to_numeric(df["pooled_balanced_accuracy"], errors="coerce")
        missing_mask = df["pooled_balanced_accuracy"].isna() & df["pooled_recall"].notna() & df["pooled_specificity"].notna()
        df.loc[missing_mask, "pooled_balanced_accuracy"] = (df.loc[missing_mask, "pooled_recall"] + df.loc[missing_mask, "pooled_specificity"]) / 2.0

    # Parse lookback and resample if missing
    if "sampler_pkg" in df.columns:
        if "lookback_hours" not in df.columns or df["lookback_hours"].isna().all():
            def parse_sampler(s):
                try:
                    parts = str(s).replace("rolling_", "").split("_")
                    lb = int(parts[0].replace("h", ""))
                    rf = int(parts[1].replace("h", ""))
                    return lb, rf
                except Exception:
                    return None, None
            parsed = df["sampler_pkg"].apply(parse_sampler)
            df["lookback_hours"] = [p[0] for p in parsed]
            df["resample_freq_hours"] = [p[1] for p in parsed]

    # Create composite RQ3 Model Key (Model + Featurizer)
    def make_model_feat_label(row):
        m = str(row.get("model_pkg", "unknown"))
        f = str(row.get("featurizer_pkg", "raw"))
        m_clean = m.replace("flaml_", "")
        if f in ["raw", "none", "nan", ""]:
            return f"{m_clean} (Raw)"
        elif f == "summary_stats":
            return f"{m_clean} + SummaryStats"
        elif f == "catch22":
            return f"{m_clean} + Catch22"
        else:
            return f"{m_clean} + {f}"

    df["model_featurizer_label"] = df.apply(make_model_feat_label, axis=1)

    return df


# --- SIDEBAR: DYNAMIC DATA SOURCE ---
st.sidebar.title("⚙️ Data & Controls")

st.sidebar.subheader("📂 Results Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV Results", type=["csv"])

default_paths = [
    Path("reports/sweep_results/all_sweeps_results.csv"),
    Path("reports/sweep_analysis/all_sweeps_results.csv"),
    Path("all_sweeps_results.csv"),
]
found_default = next((p for p in default_paths if p.exists()), None)
default_str = str(found_default) if found_default else "reports/sweep_results/all_sweeps_results.csv"

custom_path = st.sidebar.text_input("Or CSV File Path:", value=default_str)

df = None
data_source_name = ""

if uploaded_file is not None:
    try:
        df = load_and_preprocess_csv(uploaded_file)
        data_source_name = f"Uploaded: {uploaded_file.name}"
        st.sidebar.success(f"Loaded {len(df):,} runs from upload")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded CSV: {e}")
elif custom_path and Path(custom_path).exists():
    try:
        df = load_and_preprocess_csv(custom_path)
        data_source_name = custom_path
        st.sidebar.success(f"Loaded {len(df):,} runs from {Path(custom_path).name}")
    except Exception as e:
        st.sidebar.error(f"Error loading {custom_path}: {e}")
else:
    st.sidebar.warning(f"File not found: {custom_path}")

if df is None:
    st.error("No data loaded. Please specify a valid CSV path or upload a results CSV file.")
    st.stop()

# --- SIDEBAR: GLOBAL METRIC SELECTOR ---
st.sidebar.subheader("📊 Primary Metric")
METRIC_OPTIONS = {
    "pooled_auroc": "Pooled AUROC",
    "pooled_balanced_accuracy": "Pooled Balanced Accuracy",
    "pooled_f1": "Pooled Macro F1",
    "pooled_precision": "Pooled Macro Precision",
    "pooled_recall": "Pooled Recall (Sensitivity)",
    "pooled_specificity": "Pooled Specificity",
    "pooled_sens_at_spec_0.9": "Pooled Sensitivity @ 90% Spec",
    "within_person_auroc_mean": "Within-Person AUROC (Mean)",
}

# Filter out metrics that are completely missing
available_metrics = {k: v for k, v in METRIC_OPTIONS.items() if k in df.columns and df[k].notna().any()}
metric_key = st.sidebar.selectbox("Choose Evaluation Metric", options=list(available_metrics.keys()), format_func=lambda k: available_metrics[k])
metric_label = available_metrics[metric_key]

# Modality display names mapping
MODALITY_DISPLAY = {
    "step": "Step Only (Unimodal)",
    "calorie": "Calorie Only (Unimodal)",
    "distance": "Distance Only (Unimodal)",
    "step_distance": "Step + Distance (Bimodal)",
    "step_calorie": "Step + Calorie (Bimodal)",
    "calorie_distance": "Calorie + Distance (Bimodal)",
    "step_calorie_distance": "Step + Calorie + Distance (Trimodal)",
}


# --- HEADER & KPI CARDS ---
st.title("Wearable Sensor Modality & Model Synergy Dashboard")
st.caption(f"Active Data Source: `{data_source_name}` | Total Runs: `{len(df):,}` | Active Metric: **{metric_label}**")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    top_run = df.sort_values(by=metric_key, ascending=False).iloc[0]
    st.metric("🏆 Global Top Score", f"{top_run[metric_key]:.4f}", f"{top_run['model_pkg']} ({top_run['modalities_choice']})")
with kpi_col2:
    mod_means = df.groupby("modalities_choice")[metric_key].mean()
    best_mod = mod_means.idxmax()
    st.metric("🥇 Best Modality Avg", f"{mod_means.max():.4f}", MODALITY_DISPLAY.get(best_mod, best_mod))
with kpi_col3:
    model_means = df.groupby("model_pkg")[metric_key].mean()
    best_model = model_means.idxmax()
    st.metric("🤖 Best Model Family", f"{model_means.max():.4f}", best_model.replace("flaml_", ""))
with kpi_col4:
    valid_runs = df[metric_key].notna().sum()
    st.metric("📈 Evaluated Runs", f"{valid_runs:,} / {len(df):,}")

# --- TABS FOR RQ1, RQ2, RQ3, & DATA EXPLORER ---
tab_rq1, tab_rq2, tab_rq3, tab_data = st.tabs([
    "🎯 RQ1: Modality Hierarchy",
    "🔬 RQ2: Feature Representations",
    "⏱️ RQ3: Temporal Dynamics (Decoupled)",
    "📋 Interactive Data Explorer"
])


# ==============================================================================
# TAB 1: RQ1 - SENSOR MODALITY HIERARCHY
# ==============================================================================
with tab_rq1:
    st.header("Research Question 1: Which Sensor Modality is Best?")
    st.markdown("""
    Evaluate unimodal vs. bimodal vs. trimodal passive sensor combinations across machine learning architectures.
    Use the controls below to dynamically aggregate runs, filter model families, and isolate pure raw vs. featurized sequences.
    """)

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 1, 1, 1])
    with col_ctrl1:
        all_models = sorted(df["model_pkg"].dropna().unique().tolist())
        selected_models_rq1 = st.multiselect("Filter Models for Aggregation", options=all_models, default=all_models)
    with col_ctrl2:
        rq1_metric_key = st.selectbox(
            "Ranking & Score Metric",
            options=list(available_metrics.keys()),
            index=list(available_metrics.keys()).index(metric_key) if metric_key in available_metrics else 0,
            format_func=lambda k: available_metrics[k],
            help="Select which pooled metric determines the rankings and scores."
        )
        rq1_metric_label = available_metrics[rq1_metric_key]
    with col_ctrl3:
        feat_filter_rq1 = st.selectbox(
            "Featurizer Consideration",
            options=["Pure Raw Only (Exclude Featurizers)", "All Runs (Raw + Featurizers)", "Only Featurized Runs"],
            index=0
        )
    with col_ctrl4:
        all_sweeps = sorted(df["sweep_id"].dropna().unique().tolist())
        selected_sweeps_rq1 = st.multiselect("Filter Sweep IDs", options=all_sweeps, default=all_sweeps)

    # Filter dataframe for RQ1
    df_rq1 = df[df["model_pkg"].isin(selected_models_rq1) & df["sweep_id"].isin(selected_sweeps_rq1)].copy()
    if feat_filter_rq1 == "Pure Raw Only (Exclude Featurizers)":
        df_rq1 = df_rq1[df_rq1["featurizer_pkg"].isin(["raw", "none", "nan", ""])]
    elif feat_filter_rq1 == "Only Featurized Runs":
        df_rq1 = df_rq1[~df_rq1["featurizer_pkg"].isin(["raw", "none", "nan", ""])]

    if len(df_rq1) == 0:
        st.warning("No runs match the selected filters. Please broaden your selection.")
    else:
        # Aggregation by modality
        mod_stats = df_rq1.groupby("modalities_choice")[rq1_metric_key].agg(
            Mean="mean",
            Std="std",
            Median="median",
            Count="count",
            Q25=lambda x: x.quantile(0.25),
            Q75=lambda x: x.quantile(0.75)
        ).reset_index()

        mod_stats["Display"] = mod_stats["modalities_choice"].map(lambda m: MODALITY_DISPLAY.get(m, m))
        mod_stats = mod_stats.sort_values(by="Mean", ascending=False)

        col_chart1, col_chart2 = st.columns([3, 2])

        with col_chart1:
            st.subheader(f"Modality Ranking ({rq1_metric_label})")
            fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0b0f19')
            ax.set_facecolor('#111827')

            bars = ax.barh(
                mod_stats["Display"],
                mod_stats["Mean"],
                xerr=mod_stats["Std"],
                color="#6366f1",
                edgecolor="#818cf8",
                capsize=4,
                alpha=0.85
            )
            # Highlight top bar
            if len(bars) > 0:
                bars[0].set_color("#10b981")
                bars[0].set_edgecolor("#34d399")

            ax.set_xlabel(f"Mean {rq1_metric_label} (± 1 SD)", color="#94a3b8", fontsize=11)
            ax.set_title(f"Sensor Modality Performance Hierarchy (N={len(df_rq1)} runs)", color="#f1f5f9", fontsize=13, pad=12)
            ax.tick_params(colors="#94a3b8", labelsize=10)
            ax.invert_yaxis()
            ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#334155")

            st.pyplot(fig)
            plt.close(fig)

        with col_chart2:
            st.subheader("Statistical Summary Table")
            display_tbl = mod_stats[["Display", "Mean", "Std", "Median", "Count"]].copy()
            display_tbl["Mean ± Std"] = display_tbl.apply(lambda r: f"{r['Mean']:.4f} ± {r['Std']:.4f}" if pd.notnull(r['Std']) else f"{r['Mean']:.4f}", axis=1)
            display_tbl["Median"] = display_tbl["Median"].apply(lambda v: f"{v:.4f}")
            st.dataframe(display_tbl[["Display", "Mean ± Std", "Median", "Count"]], use_container_width=True, hide_index=True)

            # Statistical Significance vs Top Modality
            if len(mod_stats) > 1:
                top_m = mod_stats.iloc[0]["modalities_choice"]
                top_scores = df_rq1[df_rq1["modalities_choice"] == top_m][rq1_metric_key].dropna()

                sig_records = []
                for _, row in mod_stats.iloc[1:].iterrows():
                    other_m = row["modalities_choice"]
                    other_scores = df_rq1[df_rq1["modalities_choice"] == other_m][rq1_metric_key].dropna()
                    if len(top_scores) > 1 and len(other_scores) > 1:
                        t_stat, p_val = stats.ttest_ind(top_scores, other_scores, equal_var=False)
                        sig_records.append({
                            "Comparison": f"{top_m} vs {other_m}",
                            "Diff": row["Mean"] - mod_stats.iloc[0]["Mean"],
                            "p-value": p_val,
                            "Significant (p<0.05)": "✅ Yes" if p_val < 0.05 else "❌ No"
                        })
                if sig_records:
                    st.caption(f"Significance tests vs top modality (`{top_m}`):")
                    st.dataframe(pd.DataFrame(sig_records), use_container_width=True, hide_index=True)

        # ======================================================================
        # AVERAGE RANK COMPUTATION FOR MODALITIES (NON-PARAMETRIC CONSENSUS)
        # ======================================================================
        st.markdown("---")
        st.subheader(f"🏅 Modality Consensus Ranking (Ranked by {rq1_metric_label})")
        st.markdown(f"""
        For each selected model architecture, the evaluated modalities are ranked from **1 (best)** to **$K$ (worst)** 
        based on mean **{rq1_metric_label}**. The table and chart below show each modality's **Average Rank** across all selected models.
        """)

        # Compute rank for each model
        model_ranks_dict = {}
        for m in selected_models_rq1:
            m_sub = df_rq1[df_rq1["model_pkg"] == m]
            if len(m_sub) > 0:
                m_means = m_sub.groupby("modalities_choice")[rq1_metric_key].mean()
                if len(m_means) > 0:
                    m_ranks = m_means.rank(ascending=False, method="min")
                    clean_name = m.replace("flaml_", "")
                    model_ranks_dict[clean_name] = m_ranks

        if model_ranks_dict:
            rank_df = pd.DataFrame(model_ranks_dict)
            rank_summary = pd.DataFrame(index=rank_df.index)
            rank_summary["Average Rank"] = rank_df.mean(axis=1)
            rank_summary["Median Rank"] = rank_df.median(axis=1)
            rank_summary["Times Ranked #1"] = (rank_df == 1).sum(axis=1)

            # Combine summary with individual model rank columns
            for col in rank_df.columns:
                rank_summary[f"{col} Rank"] = rank_df[col].apply(lambda v: f"#{int(v)}" if pd.notnull(v) else "-")

            rank_summary = rank_summary.sort_values(by="Average Rank", ascending=True).reset_index()
            rank_summary.rename(columns={"index": "modalities_choice"}, inplace=True)
            rank_summary["Modality"] = rank_summary["modalities_choice"].map(lambda m: MODALITY_DISPLAY.get(m, m))
            rank_summary.insert(0, "Consensus Rank", [f"#{i+1}" for i in range(len(rank_summary))])

            col_rank_chart, col_rank_tbl = st.columns([3, 3])

            with col_rank_chart:
                fig_rank, ax_rank = plt.subplots(figsize=(8, 4.5), facecolor='#0b0f19')
                ax_rank.set_facecolor('#111827')

                plot_df = rank_summary.sort_values(by="Average Rank", ascending=False)
                bars_rank = ax_rank.barh(
                    plot_df["Modality"],
                    plot_df["Average Rank"],
                    color="#4f46e5",
                    edgecolor="#818cf8",
                    alpha=0.85
                )
                if len(bars_rank) > 0:
                    bars_rank[-1].set_color("#10b981")
                    bars_rank[-1].set_edgecolor("#34d399")

                ax_rank.set_xlabel("Average Rank across Models (1.0 = Best Possible)", color="#94a3b8", fontsize=11)
                ax_rank.set_title(f"Average Modality Rank across {len(model_ranks_dict)} Model Architectures", color="#f1f5f9", fontsize=12, pad=12)
                ax_rank.tick_params(colors="#94a3b8", labelsize=10)
                ax_rank.grid(axis="x", color="#334155", linestyle="--", alpha=0.5)
                for spine in ax_rank.spines.values():
                    spine.set_color("#334155")

                st.pyplot(fig_rank)
                plt.close(fig_rank)

            with col_rank_tbl:
                display_cols = ["Consensus Rank", "Modality", "Average Rank", "Median Rank", "Times Ranked #1"] + [f"{col} Rank" for col in rank_df.columns]
                format_summary = rank_summary[display_cols].copy()
                format_summary["Average Rank"] = format_summary["Average Rank"].apply(lambda v: f"{v:.2f}")
                format_summary["Median Rank"] = format_summary["Median Rank"].apply(lambda v: f"{v:.1f}")
                st.dataframe(format_summary, use_container_width=True, hide_index=True)

        # Model x Modality Grouped Breakdown
        st.markdown("---")
        st.subheader("Model Architecture Synergy across Sensor Modalities")
        pivot_df = df_rq1.pivot_table(index="modalities_choice", columns="model_pkg", values=rq1_metric_key, aggfunc="mean")
        pivot_df.index = pivot_df.index.map(lambda m: MODALITY_DISPLAY.get(m, m))

        fig_heat, ax_heat = plt.subplots(figsize=(10, 4.5), facecolor='#0b0f19')
        ax_heat.set_facecolor('#111827')
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="magma", ax=ax_heat, cbar_kws={'label': rq1_metric_label})
        ax_heat.set_title(f"Model Architecture × Modality Matrix ({rq1_metric_label})", color="#f1f5f9", fontsize=13, pad=12)
        ax_heat.tick_params(colors="#94a3b8", labelsize=10)
        st.pyplot(fig_heat)
        plt.close(fig_heat)


# ==============================================================================
# TAB 2: RQ2 - FEATURE EXTRACTION REPRESENTATIONS
# ==============================================================================
with tab_rq2:
    st.header("Research Question 2: Raw vs. Featurized Representations")
    st.markdown("""
    Evaluate feature extraction methods (**Summary Statistics** vs. **Catch22** canonical time-series features vs. **Raw Sequences**)
    for any selected modality stream across machine learning model families.
    """)

    col_q2_1, col_q2_2, col_q2_3 = st.columns([1, 1, 2])
    with col_q2_1:
        all_mods_q2 = sorted(df["modalities_choice"].dropna().unique().tolist())
        default_mod_idx = all_mods_q2.index("step_distance") if "step_distance" in all_mods_q2 else 0
        selected_mod_rq2 = st.selectbox("Select Modality Stream", options=all_mods_q2, index=default_mod_idx, format_func=lambda m: MODALITY_DISPLAY.get(m, m))

    with col_q2_2:
        rq2_metric_key = st.selectbox(
            "Sort & Focus Metric",
            options=list(available_metrics.keys()),
            index=list(available_metrics.keys()).index(metric_key) if metric_key in available_metrics else 0,
            format_func=lambda k: available_metrics[k],
            help="Select which pooled metric determines the ranking and chart visualization."
        )
        rq2_metric_label = available_metrics[rq2_metric_key]

    with col_q2_3:
        # Available model variants for this modality (or globally)
        avail_variants_in_mod = sorted(df[df["modalities_choice"] == selected_mod_rq2]["model_featurizer_label"].dropna().unique().tolist())
        if not avail_variants_in_mod:
            avail_variants_in_mod = sorted(df["model_featurizer_label"].dropna().unique().tolist())

        selected_variants_rq2 = st.multiselect(
            "Select Model & Featurizer Variants",
            options=avail_variants_in_mod,
            default=avail_variants_in_mod,
            help="Each tabular method has its raw, SummaryStats, and Catch22 variants as separate models to select and compare."
        )

    df_rq2 = df[(df["modalities_choice"] == selected_mod_rq2) & (df["model_featurizer_label"].isin(selected_variants_rq2))].copy()

    if len(df_rq2) == 0:
        st.warning(f"No runs found for modality `{selected_mod_rq2}` with the selected model variants.")
    else:
        st.subheader(f"Feature Extraction Performance Comparison on `{selected_mod_rq2}` (Sorted by {rq2_metric_label})")

        all_metrics_to_show = [
            ("pooled_auroc", "AUROC"),
            ("pooled_balanced_accuracy", "Bal Acc"),
            ("pooled_f1", "Macro F1"),
            ("pooled_precision", "Precision"),
            ("pooled_recall", "Recall"),
            ("pooled_specificity", "Specificity"),
            ("pooled_sens_at_spec_0.9", "Sens@90%"),
            ("within_person_auroc_mean", "WP AUROC"),
        ]

        grouped = df_rq2.groupby(["model_featurizer_label", "model_pkg", "featurizer_pkg"])
        records = []
        for (m_label, m_pkg, f_pkg), group in grouped:
            rank_val = group[rq2_metric_key].mean() if rq2_metric_key in group.columns else 0
            rec = {
                "Model Variant": m_label,
                "Base Architecture": m_pkg.replace("flaml_", ""),
                "Featurizer": f_pkg,
                "Rank_Score": rank_val if pd.notnull(rank_val) else -999,
                f"Mean {rq2_metric_label}": f"{rank_val:.4f}" if pd.notnull(rank_val) else "-",
                "Std Dev": f"± {group[rq2_metric_key].std():.4f}" if pd.notnull(group[rq2_metric_key].std()) else "-",
            }
            # Append all pooled metrics
            for m_col, m_name in all_metrics_to_show:
                if m_col in group.columns:
                    val = group[m_col].mean()
                    rec[m_name] = f"{val:.4f}" if pd.notnull(val) else "-"

            rec["Runs (N)"] = len(group)
            records.append(rec)

        res_df = pd.DataFrame(records).sort_values(by="Rank_Score", ascending=False).reset_index(drop=True)
        top_score = res_df.iloc[0]["Rank_Score"] if len(res_df) > 0 and res_df.iloc[0]["Rank_Score"] != -999 else 0
        res_df[f"Δ vs Top ({rq2_metric_label})"] = res_df["Rank_Score"].apply(
            lambda v: f"{v - top_score:+.4f}" if v != -999 and top_score != -999 else "-"
        )

        # Reorder columns logically
        lead_cols = ["Model Variant", "Base Architecture", "Featurizer", f"Mean {rq2_metric_label}", "Std Dev", f"Δ vs Top ({rq2_metric_label})"]
        metric_cols = [m_name for m_col, m_name in all_metrics_to_show if m_name in res_df.columns and m_name not in lead_cols]
        final_cols = lead_cols + metric_cols + ["Runs (N)"]
        display_tbl = res_df[[c for c in final_cols if c in res_df.columns]]

        st.dataframe(display_tbl, use_container_width=True, hide_index=True)

        # Bar chart comparison
        chart_height = max(4.5, len(res_df) * 0.4)
        fig_bar, ax_bar = plt.subplots(figsize=(10, chart_height), facecolor='#0b0f19')
        ax_bar.set_facecolor('#111827')

        valid_bars_df = res_df[res_df["Rank_Score"] != -999].copy()
        bars = ax_bar.barh(
            valid_bars_df["Model Variant"],
            valid_bars_df["Rank_Score"],
            color="#6366f1",
            edgecolor="#818cf8",
            alpha=0.85
        )
        if len(bars) > 0:
            bars[0].set_color("#10b981")
            bars[0].set_edgecolor("#34d399")

        ax_bar.set_xlabel(f"Mean {rq2_metric_label}", color="#94a3b8", fontsize=11)
        ax_bar.set_title(f"Model Variant Comparison on {MODALITY_DISPLAY.get(selected_mod_rq2, selected_mod_rq2)} ({rq2_metric_label})", color="#f1f5f9", fontsize=12, pad=12)
        ax_bar.tick_params(colors="#94a3b8", labelsize=10)
        ax_bar.invert_yaxis()
        ax_bar.grid(axis="x", color="#334155", linestyle="--", alpha=0.5)
        for spine in ax_bar.spines.values():
            spine.set_color("#334155")

        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Cross-tabulation matrix expander
        with st.expander(f"📊 View Cross-Tabulation Matrix (Base Architecture × Featurizer for {rq2_metric_label})"):
            p_rq2 = df_rq2.pivot_table(index="model_pkg", columns="featurizer_pkg", values=rq2_metric_key, aggfunc="mean").reset_index()
            col_rename = {
                "model_pkg": "Model",
                "raw": "Baseline (Raw)",
                "summary_stats": "SummaryStats (+5)",
                "catch22": "Catch22 (+22)",
            }
            p_rq2 = p_rq2.rename(columns=col_rename)
            cols_order = ["Model"]
            for col_name in ["Baseline (Raw)", "SummaryStats (+5)", "Catch22 (+22)"]:
                if col_name in p_rq2.columns:
                    cols_order.append(col_name)
            for col_name in p_rq2.columns:
                if col_name not in cols_order:
                    cols_order.append(col_name)
            p_rq2 = p_rq2[cols_order]

            display_p_rq2 = p_rq2.copy()
            for col_name in display_p_rq2.columns:
                if col_name != "Model":
                    display_p_rq2[col_name] = display_p_rq2[col_name].apply(lambda v: f"{v:.4f}" if pd.notnull(v) else "-")
            st.dataframe(display_p_rq2, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: RQ3 - TEMPORAL DYNAMICS (TRI-PARADIGM HEATMAP COMPARISON)
# ==============================================================================
with tab_rq3:
    st.header("Research Question 3: Temporal Dynamics (Lookback × Resample Surfaces)")
    st.markdown("""
    Evaluate how **lookback window size** (24h to 120h) and **resample bin width** (4h to 12h) govern performance across the three modeling paradigms:
    **Tabular Methods**, **Deep Learning Architectures**, and **Foundation Models**.
    Compare their temporal response surfaces side-by-side on the same wearable modality stream.
    """)

    # Top Controls: Modality Stream, Heatmap Metric, and Colorscale Sync
    col_q3_1, col_q3_2, col_q3_3 = st.columns([1.5, 1.5, 1])
    with col_q3_1:
        all_mods_q3 = sorted(df["modalities_choice"].dropna().unique().tolist())
        selected_mod_rq3 = st.multiselect(
            "Modality Streams to Aggregate Over",
            options=all_mods_q3,
            default=["step_calorie_distance"] if "step_calorie_distance" in all_mods_q3 else [all_mods_q3[0]],
            format_func=lambda m: MODALITY_DISPLAY.get(m, m),
            help="Select one or more wearable modality streams to analyze."
        )

    with col_q3_2:
        rq3_metric_key = st.selectbox(
            "Heatmap Metric",
            options=list(available_metrics.keys()),
            index=list(available_metrics.keys()).index(metric_key) if metric_key in available_metrics else 0,
            format_func=lambda k: available_metrics[k],
            help="Select which pooled metric to evaluate across the temporal surfaces."
        )
        rq3_metric_label = available_metrics[rq3_metric_key]

    with col_q3_3:
        sync_colors = st.checkbox(
            "Sync Colorscales",
            value=True,
            help="Synchronize min/max color limits across all 3 heatmaps for direct visual comparison."
        )

    # Categorize model options into the 3 paradigms
    all_model_labels = sorted(df["model_featurizer_label"].dropna().unique().tolist())

    tab_keywords = ["xgboost", "rf", "random_forest", "lr", "logistic", "svm", "catboost", "lightgbm", "flaml"]
    tabular_options = [m for m in all_model_labels if any(k in m.lower() for k in tab_keywords)]
    if tabular_options:
        tabular_options = ["All Tabular (Mean)"] + tabular_options

    dl_keywords = ["lstm", "transformer", "cnn", "gru", "rnn", "mlp", "tcn", "resnet"]
    dl_options = [m for m in all_model_labels if any(k in m.lower() for k in dl_keywords)]
    if dl_options:
        dl_options = ["All Deep Learning (Mean)"] + dl_options

    fm_keywords = ["chronos", "wbm", "bolt", "foundation", "timesfm", "moment", "lag_llama", "moirai"]
    fm_options = [m for m in all_model_labels if any(k in m.lower() for k in fm_keywords)]
    if fm_options:
        fm_options = ["All Foundation Models (Mean)"] + fm_options

    # Setup the 3 columns in a row
    col_tab, col_dl, col_fm = st.columns(3)

    # Column 1 Selector: Tabular
    with col_tab:
        st.subheader("1. 📊 Tabular Model")
        if tabular_options:
            default_tab_idx = 0
            for idx, opt in enumerate(tabular_options):
                if "summarystats" in opt.lower() and ("rf" in opt.lower() or "xgboost" in opt.lower()):
                    default_tab_idx = idx
                    break
            chosen_tab = st.selectbox(
                "Select Tabular Variant",
                options=tabular_options,
                index=default_tab_idx,
                key="rq3_tab_model",
                help="Choose a specific tabular algorithm and featurizer pairing."
            )
        else:
            chosen_tab = None
            st.info("No Tabular models found.")

    # Column 2 Selector: Deep Learning
    with col_dl:
        st.subheader("2. 🧠 Deep Learning")
        if dl_options:
            default_dl_idx = 0
            for idx, opt in enumerate(dl_options):
                if "lstm" in opt.lower():
                    default_dl_idx = idx
                    break
            chosen_dl = st.selectbox(
                "Select DL Model",
                options=dl_options,
                index=default_dl_idx,
                key="rq3_dl_model",
                help="Choose a deep learning sequence architecture."
            )
        else:
            chosen_dl = None
            st.info("No Deep Learning models found.")

    # Column 3 Selector: Foundation Model
    with col_fm:
        st.subheader("3. 🔮 Foundation Model")
        if fm_options:
            chosen_fm = st.selectbox(
                "Select Foundation Model",
                options=fm_options,
                index=0,
                key="rq3_fm_model",
                help="Choose a time-series foundation model architecture."
            )
        else:
            chosen_fm = None
            st.selectbox("Select Foundation Model", options=["(Awaiting sweep results...)"], disabled=True, key="rq3_fm_model")

    # Helper to retrieve data and compute matrix
    def get_heat_matrix(chosen_model, options_list):
        if not chosen_model or not selected_mod_rq3:
            return None, None
        base_df = df[
            df["modalities_choice"].isin(selected_mod_rq3) &
            df["lookback_hours"].notna() &
            df["resample_freq_hours"].notna()
        ]
        if chosen_model == "All Tabular (Mean)":
            sub_labels = [m for m in options_list if m != "All Tabular (Mean)"]
            sub_df = base_df[base_df["model_featurizer_label"].isin(sub_labels)]
        elif chosen_model == "All Deep Learning (Mean)":
            sub_labels = [m for m in options_list if m != "All Deep Learning (Mean)"]
            sub_df = base_df[base_df["model_featurizer_label"].isin(sub_labels)]
        elif chosen_model == "All Foundation Models (Mean)":
            sub_labels = [m for m in options_list if m != "All Foundation Models (Mean)"]
            sub_df = base_df[base_df["model_featurizer_label"].isin(sub_labels)]
        else:
            sub_df = base_df[base_df["model_featurizer_label"] == chosen_model]

        if len(sub_df) == 0:
            return None, sub_df

        matrix = sub_df.pivot_table(
            index="lookback_hours",
            columns="resample_freq_hours",
            values=rq3_metric_key,
            aggfunc="mean"
        ).sort_index(ascending=False)
        return matrix, sub_df

    mat_tab, df_tab = get_heat_matrix(chosen_tab, tabular_options)
    mat_dl, df_dl = get_heat_matrix(chosen_dl, dl_options)
    mat_fm, df_fm = get_heat_matrix(chosen_fm, fm_options)

    # Compute shared color limits if requested
    shared_vmin, shared_vmax = None, None
    if sync_colors:
        vals = []
        for m in [mat_tab, mat_dl, mat_fm]:
            if m is not None and not m.empty:
                vals.extend(m.values.flatten())
        clean_vals = [float(v) for v in vals if pd.notna(v)]
        if clean_vals:
            shared_vmin = min(clean_vals)
            shared_vmax = max(clean_vals)

    # Helper function to render a heatmap card
    def render_heatmap_card(col, title, chosen_model, matrix, sub_df):
        with col:
            if chosen_model is None or matrix is None or matrix.empty:
                if "foundation" in title.lower() or "3." in title:
                    st.info(
                        "🕒 **Awaiting Foundation Model Results**\n\n"
                        "Sweep runs for **Chronos-Bolt**, **Chronos-2**, and **WBM** are pending. "
                        "When these results are added to `all_sweeps_results.csv`, this heatmap will automatically populate!"
                    )
                else:
                    st.warning(f"No run data available for {chosen_model} on the selected modality.")
                return

            fig, ax = plt.subplots(figsize=(4.8, 3.8), facecolor='#0b0f19')
            ax.set_facecolor('#111827')

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                vmin=shared_vmin,
                vmax=shared_vmax,
                linewidths=0.5,
                linecolor="#1f2937",
                cbar_kws={'label': rq3_metric_label, 'shrink': 0.8},
                annot_kws={"size": 8.5, "weight": "bold"},
                ax=ax
            )

            ax.set_title(
                f"{chosen_model}\n(N={len(sub_df)} runs)",
                color="#f1f5f9",
                fontsize=10.5,
                pad=8
            )
            ax.set_xlabel("Resample Bin Width (Hours)", color="#94a3b8", fontsize=8.5)
            ax.set_ylabel("Lookback Window (Hours)", color="#94a3b8", fontsize=8.5)
            ax.tick_params(colors="#94a3b8", labelsize=8.5)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # Optimal Point readout
            stacked = matrix.stack()
            if not stacked.empty:
                best_val = stacked.max()
                best_idx = stacked.idxmax()
                st.metric(f"Top {rq3_metric_label}", f"{best_val:.4f}")
                st.markdown(f"🎯 **Optimal**: Lookback `{best_idx[0]}h` × Bin `{best_idx[1]}h`")

    render_heatmap_card(col_tab, "Tabular", chosen_tab, mat_tab, df_tab)
    render_heatmap_card(col_dl, "Deep Learning", chosen_dl, mat_dl, df_dl)
    render_heatmap_card(col_fm, "Foundation Model", chosen_fm, mat_fm, df_fm)


# ==============================================================================
# TAB 4: INTERACTIVE DATAFRAME EXPLORER
# ==============================================================================
with tab_data:
    st.header("Interactive Dataframe Explorer")
    st.markdown("Search, filter, inspect, and download any subset of the evaluated walk-forward runs.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        mod_filter = st.multiselect("Modality", options=sorted(df["modalities_choice"].unique()), default=[])
    with col_f2:
        model_filter = st.multiselect("Model Family", options=sorted(df["model_pkg"].unique()), default=[])
    with col_f3:
        feat_filter = st.multiselect("Featurizer", options=sorted(df["featurizer_pkg"].unique()), default=[])

    filtered_df = df.copy()
    if mod_filter:
        filtered_df = filtered_df[filtered_df["modalities_choice"].isin(mod_filter)]
    if model_filter:
        filtered_df = filtered_df[filtered_df["model_pkg"].isin(model_filter)]
    if feat_filter:
        filtered_df = filtered_df[filtered_df["featurizer_pkg"].isin(feat_filter)]

    st.write(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records:")

    display_cols = [
        "run_id", "model_pkg", "modalities_choice", "featurizer_pkg",
        "sampler_pkg", "lookback_hours", "resample_freq_hours",
        "pooled_auroc", "pooled_balanced_accuracy", "pooled_f1",
        "pooled_precision", "pooled_recall", "pooled_specificity", "sweep_id"
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(filtered_df[display_cols], use_container_width=True)

    # Download CSV button
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="lemurs_sweep_filtered_results.csv",
        mime="text/csv",
    )
