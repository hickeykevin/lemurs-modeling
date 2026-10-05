"""Generates an interactive, standalone HTML dashboard and Jupyter Notebook
to analyze the combined sweep results (9dvq8uiy, a6sl4omd, zoznzd8y)
and answer RQ1: "Which sensor modality is best?"
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

import sys

def generate_interactive_dashboard(custom_csv_path: str = None):
    # Check CLI argument, explicit argument, or search standard locations
    if custom_csv_path:
        csv_candidates = [Path(custom_csv_path)]
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        csv_candidates = [Path(sys.argv[1])]
    else:
        csv_candidates = [
            Path("reports/sweep_analysis/all_sweeps_results.csv"),
            Path("reports/sweep_results/all_sweeps_results.csv"),
            Path("all_sweeps_results.csv"),
        ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find all_sweeps_results.csv. Searched: {[str(p) for p in csv_candidates]}. "
            "Please run gather_sweep_results.py first or specify path as argument."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Prepare sanitized records for JavaScript consumption
    # Only keep necessary columns to keep JSON compact and speedy
    records = []
    for _, row in df.iterrows():
        # Balanced accuracy calculation
        bal_acc = None
        if pd.notnull(row.get("pooled_balanced_accuracy")):
            bal_acc = round(float(row["pooled_balanced_accuracy"]), 4)
        elif pd.notnull(row.get("pooled_recall")) and pd.notnull(row.get("pooled_specificity")):
            bal_acc = round((float(row["pooled_recall"]) + float(row["pooled_specificity"])) / 2.0, 4)

        # Decoupled model + featurizer label for RQ3
        m = str(row.get("model_pkg", ""))
        f = str(row.get("featurizer_pkg", ""))
        if str(row.get("sweep_id")) == "9dvq8uiy":
            f_norm = "raw"
        else:
            f_norm = "raw" if f in ["raw", "none", "nan", "", "none (raw sequences)"] else f
        m_clean = m.replace("flaml_", "")
        if f_norm == "raw":
            m_feat = f"{m_clean} (raw)"
        else:
            m_feat = f"{m_clean} + {f_norm}"

        records.append({
            "sweep": str(row["sweep_id"]),
            "exp": str(row["experiment"]),
            "type": str(row["model_type"]),
            "run_id": str(row["run_id"]),
            "run_name": str(row["run_name"]),
            "model": str(row["model_pkg"]),
            "sampler": str(row["sampler_pkg"]),
            "lookback": int(row["lookback_hours"]) if pd.notnull(row["lookback_hours"]) else None,
            "resample": int(row["resample_freq_hours"]) if pd.notnull(row["resample_freq_hours"]) else None,
            "modality": str(row["modalities_choice"]),
            "n_mods": int(row["num_modalities"]) if pd.notnull(row["num_modalities"]) else 1,
            "featurizer": str(row["featurizer_pkg"]),
            "m_feat": m_feat,
            "auroc": round(float(row["pooled_auroc"]), 4) if pd.notnull(row["pooled_auroc"]) else None,
            "bal_acc": bal_acc,
            "f1": round(float(row["pooled_f1"]), 4) if pd.notnull(row["pooled_f1"]) else None,
            "prec": round(float(row["pooled_precision"]), 4) if pd.notnull(row["pooled_precision"]) else None,
            "rec": round(float(row["pooled_recall"]), 4) if pd.notnull(row["pooled_recall"]) else None,
            "spec": round(float(row["pooled_specificity"]), 4) if pd.notnull(row["pooled_specificity"]) else None,
            "sens90": round(float(row["pooled_sens_at_spec_0.9"]), 4) if pd.notnull(row["pooled_sens_at_spec_0.9"]) else None,
            "wp_auroc": round(float(row["within_person_auroc_mean"]), 4) if pd.notnull(row["within_person_auroc_mean"]) else None,
            "wp_auroc_med": round(float(row["within_person_auroc_median"]), 4) if pd.notnull(row["within_person_auroc_median"]) else None,
            "fold_auroc": round(float(row["fold_test_auroc_mean"]), 4) if pd.notnull(row["fold_test_auroc_mean"]) else None,
            "fold_f1": round(float(row["fold_test_f1_mean"]), 4) if pd.notnull(row["fold_test_f1_mean"]) else None,
            "fold_spec": round(float(row["fold_test_spec_mean"]), 4) if pd.notnull(row["fold_test_spec_mean"]) else None,
        })

    json_data = json.dumps(records)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RQ1 Modality & Model Synergy Dashboard | Passive Sensing Wearables</title>
  <!-- TailwindCSS for responsive modern layout -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Plotly.js for publication-grade interactive data visualizations -->
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <!-- Google Fonts: Inter -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    body {{
      font-family: 'Inter', sans-serif;
      background-color: #0b0f19;
      color: #f1f5f9;
    }}
    .mono {{ font-family: 'JetBrains Mono', monospace; }}
    .glass-card {{
      background: rgba(17, 24, 39, 0.75);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }}
    .glow-indigo {{
      box-shadow: 0 0 25px -5px rgba(99, 102, 241, 0.3);
    }}
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
      width: 8px;
      height: 8px;
    }}
    ::-webkit-scrollbar-track {{
      background: #0f172a;
    }}
    ::-webkit-scrollbar-thumb {{
      background: #334155;
      border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
      background: #475569;
    }}
  </style>
</head>
<body class="min-h-screen p-4 md:p-8">

  <!-- TOP HEADER -->
  <header class="max-w-7xl mx-auto mb-8">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
      <div>
        <div class="flex items-center gap-2 mb-2">
          <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider bg-indigo-500/20 text-indigo-400 rounded-md border border-indigo-500/30">Research Question 1 & 2</span>
          <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 rounded-md border border-emerald-500/30">1,000 Walk-Forward Runs</span>
          <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider bg-amber-500/20 text-amber-400 rounded-md border border-amber-500/30">DualScaler + Hours Slept</span>
        </div>
        <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-indigo-300 bg-clip-text text-transparent">
          Wearable Modality Synergy & Architecture Evaluation
        </h1>
        <p class="text-sm md:text-base text-slate-400 mt-1 max-w-3xl">
          Comparative empirical evaluation of unimodal, bimodal, and trimodal passive wearable sensing streams across 6 machine learning model families and 20 rolling temporal resolutions.
        </p>
      </div>
      <div class="flex flex-wrap items-center gap-2">
        <label class="cursor-pointer px-3.5 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow-lg shadow-indigo-500/20 transition flex items-center gap-1.5">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
          <span>📂 Load Custom CSV</span>
          <input type="file" id="csv_file_input" accept=".csv" class="hidden" onchange="handleCsvUpload(event)">
        </label>
        <span id="data_source_badge" class="px-3 py-2 text-xs font-medium text-slate-300 bg-slate-800 rounded-lg border border-slate-700">Master Dataset (1,040 runs)</span>
        <button onclick="exportFilteredCSV()" class="px-3.5 py-2 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
          Export Filtered CSV
        </button>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto space-y-8">

    <!-- INTERACTIVE CONTROL PANEL -->
    <section class="glass-card rounded-2xl p-6 glow-indigo">
      <div class="flex flex-col gap-6">

        <!-- Top Controls: Model Selector & Quick Presets -->
        <div>
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
            <label class="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-1.5">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
              Select Models to Aggregate / Compare
            </label>
            <!-- Quick Preset Buttons -->
            <div class="flex flex-wrap gap-1.5">
              <button onclick="setPreset('all')" class="px-2.5 py-1 text-xs font-medium rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700 transition">All 6 Models</button>
              <button onclick="setPreset('tree')" class="px-2.5 py-1 text-xs font-medium rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700 transition">FLAML Tree (XGB + RF)</button>
              <button onclick="setPreset('xgb')" class="px-2.5 py-1 text-xs font-medium rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700 transition">XGBoost Only</button>
              <button onclick="setPreset('dl')" class="px-2.5 py-1 text-xs font-medium rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700 transition">Deep Learning (LSTM + Transf)</button>
              <button onclick="setPreset('top')" class="px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-900/60 text-indigo-300 hover:bg-indigo-800/80 transition border border-indigo-700/50">Top 3 (XGB, RF, Transf)</button>
            </div>
          </div>

          <!-- Model Checkboxes -->
          <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
            <label class="flex items-center gap-2 p-2.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 cursor-pointer transition select-none">
              <input type="checkbox" id="m_flaml_xgboost" class="model-cb text-indigo-500 rounded focus:ring-0 w-4 h-4" checked onchange="updateDashboard()">
              <span class="text-xs font-semibold text-slate-200">XGBoost (FLAML)</span>
            </label>
            <label class="flex items-center gap-2 p-2.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 cursor-pointer transition select-none">
              <input type="checkbox" id="m_flaml_rf" class="model-cb text-indigo-500 rounded focus:ring-0 w-4 h-4" checked onchange="updateDashboard()">
              <span class="text-xs font-semibold text-slate-200">Random Forest</span>
            </label>
            <label class="flex items-center gap-2 p-2.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 cursor-pointer transition select-none">
              <input type="checkbox" id="m_flaml_lr" class="model-cb text-indigo-500 rounded focus:ring-0 w-4 h-4" checked onchange="updateDashboard()">
              <span class="text-xs font-semibold text-slate-200">Logistic Reg.</span>
            </label>
            <label class="flex items-center gap-2 p-2.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 cursor-pointer transition select-none">
              <input type="checkbox" id="m_flaml_svm" class="model-cb text-indigo-500 rounded focus:ring-0 w-4 h-4" checked onchange="updateDashboard()">
              <span class="text-xs font-semibold text-slate-200">SVM (Linear/RBF)</span>
            </label>
            <label class="flex items-center gap-2 p-2.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 cursor-pointer transition select-none">
              <input type="checkbox" id="m_transformer" class="model-cb text-indigo-500 rounded focus:ring-0 w-4 h-4" checked onchange="updateDashboard()">
              <span class="text-xs font-semibold text-slate-200">Transformer</span>
            </label>
            <label class="flex items-center gap-2 p-2.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 cursor-pointer transition select-none">
              <input type="checkbox" id="m_lstm" class="model-cb text-indigo-500 rounded focus:ring-0 w-4 h-4" checked onchange="updateDashboard()">
              <span class="text-xs font-semibold text-slate-200">LSTM</span>
            </label>
          </div>
        </div>

        <!-- FEATURIZER EXCLUSION TOGGLE BAR (RQ1 FOCUS) -->
        <div class="p-3.5 rounded-xl bg-slate-900/90 border border-indigo-500/30 flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"/></svg>
            </div>
            <div>
              <div class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <span>Featurizer Exclusion Toggle</span>
                <span id="active_feat_badge" class="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">All Runs</span>
              </div>
              <div class="text-[11px] text-slate-400">Toggle off runs that used a featurizer to analyze pure raw sequences, or exclude RQ2 feature sweeps</div>
            </div>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            <button onclick="setFeaturizerMode('all')" id="btn_feat_all" class="px-3 py-1.5 text-xs font-bold rounded-lg bg-indigo-600 text-white shadow transition">
              All Runs
            </button>
            <button onclick="setFeaturizerMode('no_featurizer')" id="btn_feat_none" class="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 transition border border-slate-700 flex items-center gap-1.5">
              <span>🚫</span> <span>No Featurizer (Raw Seq Only)</span>
            </button>
            <button onclick="setFeaturizerMode('exclude_rq2')" id="btn_feat_excl_rq2" class="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 transition border border-slate-700 flex items-center gap-1.5">
              <span>🎯</span> <span>Pure RQ1 (Exclude RQ2 Featurizers)</span>
            </button>
            <button onclick="setFeaturizerMode('featurized_only')" id="btn_feat_only" class="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 transition border border-slate-700">
              Featurized Only
            </button>
          </div>
        </div>

        <!-- Filter Sub-Row: Metric, Sweeps, Featurizer, Lookback, Resample -->
        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4 pt-4 border-t border-slate-800">
          
          <!-- Metric Selector -->
          <div>
            <label class="block text-xs font-semibold text-slate-400 mb-1.5">Evaluation Metric</label>
            <select id="metric_select" onchange="updateDashboard()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2.5 focus:border-indigo-500 outline-none">
              <option value="auroc" selected>Pooled AUROC (Primary)</option>
              <option value="bal_acc">Pooled Balanced Accuracy</option>
              <option value="f1">Pooled Macro F1</option>
              <option value="prec">Pooled Macro Precision</option>
              <option value="rec">Pooled Recall (Sensitivity)</option>
              <option value="spec">Pooled Specificity</option>
              <option value="sens90">Pooled Sensitivity @ 90% Spec</option>
            </select>
          </div>

          <!-- Sweep Filter -->
          <div>
            <label class="block text-xs font-semibold text-slate-400 mb-1.5">W&B Sweep</label>
            <select id="sweep_select" onchange="updateDashboard()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2.5 focus:border-indigo-500 outline-none">
              <option value="all" selected>All Sweeps (1,000 runs)</option>
              <option value="9dvq8uiy">RQ1 FLAML Sweep (9dvq8uiy - 560 runs)</option>
              <option value="a6sl4omd">RQ1 Deep Learning (a6sl4omd - 280 runs)</option>
              <option value="zoznzd8y">RQ2 Feature Sweeps (zoznzd8y - 160 runs)</option>
            </select>
          </div>

          <!-- Featurizer Filter -->
          <div>
            <label class="block text-xs font-semibold text-slate-400 mb-1.5">Featurizer Extraction</label>
            <select id="feat_select" onchange="updateDashboard()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2.5 focus:border-indigo-500 outline-none">
              <option value="all" selected>All Featurizers</option>
              <option value="summary_stats">Summary Stats (5 stats)</option>
              <option value="catch22">Catch22 (22 dynamic stats)</option>
              <option value="none">None (Raw 3D Sequences - DL)</option>
            </select>
          </div>

          <!-- Lookback Filter -->
          <div>
            <label class="block text-xs font-semibold text-slate-400 mb-1.5">Lookback Window</label>
            <select id="lookback_select" onchange="updateDashboard()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2.5 focus:border-indigo-500 outline-none">
              <option value="all" selected>All Lookbacks (24h - 120h)</option>
              <option value="24">24 Hours (1 Day)</option>
              <option value="48">48 Hours (2 Days)</option>
              <option value="72">72 Hours (3 Days)</option>
              <option value="96">96 Hours (4 Days)</option>
              <option value="120">120 Hours (5 Days)</option>
            </select>
          </div>

          <!-- Resample Filter -->
          <div>
            <label class="block text-xs font-semibold text-slate-400 mb-1.5">Resample Frequency</label>
            <select id="resample_select" onchange="updateDashboard()" class="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2.5 focus:border-indigo-500 outline-none">
              <option value="all" selected>All Frequencies (4h - 12h)</option>
              <option value="4">4-Hour Bins</option>
              <option value="6">6-Hour Bins</option>
              <option value="8">8-Hour Bins</option>
              <option value="12">12-Hour Bins</option>
            </select>
          </div>

        </div>

      </div>
    </section>

    <!-- KEY STATS KPI CARDS -->
    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="glass-card rounded-xl p-5 border-l-4 border-emerald-500">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Top Modality (RQ1)</span>
        <div class="text-2xl font-black text-white mt-1" id="kpi_top_modality">-</div>
        <div class="text-xs text-emerald-400 mt-1 font-medium" id="kpi_top_modality_score">-</div>
      </div>

      <div class="glass-card rounded-xl p-5 border-l-4 border-indigo-500">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Peak Single Run</span>
        <div class="text-2xl font-black text-white mt-1" id="kpi_peak_score">-</div>
        <div class="text-xs text-indigo-300 mt-1 truncate" id="kpi_peak_desc">-</div>
      </div>

      <div class="glass-card rounded-xl p-5 border-l-4 border-purple-500">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Modality Synergy Gain</span>
        <div class="text-2xl font-black text-white mt-1" id="kpi_synergy_gain">-</div>
        <div class="text-xs text-purple-300 mt-1" id="kpi_synergy_desc">Multimodal vs Best Unimodal</div>
      </div>

      <div class="glass-card rounded-xl p-5 border-l-4 border-cyan-500">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Active Evaluated Runs</span>
        <div class="text-2xl font-black text-white mt-1" id="kpi_active_runs">-</div>
        <div class="text-xs text-slate-400 mt-1" id="kpi_total_runs">out of 1,000 total runs</div>
      </div>
    </section>

    <!-- PRIMARY RQ1 CHART: MODALITY RANKING WITH ERROR BARS -->
    <section class="glass-card rounded-2xl p-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4">
        <div>
          <h2 class="text-xl font-bold text-white flex items-center gap-2">
            <span>RQ1: Modality Performance Hierarchy</span>
            <span class="text-xs font-normal text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700" id="ranking_metric_label">Pooled AUROC</span>
          </h2>
          <p class="text-xs text-slate-400">
            Average metric value $\\pm$ 1 standard deviation across rolling window configurations for selected model families.
          </p>
        </div>
        <div class="flex items-center gap-3 text-xs">
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-blue-500"></span> Unimodal (1)</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-indigo-500"></span> Bimodal (2)</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-500"></span> Trimodal (3)</span>
        </div>
      </div>
      <div id="plot_modality_ranking" class="w-full h-80"></div>
    </section>

    <!-- DUAL COLUMN: DISTRIBUTION BOXPLOTS & MODEL BREAKDOWN -->
    <section class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      
      <!-- Modality Boxplots -->
      <div class="glass-card rounded-2xl p-6">
        <h3 class="text-lg font-bold text-white mb-1">Performance Distribution by Modality</h3>
        <p class="text-xs text-slate-400 mb-4">Boxplots illustrating medians, quartiles, and sampling variance.</p>
        <div id="plot_modality_boxplots" class="w-full h-80"></div>
      </div>

      <!-- Model x Modality Breakdown -->
      <div class="glass-card rounded-2xl p-6">
        <h3 class="text-lg font-bold text-white mb-1">Model Family Sensitivity</h3>
        <p class="text-xs text-slate-400 mb-4">Comparing each model's response across the sensor combinations.</p>
        <div id="plot_model_modality_bars" class="w-full h-80"></div>
      </div>

    </section>

    <!-- PAIRWISE STATISTICAL SIGNIFICANCE MATRIX -->
    <section class="glass-card rounded-2xl p-6">
      <div class="mb-4">
        <h3 class="text-lg font-bold text-white">Statistical Synergy & Paired Differences</h3>
        <p class="text-xs text-slate-400">
          Head-to-head comparison against the #1 ranked modality on identical rolling sampler windows (paired samples).
        </p>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs border-collapse">
          <thead>
            <tr class="border-b border-slate-700 text-slate-400 uppercase tracking-wider font-semibold">
              <th class="py-3 px-4">Rank</th>
              <th class="py-3 px-4">Modality</th>
              <th class="py-3 px-4">Complexity</th>
              <th class="py-3 px-4">Mean Score</th>
              <th class="py-3 px-4">Median</th>
              <th class="py-3 px-4">Std Dev</th>
              <th class="py-3 px-4">Peak Run</th>
              <th class="py-3 px-4">&Delta; vs Top Modality</th>
              <th class="py-3 px-4">Paired t-test p-value</th>
              <th class="py-3 px-4">Significance</th>
            </tr>
          </thead>
          <tbody id="stats_table_body" class="divide-y divide-slate-800 text-slate-200">
            <!-- Dynamically populated -->
          </tbody>
        </table>
      </div>
    </section>

    <!-- TEMPORAL DYNAMICS HEATMAP (RQ3) -->
    <section class="glass-card rounded-2xl p-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4">
        <div>
          <h3 class="text-lg font-bold text-white">RQ3: Temporal Dynamics Surface</h3>
          <p class="text-xs text-slate-400">Interaction between Lookback Duration and Resampling Frequency with decoupled model & featurizer selection.</p>
        </div>
        <div class="flex flex-wrap items-center gap-3">
          <div class="flex items-center gap-2">
            <label class="text-xs text-slate-400 font-medium">Modality:</label>
            <select id="heatmap_modality_select" onchange="renderTemporalHeatmap()" class="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2 outline-none">
              <!-- Dynamically populated with modalities -->
            </select>
          </div>
          <div class="flex items-center gap-2">
            <label class="text-xs text-slate-400 font-medium">Model Variant:</label>
            <select id="heatmap_model_select" onchange="renderTemporalHeatmap()" class="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2 outline-none">
              <!-- Dynamically populated with decoupled model+featurizer variants -->
            </select>
          </div>
          <div class="flex items-center gap-2">
            <label class="text-xs text-slate-400 font-medium">Metric:</label>
            <select id="heatmap_metric_select" onchange="renderTemporalHeatmap()" class="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg p-2 outline-none">
              <option value="auroc">Pooled AUROC</option>
              <option value="bal_acc">Pooled Balanced Accuracy</option>
              <option value="f1">Pooled Macro F1</option>
              <option value="prec">Pooled Macro Precision</option>
              <option value="rec">Pooled Recall (Sensitivity)</option>
              <option value="spec">Pooled Specificity</option>
              <option value="sens90">Pooled Sensitivity @ 90% Spec</option>
            </select>
          </div>
        </div>
      </div>
      <div id="plot_temporal_heatmap" class="w-full h-80"></div>
    </section>

    <!-- DETAILED RUN TABLE -->
    <section id="table-section" class="glass-card rounded-2xl p-6">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <h3 class="text-lg font-bold text-white">Evaluated Runs Explorer</h3>
          <p class="text-xs text-slate-400">Browse and inspect individual walk-forward runs matching your filter criteria.</p>
        </div>
        <div class="flex items-center gap-2">
          <input type="text" id="table_search" placeholder="Search sampler, model, etc..." onkeyup="filterTable()" class="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 outline-none w-64">
        </div>
      </div>
      <div class="overflow-x-auto max-h-96">
        <table class="w-full text-left text-xs border-collapse" id="runs_table">
          <thead class="sticky top-0 bg-slate-900 border-b border-slate-700 text-slate-400 uppercase font-semibold">
            <tr>
              <th class="py-2.5 px-3">Run Name</th>
              <th class="py-2.5 px-3">Model</th>
              <th class="py-2.5 px-3">Modality</th>
              <th class="py-2.5 px-3">Lookback</th>
              <th class="py-2.5 px-3">Freq</th>
              <th class="py-2.5 px-3">Featurizer</th>
              <th class="py-2.5 px-3 font-bold text-amber-400 cursor-pointer" id="table_active_metric_th" onclick="sortActiveMetric()">Selected Metric &uarr;&darr;</th>
              <th class="py-2.5 px-3 font-bold text-indigo-400 cursor-pointer" onclick="sortTable('auroc')">AUROC &uarr;&darr;</th>
              <th class="py-2.5 px-3 cursor-pointer text-emerald-400" onclick="sortTable('bal_acc')">Bal Acc &uarr;&darr;</th>
              <th class="py-2.5 px-3 cursor-pointer" onclick="sortTable('f1')">F1 &uarr;&darr;</th>
              <th class="py-2.5 px-3 cursor-pointer" onclick="sortTable('prec')">Prec &uarr;&darr;</th>
              <th class="py-2.5 px-3 cursor-pointer" onclick="sortTable('rec')">Rec &uarr;&darr;</th>
              <th class="py-2.5 px-3 cursor-pointer" onclick="sortTable('spec')">Spec &uarr;&darr;</th>
              <th class="py-2.5 px-3 cursor-pointer" onclick="sortTable('sens90')">Sens@90% &uarr;&darr;</th>
            </tr>
          </thead>
          <tbody id="runs_table_body" class="divide-y divide-slate-800 text-slate-300">
            <!-- Dynamically populated -->
          </tbody>
        </table>
      </div>
      <div class="mt-3 flex items-center justify-between text-xs text-slate-400">
        <span id="table_pagination_info">Showing 0 of 0 runs</span>
        <div class="flex items-center gap-1.5" id="pagination_controls">
          <!-- Pagination buttons -->
        </div>
      </div>
    </section>

  </main>

  <footer class="max-w-7xl mx-auto mt-12 pt-6 border-t border-slate-800 text-center text-xs text-slate-500">
    Passive Sensing Wearables Research &bull; Cyclical Walk-Forward Validation &bull; 1,000 W&B Sweeps Processed
  </footer>

  <!-- DATA & APPLICATION LOGIC -->
  <script>
    const ALL_RUNS = {json_data};
    let filteredRuns = [];
    let currentTableData = [];
    let currentPage = 1;
    const PAGE_SIZE = 25;
    let currentSortCol = 'auroc';
    let sortAsc = false;

    // Modality display metadata
    const MODALITY_META = {{
      'step': {{ name: 'Step Count', complexity: 1, color: '#3b82f6' }},
      'calorie': {{ name: 'Calories', complexity: 1, color: '#3b82f6' }},
      'distance': {{ name: 'Distance', complexity: 1, color: '#3b82f6' }},
      'step_calorie': {{ name: 'Step + Calorie', complexity: 2, color: '#8b5cf6' }},
      'step_distance': {{ name: 'Step + Distance', complexity: 2, color: '#8b5cf6' }},
      'calorie_distance': {{ name: 'Calorie + Distance', complexity: 2, color: '#8b5cf6' }},
      'step_calorie_distance': {{ name: 'Step + Cal + Dist', complexity: 3, color: '#10b981' }}
    }};

    function setPreset(type) {{
      const checkboxes = {{
        'flaml_xgboost': document.getElementById('m_flaml_xgboost'),
        'flaml_rf': document.getElementById('m_flaml_rf'),
        'flaml_lr': document.getElementById('m_flaml_lr'),
        'flaml_svm': document.getElementById('m_flaml_svm'),
        'transformer': document.getElementById('m_transformer'),
        'lstm': document.getElementById('m_lstm')
      }};

      // Reset all
      Object.values(checkboxes).forEach(cb => cb.checked = false);

      if (type === 'all') {{
        Object.values(checkboxes).forEach(cb => cb.checked = true);
      }} else if (type === 'tree') {{
        checkboxes['flaml_xgboost'].checked = true;
        checkboxes['flaml_rf'].checked = true;
      }} else if (type === 'xgb') {{
        checkboxes['flaml_xgboost'].checked = true;
      }} else if (type === 'dl') {{
        checkboxes['transformer'].checked = true;
        checkboxes['lstm'].checked = true;
      }} else if (type === 'top') {{
        checkboxes['flaml_xgboost'].checked = true;
        checkboxes['flaml_rf'].checked = true;
        checkboxes['transformer'].checked = true;
      }}
      updateDashboard();
    }}

    function getSelectedModels() {{
      const models = [];
      if (document.getElementById('m_flaml_xgboost').checked) models.push('flaml_xgboost');
      if (document.getElementById('m_flaml_rf').checked) models.push('flaml_rf');
      if (document.getElementById('m_flaml_lr').checked) models.push('flaml_lr');
      if (document.getElementById('m_flaml_svm').checked) models.push('flaml_svm');
      if (document.getElementById('m_transformer').checked) models.push('transformer');
      if (document.getElementById('m_lstm').checked) models.push('lstm');
      return models;
    }}

    let currentFeatMode = 'all'; // 'all', 'no_featurizer', 'exclude_rq2', 'featurized_only'

    function setFeaturizerMode(mode) {{
      currentFeatMode = mode;

      const buttons = {{
        'all': document.getElementById('btn_feat_all'),
        'no_featurizer': document.getElementById('btn_feat_none'),
        'exclude_rq2': document.getElementById('btn_feat_excl_rq2'),
        'featurized_only': document.getElementById('btn_feat_only')
      }};

      const activeBadge = document.getElementById('active_feat_badge');

      Object.entries(buttons).forEach(([k, btn]) => {{
        if (!btn) return;
        if (k === mode) {{
          btn.className = "px-3 py-1.5 text-xs font-bold rounded-lg bg-indigo-600 text-white shadow transition";
        }} else {{
          btn.className = "px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 transition border border-slate-700";
        }}
      }});

      if (mode === 'all') {{
        activeBadge.innerText = 'All Runs';
        activeBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30';
      }} else if (mode === 'no_featurizer') {{
        activeBadge.innerText = '🚫 Featurizers Excluded (Raw Seq)';
        activeBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
        // Auto-select DL models if none were checked
        const anyDLChecked = document.getElementById('m_transformer').checked || document.getElementById('m_lstm').checked;
        if (!anyDLChecked) {{
          document.getElementById('m_transformer').checked = true;
          document.getElementById('m_lstm').checked = true;
        }}
      }} else if (mode === 'exclude_rq2') {{
        activeBadge.innerText = '🎯 Pure RQ1 Only (No RQ2)';
        activeBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30';
      }} else if (mode === 'featurized_only') {{
        activeBadge.innerText = 'Featurized Tabular Only';
        activeBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30';
      }}

      updateDashboard();
    }}

    function filterData() {{
      const selectedModels = getSelectedModels();
      const sweepFilter = document.getElementById('sweep_select').value;
      const featFilter = document.getElementById('feat_select').value;
      const lookbackFilter = document.getElementById('lookback_select').value;
      const resampleFilter = document.getElementById('resample_select').value;
      const metricKey = document.getElementById('metric_select').value;

      filteredRuns = ALL_RUNS.filter(r => {{
        // Featurizer Mode Toggle
        if (currentFeatMode === 'no_featurizer') {{
          // Strictly exclude any run that used a featurizer (keep only raw sequences)
          if (!r.featurizer.toLowerCase().includes('none')) return false;
        }} else if (currentFeatMode === 'exclude_rq2') {{
          // Strictly exclude RQ2 featurizer sweep (zoznzd8y) and catch22
          if (r.sweep === 'zoznzd8y' || r.featurizer.toLowerCase().includes('catch22')) return false;
        }} else if (currentFeatMode === 'featurized_only') {{
          // Strictly exclude raw sequence runs
          if (r.featurizer.toLowerCase().includes('none')) return false;
        }}

        // Model match
        if (!selectedModels.includes(r.model)) return false;
        // Sweep match
        if (sweepFilter !== 'all' && r.sweep !== sweepFilter) return false;
        // Featurizer match
        if (featFilter !== 'all' && !r.featurizer.toLowerCase().includes(featFilter.toLowerCase())) return false;
        // Lookback match
        if (lookbackFilter !== 'all' && r.lookback !== parseInt(lookbackFilter)) return false;
        // Resample match
        if (resampleFilter !== 'all' && r.resample !== parseInt(resampleFilter)) return false;
        // Must have valid metric
        if (r[metricKey] === null || r[metricKey] === undefined) return false;
        return true;
      }});
    }}

    function computeStatsByModality(metricKey) {{
      const groups = {{}};
      filteredRuns.forEach(r => {{
        const mod = r.modality;
        if (!groups[mod]) groups[mod] = [];
        groups[mod].push(r[metricKey]);
      }});

      const stats = [];
      for (const [mod, values] of Object.entries(groups)) {{
        if (values.length === 0) continue;
        const n = values.length;
        const mean = values.reduce((a, b) => a + b, 0) / n;
        const sorted = [...values].sort((a, b) => a - b);
        const median = n % 2 === 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2 : sorted[Math.floor(n/2)];
        const min = sorted[0];
        const max = sorted[sorted.length - 1];
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n > 1 ? n - 1 : 1);
        const std = Math.sqrt(variance);

        stats.push({{
          modality: mod,
          displayName: MODALITY_META[mod]?.name || mod,
          complexity: MODALITY_META[mod]?.complexity || 1,
          color: MODALITY_META[mod]?.color || '#6366f1',
          n: n,
          mean: mean,
          median: median,
          std: std,
          min: min,
          max: max,
          values: values
        }});
      }}

      // Sort by mean descending
      stats.sort((a, b) => b.mean - a.mean);
      return stats;
    }}

    function updateKPIs(stats, metricKey) {{
      document.getElementById('kpi_active_runs').innerText = filteredRuns.length.toLocaleString();
      document.getElementById('kpi_total_runs').innerText = `out of ${{ALL_RUNS.length.toLocaleString()}} total runs`;

      if (stats.length === 0) {{
        document.getElementById('kpi_top_modality').innerText = 'No Data';
        document.getElementById('kpi_top_modality_score').innerText = '-';
        document.getElementById('kpi_peak_score').innerText = '-';
        document.getElementById('kpi_peak_desc').innerText = '-';
        document.getElementById('kpi_synergy_gain').innerText = '-';
        return;
      }}

      const top = stats[0];
      document.getElementById('kpi_top_modality').innerText = top.displayName;
      document.getElementById('kpi_top_modality_score').innerText = `Mean: ${{top.mean.toFixed(4)}} ± ${{top.std.toFixed(4)}}`;

      // Find overall peak single run
      let peakRun = filteredRuns[0];
      filteredRuns.forEach(r => {{
        if (r[metricKey] > peakRun[metricKey]) peakRun = r;
      }});

      document.getElementById('kpi_peak_score').innerText = peakRun[metricKey].toFixed(4);
      document.getElementById('kpi_peak_desc').innerText = `${{peakRun.model}} | ${{peakRun.modality}} (${{peakRun.sampler}})`;

      // Modality Synergy: compare top multimodal (complexity >= 2) vs top unimodal (complexity == 1)
      const topMultimodal = stats.find(s => s.complexity >= 2);
      const topUnimodal = stats.find(s => s.complexity === 1);

      if (topMultimodal && topUnimodal) {{
        const delta = topMultimodal.mean - topUnimodal.mean;
        const sign = delta >= 0 ? '+' : '';
        document.getElementById('kpi_synergy_gain').innerText = `${{sign}}${{(delta * 100).toFixed(2)}}%`;
        document.getElementById('kpi_synergy_desc').innerText = `${{topMultimodal.displayName}} vs ${{topUnimodal.displayName}}`;
      }} else {{
        document.getElementById('kpi_synergy_gain').innerText = 'N/A';
      }}
    }}

    function renderRankingChart(stats, metricLabel) {{
      const labels = stats.map(s => s.displayName);
      const means = stats.map(s => s.mean);
      const stds = stats.map(s => s.std);
      const colors = stats.map(s => s.color);

      const trace = {{
        x: labels,
        y: means,
        error_y: {{
          type: 'data',
          array: stds,
          visible: true,
          color: 'rgba(255,255,255,0.4)',
          thickness: 1.5
        }},
        type: 'bar',
        marker: {{
          color: colors,
          opacity: 0.85,
          line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }}
        }},
        text: means.map(m => m.toFixed(4)),
        textposition: 'auto',
        hoverinfo: 'x+y+text',
      }};

      const layout = {{
        margin: {{ t: 20, r: 20, l: 50, b: 60 }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ family: 'Inter, sans-serif', color: '#94a3b8' }},
        yaxis: {{
          title: metricLabel,
          gridcolor: 'rgba(255, 255, 255, 0.05)',
          zeroline: false
        }},
        xaxis: {{
          gridcolor: 'transparent'
        }}
      }};

      Plotly.newPlot('plot_modality_ranking', [trace], layout, {{ responsive: true, displayModeBar: false }});
    }}

    function renderBoxplots(stats, metricKey, metricLabel) {{
      const traces = stats.map(s => ({{
        y: s.values,
        name: s.displayName,
        type: 'box',
        boxpoints: 'all',
        jitter: 0.3,
        pointpos: -1.8,
        marker: {{ color: s.color, size: 3 }},
        line: {{ width: 1.5 }}
      }}));

      const layout = {{
        margin: {{ t: 20, r: 20, l: 50, b: 60 }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ family: 'Inter, sans-serif', color: '#94a3b8' }},
        yaxis: {{
          title: metricLabel,
          gridcolor: 'rgba(255, 255, 255, 0.05)',
          zeroline: false
        }},
        xaxis: {{
          gridcolor: 'transparent'
        }},
        showlegend: false
      }};

      Plotly.newPlot('plot_modality_boxplots', traces, layout, {{ responsive: true, displayModeBar: false }});
    }}

    function renderModelModalityBars(stats, metricKey, metricLabel) {{
      const selectedModels = getSelectedModels();
      const modalities = stats.map(s => s.modality);
      const modLabels = stats.map(s => s.displayName);

      const traces = selectedModels.map(model => {{
        const yVals = modalities.map(mod => {{
          const subset = filteredRuns.filter(r => r.model === model && r.modality === mod);
          if (subset.length === 0) return null;
          const sum = subset.reduce((acc, cur) => acc + cur[metricKey], 0);
          return sum / subset.length;
        }});

        return {{
          x: modLabels,
          y: yVals,
          name: model.replace('flaml_', ''),
          type: 'bar'
        }};
      }});

      const layout = {{
        barmode: 'group',
        margin: {{ t: 20, r: 20, l: 50, b: 60 }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ family: 'Inter, sans-serif', color: '#94a3b8' }},
        yaxis: {{
          title: metricLabel,
          gridcolor: 'rgba(255, 255, 255, 0.05)',
          zeroline: false
        }},
        xaxis: {{
          gridcolor: 'transparent'
        }},
        legend: {{
          orientation: 'h',
          y: 1.15,
          font: {{ size: 10 }}
        }}
      }};

      Plotly.newPlot('plot_model_modality_bars', traces, layout, {{ responsive: true, displayModeBar: false }});
    }}

    function renderSignificanceTable(stats, metricKey) {{
      const tbody = document.getElementById('stats_table_body');
      tbody.innerHTML = '';

      if (stats.length === 0) return;

      const top = stats[0];

      // Build map of top modality by (sampler, model) for paired testing
      const topPairs = {{}};
      filteredRuns.filter(r => r.modality === top.modality).forEach(r => {{
        const key = `${{r.model}}_${{r.sampler}}`;
        topPairs[key] = r[metricKey];
      }});

      stats.forEach((s, idx) => {{
        let deltaStr = '-';
        let pValStr = '-';
        let sigBadge = '<span class="text-slate-500">-</span>';

        if (idx > 0) {{
          const pairedDiffs = [];
          filteredRuns.filter(r => r.modality === s.modality).forEach(r => {{
            const key = `${{r.model}}_${{r.sampler}}`;
            if (topPairs[key] !== undefined) {{
              pairedDiffs.push(r[metricKey] - topPairs[key]);
            }}
          }});

          if (pairedDiffs.length > 2) {{
            const meanDiff = pairedDiffs.reduce((a, b) => a + b, 0) / pairedDiffs.length;
            deltaStr = `${{meanDiff >= 0 ? '+' : ''}}${{meanDiff.toFixed(4)}}`;

            // Paired t-test
            const varDiff = pairedDiffs.reduce((a, b) => a + Math.pow(b - meanDiff, 2), 0) / (pairedDiffs.length - 1);
            const seDiff = Math.sqrt(varDiff / pairedDiffs.length);
            const tStat = seDiff > 0 ? meanDiff / seDiff : 0;
            // Approximate p-value from t-stat
            const pVal = 2 * (1 - normalCdf(Math.abs(tStat)));
            pValStr = pVal < 0.001 ? '< 0.001' : pVal.toFixed(4);

            if (pVal < 0.001) {{
              sigBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">p &lt; 0.001 ***</span>';
            }} else if (pVal < 0.05) {{
              sigBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">p &lt; 0.05 *</span>';
            }} else {{
              sigBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400">ns</span>';
            }}
          }}
        }} else {{
          deltaStr = 'Reference (#1)';
          sigBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Winner</span>';
        }}

        const complexityBadge = s.complexity === 1 ? '<span class="text-blue-400">Unimodal (1)</span>' :
                               s.complexity === 2 ? '<span class="text-indigo-400">Bimodal (2)</span>' :
                               '<span class="text-emerald-400">Trimodal (3)</span>';

        const row = document.createElement('tr');
        row.className = 'hover:bg-slate-800/40 transition';
        row.innerHTML = `
          <td class="py-3 px-4 font-bold text-slate-400">#${{idx + 1}}</td>
          <td class="py-3 px-4 font-semibold text-white">${{s.displayName}}</td>
          <td class="py-3 px-4 font-medium">${{complexityBadge}}</td>
          <td class="py-3 px-4 font-bold text-indigo-300 mono">${{s.mean.toFixed(4)}}</td>
          <td class="py-3 px-4 mono text-slate-400">${{s.median.toFixed(4)}}</td>
          <td class="py-3 px-4 mono text-slate-400">±${{s.std.toFixed(4)}}</td>
          <td class="py-3 px-4 mono text-emerald-400">${{s.max.toFixed(4)}}</td>
          <td class="py-3 px-4 mono font-semibold ${{deltaStr.startsWith('-') ? 'text-rose-400' : 'text-emerald-400'}}">${{deltaStr}}</td>
          <td class="py-3 px-4 mono text-slate-400">${{pValStr}}</td>
          <td class="py-3 px-4">${{sigBadge}}</td>
        `;
        tbody.appendChild(row);
      }});
    }}

    function normalCdf(x) {{
      const t = 1 / (1 + 0.2316419 * Math.abs(x));
      const d = 0.3989423 * Math.exp(-x * x / 2);
      const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
      return x > 0 ? 1 - prob : prob;
    }}

    function populateHeatmapDropdown(stats) {{
      const select = document.getElementById('heatmap_modality_select');
      const prevVal = select.value;
      select.innerHTML = '<option value="all">All Modalities (Combined Average)</option>';
      stats.forEach(s => {{
        const opt = document.createElement('option');
        opt.value = s.modality;
        opt.innerText = s.displayName;
        select.appendChild(opt);
      }});
      if (prevVal && (prevVal === 'all' || stats.some(s => s.modality === prevVal))) {{
        select.value = prevVal;
      }} else if (stats.length > 0) {{
        select.value = stats[0].modality;
      }}

      // Populate decoupled model variants dropdown for RQ3
      const modelSelect = document.getElementById('heatmap_model_select');
      const prevModel = modelSelect.value;
      const modelFeats = [...new Set(RAW_DATA.map(r => r.m_feat))].filter(Boolean).sort();
      modelSelect.innerHTML = '<option value="all">All Model Variants Combined</option>';
      modelFeats.forEach(mf => {{
        const opt = document.createElement('option');
        opt.value = mf;
        opt.innerText = mf;
        modelSelect.appendChild(opt);
      }});
      if (prevModel && (prevModel === 'all' || modelFeats.includes(prevModel))) {{
        modelSelect.value = prevModel;
      }} else {{
        modelSelect.value = 'all';
      }}
    }}

    function renderTemporalHeatmap() {{
      const mod = document.getElementById('heatmap_modality_select').value;
      const modelChoice = document.getElementById('heatmap_model_select').value;
      const metricSelectEl = document.getElementById('heatmap_metric_select') || document.getElementById('metric_select');
      const metricKey = metricSelectEl.value;
      const metricLabel = metricSelectEl.options[metricSelectEl.selectedIndex].text;

      const lookbacks = [24, 48, 72, 96, 120];
      const resamples = [4, 6, 8, 12];

      const zValues = [];
      lookbacks.forEach(lb => {{
        const row = [];
        resamples.forEach(res => {{
          const matching = RAW_DATA.filter(r => 
            (mod === 'all' || r.modality === mod) && 
            (modelChoice === 'all' || r.m_feat === modelChoice) && 
            r.lookback === lb && 
            r.resample === res && 
            r[metricKey] !== null && 
            r[metricKey] !== undefined
          );
          if (matching.length === 0) {{
            row.push(null);
          }} else {{
            const avg = matching.reduce((a, b) => a + b[metricKey], 0) / matching.length;
            row.push(avg);
          }}
        }});
        zValues.push(row);
      }});

      const trace = {{
        z: zValues,
        x: resamples.map(r => `${{r}}h Bins`),
        y: lookbacks.map(l => `${{l}}h Lookback`),
        type: 'heatmap',
        colorscale: 'Viridis',
        hoverongaps: false
      }};

      const layout = {{
        margin: {{ t: 20, r: 20, l: 90, b: 40 }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ family: 'Inter, sans-serif', color: '#94a3b8' }},
        xaxis: {{ title: 'Resample Bin Width' }},
        yaxis: {{ title: 'Lookback Window' }}
      }};

      Plotly.newPlot('plot_temporal_heatmap', [trace], layout, {{ responsive: true, displayModeBar: false }});
    }}

    function updateTable() {{
      currentTableData = [...filteredRuns];
      sortCurrentTable();
      renderTablePage();
    }}

    function sortTable(col) {{
      if (currentSortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        currentSortCol = col;
        sortAsc = false;
      }}
      sortCurrentTable();
      renderTablePage();
    }}

    function sortCurrentTable() {{
      currentTableData.sort((a, b) => {{
        const vA = a[currentSortCol];
        const vB = b[currentSortCol];
        if (vA === null) return 1;
        if (vB === null) return -1;
        return sortAsc ? (vA > vB ? 1 : -1) : (vA < vB ? 1 : -1);
      }});
    }}

    function filterTable() {{
      const q = document.getElementById('table_search').value.toLowerCase();
      currentTableData = filteredRuns.filter(r => {{
        return r.run_name.toLowerCase().includes(q) ||
               r.model.toLowerCase().includes(q) ||
               r.modality.toLowerCase().includes(q) ||
               r.sampler.toLowerCase().includes(q) ||
               r.featurizer.toLowerCase().includes(q);
      }});
      currentPage = 1;
      renderTablePage();
    }}

    function renderTablePage() {{
      const tbody = document.getElementById('runs_table_body');
      tbody.innerHTML = '';

      const total = currentTableData.length;
      const start = (currentPage - 1) * PAGE_SIZE;
      const end = Math.min(start + PAGE_SIZE, total);
      const pageSlice = currentTableData.slice(start, end);

      const activeMetricKey = document.getElementById('metric_select').value;
      pageSlice.forEach(r => {{
        const row = document.createElement('tr');
        row.className = 'hover:bg-slate-800/50 transition';
        const activeVal = r[activeMetricKey] !== null && r[activeMetricKey] !== undefined ? r[activeMetricKey].toFixed(4) : '-';
        row.innerHTML = `
          <td class="py-2.5 px-3 font-medium text-slate-200 truncate max-w-[200px]" title="${{r.run_name}}">${{r.run_name}}</td>
          <td class="py-2.5 px-3 font-semibold text-slate-300">${{r.model.replace('flaml_', '')}}</td>
          <td class="py-2.5 px-3 font-medium text-indigo-300">${{r.modality}}</td>
          <td class="py-2.5 px-3 mono text-slate-400">${{r.lookback}}h</td>
          <td class="py-2.5 px-3 mono text-slate-400">${{r.resample}}h</td>
          <td class="py-2.5 px-3 text-slate-400 truncate max-w-[120px]" title="${{r.featurizer}}">${{r.featurizer.split(' ')[0]}}</td>
          <td class="py-2.5 px-3 mono font-bold text-amber-400 bg-amber-400/5">${{activeVal}}</td>
          <td class="py-2.5 px-3 mono font-bold text-indigo-400">${{r.auroc !== null ? r.auroc.toFixed(4) : '-'}}</td>
          <td class="py-2.5 px-3 mono text-emerald-400">${{r.bal_acc !== null && r.bal_acc !== undefined ? r.bal_acc.toFixed(4) : '-'}}</td>
          <td class="py-2.5 px-3 mono text-slate-300">${{r.f1 !== null ? r.f1.toFixed(4) : '-'}}</td>
          <td class="py-2.5 px-3 mono text-slate-300">${{r.prec !== null ? r.prec.toFixed(4) : '-'}}</td>
          <td class="py-2.5 px-3 mono text-slate-300">${{r.rec !== null ? r.rec.toFixed(4) : '-'}}</td>
          <td class="py-2.5 px-3 mono text-slate-300">${{r.spec !== null ? r.spec.toFixed(4) : '-'}}</td>
          <td class="py-2.5 px-3 mono text-slate-400">${{r.sens90 !== null ? r.sens90.toFixed(4) : '-'}}</td>
        `;
        tbody.appendChild(row);
      }});

      document.getElementById('table_pagination_info').innerText = `Showing ${{total > 0 ? start + 1 : 0}} to ${{end}} of ${{total}} runs`;

      const pageControls = document.getElementById('pagination_controls');
      const maxPages = Math.ceil(total / PAGE_SIZE) || 1;
      pageControls.innerHTML = `
        <button onclick="changePage(currentPage - 1)" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed" ${{currentPage <= 1 ? 'disabled' : ''}}>Prev</button>
        <span class="px-2 font-medium text-slate-300">${{currentPage}} / ${{maxPages}}</span>
        <button onclick="changePage(currentPage + 1)" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed" ${{currentPage >= maxPages ? 'disabled' : ''}}>Next</button>
      `;
    }}

    function changePage(p) {{
      const maxPages = Math.ceil(currentTableData.length / PAGE_SIZE) || 1;
      if (p < 1 || p > maxPages) return;
      currentPage = p;
      renderTablePage();
    }}

    function handleCsvUpload(event) {{
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(e) {{
        const text = e.target.result;
        try {{
          const lines = text.trim().split(/\\r?\\n/);
          if (lines.length < 2) {{
            alert('Uploaded CSV file contains no data.');
            return;
          }}
          const headers = lines[0].split(',').map(h => h.trim());
          const newRecords = [];

          for (let i = 1; i < lines.length; i++) {{
            const line = lines[i].trim();
            if (!line) continue;
            const vals = line.split(',');
            const row = {{}};
            headers.forEach((h, idx) => {{
              row[h] = vals[idx] !== undefined ? vals[idx].trim() : '';
            }});

            const m = row.model_pkg || '';
            const f = row.featurizer_pkg || '';
            const fNorm = (!f || f === 'none' || f === 'none (raw sequences)' || f === 'nan') ? 'raw' : f;
            const mClean = m.replace('flaml_', '');
            const mFeat = (fNorm === 'raw') ? `${{mClean}} (raw)` : `${{mClean}} + ${{fNorm}}`;

            const aurocVal = row.pooled_auroc && !isNaN(row.pooled_auroc) ? parseFloat(row.pooled_auroc) : null;
            const f1Val = row.pooled_f1 && !isNaN(row.pooled_f1) ? parseFloat(row.pooled_f1) : null;
            const precVal = row.pooled_precision && !isNaN(row.pooled_precision) ? parseFloat(row.pooled_precision) : null;
            const recVal = row.pooled_recall && !isNaN(row.pooled_recall) ? parseFloat(row.pooled_recall) : null;
            const specVal = row.pooled_specificity && !isNaN(row.pooled_specificity) ? parseFloat(row.pooled_specificity) : null;
            const sens90Val = row.pooled_sens_at_spec_0.9 && !isNaN(row.pooled_sens_at_spec_0.9) ? parseFloat(row.pooled_sens_at_spec_0.9) : null;

            let balAccVal = null;
            if (row.pooled_balanced_accuracy && !isNaN(row.pooled_balanced_accuracy)) {{
              balAccVal = parseFloat(row.pooled_balanced_accuracy);
            }} else if (recVal !== null && specVal !== null) {{
              balAccVal = (recVal + specVal) / 2.0;
            }}

            let lb = row.lookback_hours ? parseInt(row.lookback_hours) : null;
            let rf = row.resample_freq_hours ? parseInt(row.resample_freq_hours) : null;
            if ((!lb || !rf) && row.sampler_pkg) {{
              const p = row.sampler_pkg.replace('rolling_', '').split('_');
              if (p.length >= 2) {{
                lb = parseInt(p[0].replace('h', ''));
                rf = parseInt(p[1].replace('h', ''));
              }}
            }}

            newRecords.push({{
              sweep: row.sweep_id || 'custom',
              exp: row.experiment || 'custom',
              type: row.model_type || 'Custom',
              run_id: row.run_id || `run_${{i}}`,
              run_name: row.run_name || `run_${{i}}`,
              model: m,
              sampler: row.sampler_pkg || '',
              lookback: lb,
              resample: rf,
              modality: row.modalities_choice || '',
              n_mods: row.num_modalities ? parseInt(row.num_modalities) : 1,
              featurizer: fNorm,
              m_feat: mFeat,
              auroc: aurocVal,
              bal_acc: balAccVal,
              f1: f1Val,
              prec: precVal,
              rec: recVal,
              spec: specVal,
              sens90: sens90Val,
              wp_auroc: row.within_person_auroc_mean && !isNaN(row.within_person_auroc_mean) ? parseFloat(row.within_person_auroc_mean) : null,
              wp_auroc_med: null,
              fold_auroc: null,
              fold_f1: null,
              fold_spec: null,
            }});
          }}

          if (newRecords.length > 0) {{
            RAW_DATA = newRecords;
            const badge = document.getElementById('data_source_badge');
            badge.innerText = `Uploaded: ${{file.name}} (${{newRecords.length}} runs)`;
            badge.className = "px-3 py-2 text-xs font-semibold text-emerald-300 bg-emerald-950/60 rounded-lg border border-emerald-500/40";
            updateDashboard();
          }}
        }} catch (err) {{
          alert('Error parsing CSV file: ' + err.message);
        }}
      }};
      reader.readAsText(file);
    }}

    function exportFilteredCSV() {{
      if (filteredRuns.length === 0) return alert('No runs match filter.');
      const headers = Object.keys(filteredRuns[0]);
      let csv = headers.join(',') + '\\n';
      filteredRuns.forEach(r => {{
        csv += headers.map(h => JSON.stringify(r[h] ?? '')).join(',') + '\\n';
      }});

      const blob = new Blob([csv], {{ type: 'text/csv' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `wearables_modality_analysis_${{new Date().toISOString().slice(0,10)}}.csv`;
      a.click();
    }}

    function sortActiveMetric() {{
      const metricKey = document.getElementById('metric_select').value;
      sortTable(metricKey);
    }}

    function updateDashboard() {{
      filterData();
      const metricKey = document.getElementById('metric_select').value;
      const metricLabel = document.getElementById('metric_select').options[document.getElementById('metric_select').selectedIndex].text;
      document.getElementById('ranking_metric_label').innerText = metricLabel;

      const activeTh = document.getElementById('table_active_metric_th');
      if (activeTh) activeTh.innerHTML = `${{metricLabel}} &uarr;&darr;`;

      const stats = computeStatsByModality(metricKey);
      updateKPIs(stats, metricKey);
      renderRankingChart(stats, metricLabel);
      renderBoxplots(stats, metricKey, metricLabel);
      renderModelModalityBars(stats, metricKey, metricLabel);
      renderSignificanceTable(stats, metricKey);
      populateHeatmapDropdown(stats);
      renderTemporalHeatmap();
      updateTable();
    }}

    // Initial boot
    document.addEventListener('DOMContentLoaded', () => {{
      updateDashboard();
    }});
  </script>
</body>
</html>
"""

    out_html_path = Path("reports/sweep_analysis/rq1_interactive_modality_dashboard.html")
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated standalone interactive HTML dashboard at: {out_html_path}")

    # Also generate the companion Jupyter notebook
    generate_companion_notebook()


def generate_companion_notebook():
    nb_path = Path("notebooks/RQ1_combined_sweep_analysis.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RQ1 & RQ2: Sensor Modality Hierarchy & Model Architecture Synergy\n",
                    "\n",
                    "This notebook evaluates the predictive utility of passive wearable sensor streams across:\n",
                    "- **7 Modalities**: `step`, `calorie`, `distance`, `step_calorie`, `step_distance`, `calorie_distance`, `step_calorie_distance`\n",
                    "- **6 Model Families**: `flaml_xgboost`, `flaml_rf`, `flaml_lr`, `flaml_svm`, `transformer`, `lstm`\n",
                    "- **20 Rolling Samplers**: 5 Lookbacks (24h, 48h, 72h, 96h, 120h) x 4 Frequencies (4h, 6h, 8h, 12h)\n",
                    "- **1,000 Cyclical Walk-Forward Evaluations** from W&B sweeps `9dvq8uiy`, `a6sl4omd`, and `zoznzd8y`.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from scipy import stats\n",
                    "\n",
                    "# Set aesthetic plotting theme\n",
                    "sns.set_theme(style='darkgrid')\n",
                    "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n",
                    "\n",
                    "# Load master results\n",
                    "df = pd.read_csv('../reports/sweep_results/all_sweeps_results.csv')\n",
                    "print(f'Total runs loaded: {len(df)}')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Modality Performance Hierarchy (Which Modality is Best?)\n",
                    "Computing the grand mean and standard deviation of pooled AUROC across all valid runs."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def analyze_modalities(models=None, metric='pooled_auroc'):\n",
                    "    sub = df.copy()\n",
                    "    if models:\n",
                    "        sub = sub[sub['model_pkg'].isin(models)]\n",
                    "    \n",
                    "    agg = sub.groupby('modalities_choice')[metric].agg(\n",
                    "        mean='mean',\n",
                    "        std='std',\n",
                    "        median='median',\n",
                    "        min='min',\n",
                    "        max='max',\n",
                    "        count='count'\n",
                    "    ).reset_index().sort_values(by='mean', ascending=False)\n",
                    "    \n",
                    "    return agg\n",
                    "\n",
                    "# Overall Modality Ranking across all models\n",
                    "modality_ranking = analyze_modalities()\n",
                    "modality_ranking"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot Modality Ranking\n",
                    "plt.figure(figsize=(10, 5), dpi=150)\n",
                    "palette = ['#3b82f6' if m in ['step', 'calorie', 'distance'] else '#8b5cf6' if m != 'step_calorie_distance' else '#10b981' for m in modality_ranking['modalities_choice']]\n",
                    "sns.barplot(data=modality_ranking, x='modalities_choice', y='mean', palette=palette)\n",
                    "plt.errorbar(x=range(len(modality_ranking)), y=modality_ranking['mean'], yerr=modality_ranking['std'], fmt='none', c='black', capsize=4)\n",
                    "plt.title('RQ1: Wearable Modality Performance Hierarchy (Pooled AUROC)', fontsize=13, fontweight='bold')\n",
                    "plt.xlabel('Modality Combination', fontsize=11)\n",
                    "plt.ylabel('Mean Pooled AUROC', fontsize=11)\n",
                    "plt.xticks(rotation=25)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Model Architecture Breakdown (e.g. XGBoost Only vs RF vs Deep Learning)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Compare XGBoost only\n",
                    "xgb_ranking = analyze_modalities(models=['flaml_xgboost'])\n",
                    "print('--- XGBoost Ranking ---')\n",
                    "display(xgb_ranking)\n",
                    "\n",
                    "# Compare Deep Learning (Transformer + LSTM)\n",
                    "dl_ranking = analyze_modalities(models=['transformer', 'lstm'])\n",
                    "print('--- Deep Learning (Transformer + LSTM) Ranking ---')\n",
                    "display(dl_ranking)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Heatmap: Model x Modality\n",
                    "pivot = df.pivot_table(index='model_pkg', columns='modalities_choice', values='pooled_auroc', aggfunc='mean')\n",
                    "plt.figure(figsize=(10, 4), dpi=150)\n",
                    "sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu')\n",
                    "plt.title('Model Architecture vs Sensor Modality (Mean Pooled AUROC)', fontweight='bold')\n",
                    "plt.ylabel('Model Family')\n",
                    "plt.xlabel('Modality Choice')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Head-to-Head Paired Statistical Significance\n",
                    "Testing whether multimodal combinations statistically outperform unimodal baselines on identical temporal window splits."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "top_mod = modality_ranking.iloc[0]['modalities_choice']\n",
                    "print(f'Testing all modalities head-to-head against winner: {top_mod}')\n",
                    "\n",
                    "paired_piv = df.pivot_table(index=['model_pkg', 'sampler_pkg'], columns='modalities_choice', values='pooled_auroc')\n",
                    "\n",
                    "sig_results = []\n",
                    "for col in paired_piv.columns:\n",
                    "    if col == top_mod:\n",
                    "        continue\n",
                    "    valid = paired_piv[[top_mod, col]].dropna()\n",
                    "    diff = valid[top_mod] - valid[col]\n",
                    "    t_stat, p_val = stats.ttest_rel(valid[top_mod], valid[col])\n",
                    "    try:\n",
                    "        w_stat, w_pval = stats.wilcoxon(diff)\n",
                    "    except Exception:\n",
                    "        w_pval = np.nan\n",
                    "        \n",
                    "    sig_results.append({\n",
                    "        'Modality': col,\n",
                    "        'N Pairs': len(valid),\n",
                    "        'Mean Difference (Top - Other)': diff.mean(),\n",
                    "        't-statistic': t_stat,\n",
                    "        't-test p-value': p_val,\n",
                    "        'Wilcoxon p-value': w_pval,\n",
                    "        'Significant (p<0.05)': 'Yes (***)' if p_val < 0.001 else ('Yes (*)' if p_val < 0.05 else 'No (ns)')\n",
                    "    })\n",
                    "\n",
                    "sig_df = pd.DataFrame(sig_results).sort_values(by='Mean Difference (Top - Other)')\n",
                    "sig_df"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated companion Jupyter Notebook at: {nb_path}")


if __name__ == "__main__":
    generate_interactive_dashboard()
