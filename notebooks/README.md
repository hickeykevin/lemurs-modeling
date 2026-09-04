# Jupyter Notebooks & Sweep Analysis (`notebooks/`)

This directory contains Jupyter notebooks, standalone statistical research question analysis scripts, and programmatic notebook generation tools organized by research area and purpose.

---

## 📁 Directory Structure

```text
notebooks/
├── sweep_analysis/      # Multi-factor hyperparameter sweep evaluation, standalone RQ scripts, & statistical tests
├── data_exploration/    # Exploratory data analysis on EMA surveys & raw sensor streams
└── guides/              # Starter code & modeling strategy notes
```

---

## 📊 1. Sweep Analysis (`notebooks/sweep_analysis/`)

### Standalone Hypothesis Testing Scripts
- **[rq1_analysis.py](sweep_analysis/rq1_analysis.py)** (*Modality Value & Multimodal Synergy*):
  - Non-parametric Friedman & Repeated Measures ANOVA across matched conditions.
  - Pairwise Wilcoxon signed-rank tests with Benjamini-Hochberg FDR / Holm corrections.
  - Cohen's $d_z$ effect sizes, JZS Bayes Factors ($\text{BF}_{10}$), and marginal synergy vs. redundancy analysis.
- **[rq2_analysis.py](sweep_analysis/rq2_analysis.py)** (*Model Architecture Comparison*):
  - Omnibus Friedman tests & Kendall's $W$ comparing Deep Sequence Models (LSTM, Transformer) against Tabular AutoML (FLAML XGBoost).
  - Head-to-head pairwise contrasts and JZS Bayes factors.
  - Horizon moderation / interaction tests ($24\text{h} \to 120\text{h}$).
- **[rq3_analysis.py](sweep_analysis/rq3_analysis.py)** (*Temporal Dynamics & Lookback Windows*):
  - 2-Way Factorial ANOVA (Lookback Horizon $\times$ Sampling Granularity) with partial $\eta^2$ effect sizes.
  - Lookback horizon decay trends ($24\text{h} \to 120\text{h}$) and polynomial fits.
  - Granularity pairwise contrasts ($4\text{h}$, $6\text{h}$, $8\text{h}$, $12\text{h}$) and optimal hyperparameter configuration rankings.
- **[run_statistical_analysis.py](sweep_analysis/run_statistical_analysis.py)**:
  - Master pipeline script that runs the comprehensive analysis and exports all statistical figures and tables to `reports/sweep_analysis/`.

### Interactive Notebooks & Generators
- **[generate_sweep_statistical_analysis_notebook.py](sweep_analysis/generate_sweep_statistical_analysis_notebook.py)**:
  - Active generator that programmatically builds `sweep_statistical_analysis.ipynb`.
- **[sweep_statistical_analysis.ipynb](sweep_analysis/sweep_statistical_analysis.ipynb)**:
  - The compiled, interactive analysis notebook with ANOVA tables, boxplots, heatmaps, and post-hoc tests.
- **[generate_analysis_notebook.py](sweep_analysis/generate_analysis_notebook.py)** & **[analyze_sweep.py](sweep_analysis/analyze_sweep.py)**:
  - Historical visualization and metric aggregation tools from preliminary sweeps.
- **[sweep_visualization.ipynb](sweep_analysis/sweep_visualization.ipynb)**:
  - Historical visualization notebook.

---

## 🔍 2. Data Exploration (`notebooks/data_exploration/`)

- **[analyze_suicide_risk.py](data_exploration/analyze_suicide_risk.py)**:
  - Analysis of EMA suicidal risk question distributions, class imbalances, and aggregation thresholds.
- **[explore_multiple_daily_surveys.py](data_exploration/explore_multiple_daily_surveys.py)**:
  - Investigation of multiple intraday survey responses, submission timestamps, and deduplication behavior.
- **[2)DataCleaningCalorie.ipynb](data_exploration/2%29DataCleaningCalorie.ipynb)** & **[2)DataCleaningDistance.ipynb](data_exploration/2%29DataCleaningDistance.ipynb)**:
  - Exploratory cleaning of raw Apple Health / FitBit calorie and distance passive sensing streams.

---

## 📚 3. Guides & Tutorials (`notebooks/guides/`)

- **[starter_code.ipynb](guides/starter_code.ipynb)**:
  - Onboarding notebook demonstrating basic dataset loading, model initialization, and training loops.
- **[modeling_strategies.md](guides/modeling_strategies.md)**:
  - Reference notes on survey aggregation, sequence alignment, and evaluation schemes.
