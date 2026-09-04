import json
import os

def build_complete_notebook_all_rqs():
    cells = []
    
    # ----------------------------------------------------
    # Cell 1: Title
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Quantitative Statistical Evaluation: Passive Sensing for EMA-Assessed Suicidal Risk\n",
            "### Hyperparameter Sweep `r2mhb7wj` Analysis (Rolling Sampler Evaluation across 5-Fold Walk-Forward Cross Validation)\n",
            "\n",
            "This notebook provides the complete statistical analysis for publication answering:\n",
            "- **RQ1 (Modality Value & Multimodal Synergy)**: *Are passive sensing data predictive of EMA suicidal risk, and which modality or combination provides greatest value?*\n",
            "- **RQ2 (Model Architecture)**: *Do deep sequence models (LSTM, Transformer) better predict suicidal risk than classical ML (FLAML XGBoost)?*\n",
            "- **RQ3 (Temporal Dynamics)**: *How do temporal design choices (lookback horizon and sampling granularity under consistent 120h boundary purging) affect predictive performance?*\n",
            "\n",
            "---"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 2: Imports & Styling
    # ----------------------------------------------------
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import netrc\n",
            "import requests\n",
            "from pathlib import Path\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import scipy.stats as stats\n",
            "from scipy.integrate import quad\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Visual styling for publication\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams.update({\n",
            "    'font.family': 'sans-serif',\n",
            "    'font.size': 11,\n",
            "    'axes.labelsize': 12,\n",
            "    'axes.titlesize': 13,\n",
            "    'xtick.labelsize': 10,\n",
            "    'ytick.labelsize': 10,\n",
            "    'legend.fontsize': 10,\n",
            "    'figure.titlesize': 14,\n",
            "    'figure.dpi': 150,\n",
            "    'pdf.fonttype': 42,\n",
            "    'ps.fonttype': 42,\n",
            "})\n",
            "\n",
            "def find_output_dir():\n",
            "    p = Path.cwd().resolve()\n",
            "    for _ in range(5):\n",
            "        if (p / 'reports' / 'sweep_analysis' / 'sweep_r2mhb7wj_full_records.csv').exists():\n",
            "            return p / 'reports' / 'sweep_analysis'\n",
            "        if (p / 'src').exists() or (p / '.project-root').exists():\n",
            "            return p / 'reports' / 'sweep_analysis'\n",
            "        p = p.parent\n",
            "    return Path('reports/sweep_analysis').resolve()\n",
            "\n",
            "OUTPUT_DIR = find_output_dir()\n",
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)\n",
            "(OUTPUT_DIR / 'tables').mkdir(exist_ok=True)\n",
            "\n",
            "print('Setup complete. Output artifacts directory:', OUTPUT_DIR)"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 3: Data Ingestion
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Ingestion & Factor Decomposition\n",
            "We load all runs from sweep `r2mhb7wj` and filter exclusively to the **Rolling** sampler configurations ($N = 420$ runs: 20 rolling samplers $\\times$ 3 model architectures $\\times$ 7 modality subsets with consistent 120h boundary purging)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "CACHE_FILE = OUTPUT_DIR / 'sweep_r2mhb7wj_full_records.csv'\n",
            "df_raw = pd.read_csv(CACHE_FILE)\n",
            "\n",
            "# Factor parsing\n",
            "def parse_sampler(s):\n",
            "    parts = s.split('_')\n",
            "    strategy = parts[0].capitalize()\n",
            "    lookback_h = int(parts[1].replace('h', ''))\n",
            "    window_h = int(parts[2].replace('h', ''))\n",
            "    return strategy, lookback_h, lookback_h / 24.0, window_h\n",
            "\n",
            "strategy, lb_h, lb_d, win_h = zip(*df_raw['sampler_pkg'].map(parse_sampler))\n",
            "df_raw['sampler_strategy'] = strategy\n",
            "df_raw['lookback_hours'] = lb_h\n",
            "df_raw['lookback_days'] = lb_d\n",
            "df_raw['window_hours'] = win_h\n",
            "df_raw['sequence_length'] = df_raw['lookback_hours'] / df_raw['window_hours']\n",
            "\n",
            "model_map = {'default': 'LSTM', 'transformer': 'Transformer', 'flaml_xgboost': 'FLAML XGBoost'}\n",
            "df_raw['model_name'] = df_raw['model_pkg'].map(model_map)\n",
            "\n",
            "def format_modalities(m_str):\n",
            "    m_list = json.loads(m_str) if isinstance(m_str, str) else m_str\n",
            "    m_set = set(m_list)\n",
            "    if m_set == {'step'}:\n",
            "        return 'Step Only', 1, True, False, False\n",
            "    elif m_set == {'calorie'}:\n",
            "        return 'Calorie Only', 1, False, True, False\n",
            "    elif m_set == {'distance'}:\n",
            "        return 'Distance Only', 1, False, False, True\n",
            "    elif m_set == {'step', 'calorie'}:\n",
            "        return 'Step + Calorie', 2, True, True, False\n",
            "    elif m_set == {'step', 'distance'}:\n",
            "        return 'Step + Distance', 2, True, False, True\n",
            "    elif m_set == {'calorie', 'distance'}:\n",
            "        return 'Calorie + Distance', 2, False, True, True\n",
            "    elif m_set == {'step', 'calorie', 'distance'}:\n",
            "        return 'All 3 (Step+Cal+Dist)', 3, True, True, True\n",
            "    return 'Other', len(m_list), False, False, False\n",
            "\n",
            "mod_labels, mod_counts, has_step, has_cal, has_dist = zip(*df_raw['modalities_raw'].map(format_modalities))\n",
            "df_raw['modality_label'] = mod_labels\n",
            "df_raw['n_modalities'] = mod_counts\n",
            "df_raw['has_step'] = has_step\n",
            "df_raw['has_calorie'] = has_cal\n",
            "df_raw['has_distance'] = has_dist\n",
            "\n",
            "# Filter to Rolling sampler only\n",
            "df = df_raw[df_raw['sampler_strategy'] == 'Rolling'].copy().reset_index(drop=True)\n",
            "\n",
            "# JZS Bayes Factor helper\n",
            "def jzs_bayes_factor_paired(diff, r=np.sqrt(2)/2):\n",
            "    diff_clean = diff.dropna()\n",
            "    n = len(diff_clean)\n",
            "    if n < 2:\n",
            "        return 0.0, np.nan\n",
            "    t_stat, _ = stats.ttest_1samp(diff_clean, 0)\n",
            "    df_val = n - 1\n",
            "    def integrand(g):\n",
            "        term1 = (1.0 + n * g * (r**2))**(-0.5)\n",
            "        term2 = (1.0 + (t_stat**2) / (df_val * (1.0 + n * g * (r**2))))**(-n / 2.0)\n",
            "        prior = (2.0 * np.pi)**(-0.5) * (g**(-1.5)) * np.exp(-1.0 / (2.0 * g))\n",
            "        return term1 * term2 * prior\n",
            "    num, _ = quad(integrand, 0, np.inf)\n",
            "    denom = (1.0 + (t_stat**2) / df_val)**(-n / 2.0)\n",
            "    bf10 = num / denom if denom > 0 else np.inf\n",
            "    return t_stat, bf10\n",
            "\n",
            "def interpret_bf(bf10):\n",
            "    if np.isnan(bf10): return 'N/A'\n",
            "    if bf10 > 100: return 'Decisive Evidence (H1)'\n",
            "    elif bf10 > 30: return 'Very Strong Evidence (H1)'\n",
            "    elif bf10 > 10: return 'Strong Evidence (H1)'\n",
            "    elif bf10 > 3: return 'Moderate Evidence (H1)'\n",
            "    elif bf10 >= 1/3: return 'Anecdotal / Inconclusive'\n",
            "    elif bf10 >= 1/10: return 'Moderate Evidence (H0: Eq)'\n",
            "    elif bf10 >= 1/30: return 'Strong Evidence (H0: Eq)'\n",
            "    else: return 'Decisive Evidence (H0: Eq)'\n",
            "\n",
            "print(f'Tidy Dataset: {len(df)} runs ready across 420 fully crossed rolling conditions.')\n",
            "df.head()"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 4: RQ1 Analysis
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. RQ1: Modality Value & Multimodal Synergy (60 Matched Blocks)\n",
            "- **Step 1 (Omnibus)**: Non-parametric Friedman test across 60 matched blocks (20 rolling samplers $\\times$ 3 models).\n",
            "- **Step 2 (Pairwise Contrasts)**: Paired Wilcoxon + Benjamini-Hochberg FDR + JZS Bayes Factors ($\\text{BF}_{10}$).\n",
            "- **Step 3 (Marginal Synergy)**: Stepwise gain $\\Delta_{\\text{Pair}} - \\max(\\text{Single})$ and $\\Delta_{\\text{Trio}} - \\max(\\text{Pairs})$."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "piv_mod = df.pivot(index=['sampler_pkg', 'model_pkg'], columns='modality_label', values='pooled_auroc')\n",
            "mod_order = ['Step Only', 'Calorie Only', 'Distance Only', 'Step + Calorie', 'Step + Distance', 'Calorie + Distance', 'All 3 (Step+Cal+Dist)']\n",
            "piv_mod = piv_mod[mod_order]\n",
            "\n",
            "# Step 1: Friedman Omnibus Test\n",
            "f_stat_rq1, f_pval_rq1 = stats.friedmanchisquare(*[piv_mod[c] for c in mod_order])\n",
            "print(f'=== RQ1 Step 1: Friedman Omnibus Test: Chi2(6) = {f_stat_rq1:.4f}, p = {f_pval_rq1:.4e} (N = {len(piv_mod)} blocks) ===')\n",
            "\n",
            "# Step 2: Post-Hoc Pairwise Table\n",
            "pw_rq1 = []\n",
            "for i in range(len(mod_order)):\n",
            "    for j in range(i + 1, len(mod_order)):\n",
            "        m1, m2 = mod_order[i], mod_order[j]\n",
            "        diff = piv_mod[m1] - piv_mod[m2]\n",
            "        ci_l, ci_u = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))\n",
            "        w_stat, w_p = stats.wilcoxon(diff)\n",
            "        t_stat, bf10 = jzs_bayes_factor_paired(diff)\n",
            "        pw_rq1.append({\n",
            "            'Comparison': f'{m1} vs. {m2}',\n",
            "            'N_Pairs': len(diff),\n",
            "            'Mean_Delta': diff.mean(),\n",
            "            '95% CI': f'[{ci_l:+.4f}, {ci_u:+.4f}]',\n",
            "            'Cohen_dz': diff.mean() / diff.std(),\n",
            "            'Wilcoxon_p': w_p,\n",
            "            'BF10': bf10,\n",
            "            'BF01': 1.0 / bf10 if bf10 > 0 else np.nan,\n",
            "            'Interpretation': interpret_bf(bf10)\n",
            "        })\n",
            "df_pw_rq1 = pd.DataFrame(pw_rq1)\n",
            "sorted_idx = df_pw_rq1['Wilcoxon_p'].argsort()\n",
            "p_sorted = df_pw_rq1['Wilcoxon_p'].iloc[sorted_idx].values\n",
            "m_k = len(df_pw_rq1)\n",
            "q_vals = np.zeros(m_k)\n",
            "q_vals[-1] = p_sorted[-1]\n",
            "for k in range(m_k - 2, -1, -1):\n",
            "    q_vals[k] = min(q_vals[k+1], p_sorted[k] * m_k / (k + 1))\n",
            "df_pw_rq1.loc[df_pw_rq1.index[sorted_idx], 'BH_FDR_q'] = np.minimum(q_vals, 1.0)\n",
            "df_pw_rq1.to_csv(OUTPUT_DIR / 'tables' / 'rq1_pairwise_contrasts.csv', index=False)\n",
            "\n",
            "# Step 3: Synergy Table\n",
            "synergy_list = []\n",
            "for pair, (s1, s2) in {'Step + Distance': ('Step Only', 'Distance Only'), 'Step + Calorie': ('Step Only', 'Calorie Only'), 'Calorie + Distance': ('Calorie Only', 'Distance Only')}.items():\n",
            "    delta = piv_mod[pair] - np.maximum(piv_mod[s1], piv_mod[s2])\n",
            "    ci_l, ci_u = stats.t.interval(0.95, len(delta)-1, loc=delta.mean(), scale=stats.sem(delta))\n",
            "    _, w_p = stats.wilcoxon(delta)\n",
            "    _, bf10 = jzs_bayes_factor_paired(delta)\n",
            "    synergy_list.append({\n",
            "        'Transition': f'Max({s1}, {s2}) -> {pair}',\n",
            "        'Marginal_Delta': delta.mean(),\n",
            "        '95% CI': f'[{ci_l:+.4f}, {ci_u:+.4f}]',\n",
            "        'Cohen_dz': delta.mean() / delta.std(),\n",
            "        'Wilcoxon_p': w_p,\n",
            "        'BF10': bf10,\n",
            "        'Verdict': 'Synergistic (Gain > 0)' if (delta.mean() > 0 and w_p < 0.05) else 'Redundant / Diminishing'\n",
            "    })\n",
            "trio_delta = piv_mod['All 3 (Step+Cal+Dist)'] - np.maximum(np.maximum(piv_mod['Step + Distance'], piv_mod['Step + Calorie']), piv_mod['Calorie + Distance'])\n",
            "ci_l, ci_u = stats.t.interval(0.95, len(trio_delta)-1, loc=trio_delta.mean(), scale=stats.sem(trio_delta))\n",
            "_, w_p = stats.wilcoxon(trio_delta)\n",
            "_, bf10 = jzs_bayes_factor_paired(trio_delta)\n",
            "synergy_list.append({\n",
            "    'Transition': 'Max(All Pairs) -> All 3 Modalities',\n",
            "    'Marginal_Delta': trio_delta.mean(),\n",
            "    '95% CI': f'[{ci_l:+.4f}, {ci_u:+.4f}]',\n",
            "    'Cohen_dz': trio_delta.mean() / trio_delta.std(),\n",
            "    'Wilcoxon_p': w_p,\n",
            "    'BF10': bf10,\n",
            "    'Verdict': 'Synergistic (Gain > 0)' if (trio_delta.mean() > 0 and w_p < 0.05) else 'Negative Synergy / Noise Penalty'\n",
            "})\n",
            "df_syn_rq1 = pd.DataFrame(synergy_list)\n",
            "df_syn_rq1.to_csv(OUTPUT_DIR / 'tables' / 'rq1_synergy_analysis.csv', index=False)\n",
            "\n",
            "print('\\n=== RQ1 Step 3: Marginal Synergy vs. Redundancy ===')\n",
            "display(df_syn_rq1)\n",
            "df_pw_rq1[['Comparison', 'Mean_Delta', '95% CI', 'Cohen_dz', 'Wilcoxon_p', 'BH_FDR_q', 'BF10', 'Interpretation']].head(8)"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 5: RQ2 Analysis
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. RQ2: Deep Time-Series Models vs. Tree-Based ML (140 Matched Conditions)\n",
            "- **Step 1 (Omnibus Architecture Test)**: Friedman test across the 140 matched conditions (20 rolling samplers $\\times$ 7 modalities).\n",
            "- **Step 2 (Head-to-Head Contrasts)**: Paired Wilcoxon, Cohen's $d_z$, and JZS Bayes Factors ($\\text{BF}_{10}$ for superiority; $\\text{BF}_{01}$ for equivalence).\n",
            "- **Step 3 (Horizon Moderation)**: Testing if the time-series advantage over XGBoost depends on lookback duration ($24\\text{h} \\to 120\\text{h}$)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "piv_m = df.pivot(index=['sampler_pkg', 'modalities_raw'], columns='model_name', values='pooled_auroc')[['LSTM', 'Transformer', 'FLAML XGBoost']]\n",
            "\n",
            "# Step 1: Friedman Omnibus Architecture Test\n",
            "f_stat_rq2, f_pval_rq2 = stats.friedmanchisquare(piv_m['LSTM'], piv_m['Transformer'], piv_m['FLAML XGBoost'])\n",
            "kendall_w_rq2 = f_stat_rq2 / (len(piv_m) * 2)\n",
            "print(f'=== RQ2 Step 1: Friedman Omnibus Test: Chi2(2) = {f_stat_rq2:.4f}, p = {f_pval_rq2:.4e} (Kendall W = {kendall_w_rq2:.4f}, N = {len(piv_m)}) ===')\n",
            "\n",
            "# Step 2: Pairwise Head-to-Head Contrasts\n",
            "rq2_pairs = [\n",
            "    ('LSTM', 'FLAML XGBoost', 'LSTM vs. FLAML XGBoost'),\n",
            "    ('Transformer', 'FLAML XGBoost', 'Transformer vs. FLAML XGBoost'),\n",
            "    ('Transformer', 'LSTM', 'Transformer vs. LSTM')\n",
            "]\n",
            "\n",
            "rq2_pw_res = []\n",
            "for m1, m2, label in rq2_pairs:\n",
            "    diff = piv_m[m1] - piv_m[m2]\n",
            "    mean_d = diff.mean()\n",
            "    std_d = diff.std()\n",
            "    ci_l, ci_u = stats.t.interval(0.95, len(diff)-1, loc=mean_d, scale=stats.sem(diff))\n",
            "    w_stat, w_p = stats.wilcoxon(diff)\n",
            "    t_stat, bf10 = jzs_bayes_factor_paired(diff)\n",
            "    rq2_pw_res.append({\n",
            "        'Comparison': label,\n",
            "        'N_Pairs': len(diff),\n",
            "        'Mean_Delta': mean_d,\n",
            "        '95% CI': f'[{ci_l:+.4f}, {ci_u:+.4f}]',\n",
            "        'Win_Rate (%)': (diff > 0).mean() * 100,\n",
            "        'Cohen_dz': mean_d / std_d,\n",
            "        'Wilcoxon_p': w_p,\n",
            "        'BF10': bf10,\n",
            "        'BF01': 1.0 / bf10 if bf10 > 0 else np.nan,\n",
            "        'Interpretation': interpret_bf(bf10)\n",
            "    })\n",
            "\n",
            "df_rq2_pw = pd.DataFrame(rq2_pw_res)\n",
            "df_rq2_pw.to_csv(OUTPUT_DIR / 'tables' / 'rq2_pairwise_contrasts.csv', index=False)\n",
            "\n",
            "# Step 3: Horizon Moderation Analysis\n",
            "lb_map = df[['sampler_pkg', 'lookback_days']].drop_duplicates().set_index('sampler_pkg')['lookback_days']\n",
            "piv_m_res = piv_m.reset_index()\n",
            "piv_m_res['lookback_days'] = piv_m_res['sampler_pkg'].map(lb_map)\n",
            "piv_m_res['Seq_Advantage'] = ((piv_m_res['LSTM'] + piv_m_res['Transformer']) / 2.0) - piv_m_res['FLAML XGBoost']\n",
            "\n",
            "slope, intercept, r_val, p_val_trend, _ = stats.linregress(piv_m_res['lookback_days'], piv_m_res['Seq_Advantage'])\n",
            "print(f'\\n=== RQ2 Step 3: Horizon Moderation Trend: Slope = {slope:+.4f}/day, r = {r_val:.3f}, p = {p_val_trend:.4e} ===')\n",
            "df_rq2_pw"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 6: RQ3 Complete 3-Step Testing
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. RQ3: Temporal Design Choices (Lookback Horizon & Sampling Granularity)\n",
            "- **Step 1 (Factorial ANOVA)**: Decomposing variance between Lookback Horizon ($24\\text{h} \\to 120\\text{h}$) and Sampling Granularity ($4\\text{h} \\to 12\\text{h}$) with Partial $\\eta^2$.\n",
            "- **Step 2 (Horizon Trend Analysis)**: Testing stability across lookback horizons under consistent 120h boundary purging.\n",
            "- **Step 3 (Granularity Pairwise Contrasts)**: Paired comparisons ($4\\text{h}$ vs $6\\text{h}$ vs $8\\text{h}$ vs $12\\text{h}$) with Bayes Factors across 105 matched conditions."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Factorial ANOVA & Partial Eta-Squared\n",
            "grand_mean = df['pooled_auroc'].mean()\n",
            "factors = {\n",
            "    'Sampling Granularity (Hours)': 'window_hours',\n",
            "    'Lookback Horizon (Days)': 'lookback_days'\n",
            "}\n",
            "anova_rows = []\n",
            "for label, col in factors.items():\n",
            "    groups = [group['pooled_auroc'].values for _, group in df.groupby(col)]\n",
            "    f_val, p_val = stats.f_oneway(*groups)\n",
            "    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)\n",
            "    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)\n",
            "    anova_rows.append({\n",
            "        'Factor': label,\n",
            "        'df': len(groups) - 1,\n",
            "        'Sum_of_Squares': ss_between,\n",
            "        'F_statistic': f_val,\n",
            "        'p_value': p_val,\n",
            "        'Partial_Eta_Squared (eta_p2)': ss_between / (ss_between + ss_within)\n",
            "    })\n",
            "df_anova_rq3 = pd.DataFrame(anova_rows)\n",
            "df_anova_rq3.to_csv(OUTPUT_DIR / 'tables' / 'rq3_factorial_anova.csv', index=False)\n",
            "print('=== RQ3 Step 1: Multi-Factor Factorial ANOVA ===')\n",
            "display(df_anova_rq3)\n",
            "\n",
            "# Step 2: Lookback Horizon Trend Analysis\n",
            "x_vals = df['lookback_days'].values\n",
            "y_vals = df['pooled_auroc'].values\n",
            "slope_lb, _, r_lb, p_lb, _ = stats.linregress(x_vals, y_vals)\n",
            "poly_lb = np.polyfit(x_vals, y_vals, 2)\n",
            "print(f'\\n=== RQ3 Step 2: Horizon Trend: Slope = {slope_lb:+.4f} AUROC/day (r = {r_lb:.3f}, p = {p_lb:.4e}) ===')\n",
            "\n",
            "# Step 3: Granularity Pairwise Contrasts (N = 105 matched conditions)\n",
            "piv_win = df.pivot(index=['model_pkg', 'modalities_raw', 'lookback_hours'], columns='window_hours', values='pooled_auroc')\n",
            "win_pairs = []\n",
            "for i in range(len([4, 6, 8, 12])):\n",
            "    for j in range(i+1, len([4, 6, 8, 12])):\n",
            "        w1, w2 = [4, 6, 8, 12][i], [4, 6, 8, 12][j]\n",
            "        d_w = piv_win[w1] - piv_win[w2]\n",
            "        ci_l_w, ci_u_w = stats.t.interval(0.95, len(d_w)-1, loc=d_w.mean(), scale=stats.sem(d_w))\n",
            "        _, w_p_w = stats.wilcoxon(d_w)\n",
            "        _, bf10_w = jzs_bayes_factor_paired(d_w)\n",
            "        win_pairs.append({\n",
            "            'Comparison': f'{w1}h vs. {w2}h',\n",
            "            'N_Pairs': len(d_w),\n",
            "            'Mean_Delta': d_w.mean(),\n",
            "            '95% CI': f'[{ci_l_w:+.4f}, {ci_u:+.4f}]',\n",
            "            'Win_Rate (%)': (d_w > 0).mean() * 100,\n",
            "            'Cohen_dz': d_w.mean() / d_w.std(),\n",
            "            'Wilcoxon_p': w_p_w,\n",
            "            'BF10': bf10_w,\n",
            "            'BF01': 1.0 / bf10_w if bf10_w > 0 else np.nan,\n",
            "            'Interpretation': interpret_bf(bf10_w)\n",
            "        })\n",
            "df_win_pw = pd.DataFrame(win_pairs)\n",
            "df_win_pw.to_csv(OUTPUT_DIR / 'tables' / 'rq3_granularity_pairwise.csv', index=False)\n",
            "df_win_pw"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 7: Publication Visualizations (Figures 1, 2, 3)
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Publication Visualizations (Figures 1, 2, & 3)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 1: Modality Spectrum (RQ1)\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)\n",
            "sns.boxplot(data=df, x='modality_label', y='pooled_auroc', order=mod_order, hue='modality_label', legend=False, palette='Blues_d', ax=axes[0], fliersize=2)\n",
            "axes[0].set_xticks(range(len(mod_order)))\n",
            "axes[0].set_xticklabels(mod_order, rotation=35, ha='right')\n",
            "axes[0].set_title('(A) Predictive AUROC across Modality Subsets')\n",
            "axes[0].set_ylabel('Pooled AUROC')\n",
            "axes[0].set_xlabel('')\n",
            "axes[0].axhline(0.50, color='red', linestyle='--', alpha=0.7, label='Chance Baseline')\n",
            "axes[0].legend(loc='lower right')\n",
            "\n",
            "sns.violinplot(data=df, x='n_modalities', y='pooled_auroc', hue='model_name', palette='Set2', inner='quartile', ax=axes[1])\n",
            "axes[1].set_xticks([0, 1, 2])\n",
            "axes[1].set_xticklabels(['1 Modality\\n(Single)', '2 Modalities\\n(Pair)', '3 Modalities\\n(Trivariate)'])\n",
            "axes[1].set_title('(B) Modality Expansion by Model Architecture')\n",
            "axes[1].set_xlabel('Feature Set Complexity')\n",
            "axes[1].set_ylabel('')\n",
            "axes[1].legend(title='Model Architecture', loc='lower right')\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'figures' / 'fig1_rq1_modality_value.pdf', bbox_inches='tight')\n",
            "plt.savefig(OUTPUT_DIR / 'figures' / 'fig1_rq1_modality_value.png', dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "# Figure 2: Model Architecture Comparison (Single Panel across Lookback Windows) (RQ2)\n",
            "fig, ax = plt.subplots(figsize=(6.0, 4.2))\n",
            "sns.lineplot(data=df, x='lookback_hours', y='pooled_auroc', hue='model_name', style='model_name', markers=True, dashes=False, palette={'Transformer': '#e41a1c', 'LSTM': '#2ca02c', 'FLAML XGBoost': '#1f77b4'}, errorbar=('ci', 95), ax=ax)\n",
            "ax.set_title('AUROC across Lookback Windows', fontsize=12, weight='bold')\n",
            "ax.set_xlabel('Historical Lookback Window', fontsize=10)\n",
            "ax.set_ylabel('Pooled AUROC (Mean +/- 95% CI)', fontsize=10)\n",
            "ax.set_xticks([24, 48, 72, 96, 120])\n",
            "ax.set_xticklabels(['24h', '48h', '72h', '96h', '120h'])\n",
            "ax.grid(axis='y', linestyle='--', alpha=0.3)\n",
            "ax.legend(title='Model Architecture', loc='lower right', frameon=True)\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'figures' / 'fig2_rq2_model_horizon_interaction.pdf', bbox_inches='tight')\n",
            "plt.savefig(OUTPUT_DIR / 'figures' / 'fig2_rq2_model_horizon_interaction.png', dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "# Figure 3: 2D Response Surface (RQ3 - Rolling Sampler with Uncertainty)\n",
            "fig, ax = plt.subplots(figsize=(6.2, 5.0))\n",
            "means = df.pivot_table(index='lookback_hours', columns='window_hours', values='pooled_auroc', aggfunc='mean')\n",
            "stds = df.pivot_table(index='lookback_hours', columns='window_hours', values='pooled_auroc', aggfunc='std')\n",
            "annot_matrix = means.copy().astype(object)\n",
            "for r in means.index:\n",
            "    for c in means.columns:\n",
            "        m = means.loc[r, c]\n",
            "        s = stds.loc[r, c]\n",
            "        annot_matrix.loc[r, c] = f'{m:.3f}\\n±{s:.3f}'\n",
            "sns.heatmap(means, annot=annot_matrix, fmt='', cmap='YlGnBu', cbar=True, cbar_kws={'label': 'Mean AUROC'}, annot_kws={'size': 9.5, 'weight': 'normal'}, ax=ax, vmin=df['pooled_auroc'].quantile(0.10), vmax=df['pooled_auroc'].quantile(0.95))\n",
            "ax.set_title('RQ3: Temporal Response (Rolling Sampler)', fontsize=11, weight='bold', pad=10)\n",
            "ax.set_xlabel('Sampling Resolution / Window (Hours)', fontsize=10)\n",
            "ax.set_ylabel('Lookback Horizon (Hours)', fontsize=10)\n",
            "ax.set_yticklabels([f'{int(h)}h ({int(h/24)}d)' for h in means.index], rotation=0)\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'figures' / 'fig3_rq3_temporal_response_surface.pdf', bbox_inches='tight')\n",
            "plt.savefig(OUTPUT_DIR / 'figures' / 'fig3_rq3_temporal_response_surface.png', dpi=300, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 8: Top 10 Configurations & LaTeX Export
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Top Configurations & Manuscript LaTeX Export"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "top10 = df.sort_values('pooled_auroc', ascending=False).head(10)[\n",
            "    ['model_name', 'modality_label', 'sampler_strategy', 'lookback_days', 'window_hours', 'pooled_auroc', 'pooled_f1', 'within_person_auroc']\n",
            "].reset_index(drop=True)\n",
            "top10.to_csv(OUTPUT_DIR / 'tables' / 'table5_top10_configurations.csv', index=False)\n",
            "\n",
            "latex_code = top10.to_latex(index=True, float_format='%.4f', caption='Top 10 performing configurations across the 420 rolling sampler runs in sweep r2mhb7wj.', label='tab:top10_sweep_results')\n",
            "with open(OUTPUT_DIR / 'tables' / 'top10_table.tex', 'w') as f:\n",
            "    f.write(latex_code)\n",
            "\n",
            "print('=== Top 10 Parameter Configurations ===')\n",
            "display(top10)\n",
            "print(f'\\nAll tables and figures saved to: {OUTPUT_DIR.resolve()}')"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    from pathlib import Path
    out_path = Path(__file__).parent / "sweep_statistical_analysis.ipynb"
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Successfully built updated: {out_path}")


if __name__ == "__main__":
    build_complete_notebook_all_rqs()
