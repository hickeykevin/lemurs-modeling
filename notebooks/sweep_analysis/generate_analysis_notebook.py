import os
import json

def main():
    notebook_content = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "# Suicide Risk Regression - Sweep Analysis\n",
        "\n",
        "This notebook summarizes the results of hyperparameter sweep `u929626r` for the `suicide_risk_regression` target aggregator. It isolates high-performing data filters and highlights key parameter interaction effects."
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Summary of Sweep Parameters\n",
        "\n",
        "To understand the results, below is a translation of the technical config parameters into natural, translational modeling and clinical terms, along with the specific configurations swept:\n",
        "\n",
        "| Translational Parameter Name | Config Parameter | Modeling & Clinical Control | Swept Values |\n",
        "| :--- | :--- | :--- | :--- |\n",
        "| **Target Clinical Outcome** | `data/aggregator` | The self-report mental health endpoint the regression model is training to predict. | `positive_emotion_regression`, `self_harm_regression`, `suicide_risk_regression` |\n",
        "| **Platform Cohort Filter** | `data.os_filter` | Restricts the patient cohort by phone platform. Used to assess if platform-specific sensor tracking differences (e.g., Apple Health vs Google Fit) affect predictive capability. | `ios` (iPhone only), `both` (iPhone + Android) |\n",
        "| **Daily Survey Aggregation Policy** | `data.collapse_strategy` | Controls how the model handles cases where a subject submits multiple self-report surveys on the same day. | `mean` (average responses to smooth noise), `first` (use the first response of the day), `none` (treat duplicate daily reports as separate samples) |\n",
        "| **Feature Normalization Strategy** | `data/scaler` | Determines how behavioral features (such as step counts) are normalized. | `standard` (global standardization across the cohort), `subject_standard` (individual standardization per subject to account for personal baseline ranges) |\n",
        "| **Time-Series Sampling Method** | `data/sampler` | Determines how historical sequences of behavior are grouped and fed into the LSTM. | `block` (discrete, non-overlapping sequential windows), `offset` (rolling, overlapping sliding windows) |\n",
        "| **Temporal Resolution (Step Size)** | `++data.sampler.resample_freq` | The duration of each individual step in the input sequence. Smaller intervals capture fine-grained patterns, while larger ones yield aggregated trends. | `4h`, `6h`, `12h` |\n",
        "| **Historical Lookback Window** | `++data.sampler.lookback_days` | The total number of days of history analyzed by the LSTM to make a prediction. | `3`, `5`, `7` days |\n",
        "| **Prediction Lag (Start Offset)** | `++data.sampler.start_offset_hours` | How far back the behavioral lookback window begins relative to the survey time, defining the historical start boundary. | `-72h` (3 days back), `-120h` (5 days back), `-168h` (7 days back) |"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Target Regression Labeling Logic\n",
        "\n",
        "To train the regression models, raw multi-question momentary assessment survey answers must be mathematically collapsed into a single continuous score. This is handled by the project's [label_aggregators.py](file:///Users/khickey/Desktop/lemurs/lemurs-modeling/src/data/components/label_aggregators.py) implementation via the `RegressionAggregator` class.\n",
        "\n",
        "### 1. The Core Aggregation Mechanism\n",
        "For any given survey submission, the regression target is computed as follows:\n",
        "* **Target Question Selection**: The aggregator isolates a specific list of question IDs (`likert_ids`) associated with the clinical target.\n",
        "* **Numeric Mapping**: Answers are cleaned and converted to numeric values (e.g., mapping boolean `\"yes\"`/`\"no\"` answers to `1.0`/`0.0` and casting standard strings to floats).\n",
        "* **Likert Scale Zero-Shifting**: To ensure a baseline score of 0 corresponds to no severity, 1-indexed Likert scales (e.g., 1-5 or 1-7) are shifted by subtracting 1. (The code automatically checks if `0` is present in the dataset to avoid double-shifting).\n",
        "* **Target Summation**: The shifted scores are summed across the target questions.\n",
        "* **Missing Data Safety**: The sum requires at least one valid question (`min_count=1`). If all targeted questions for a survey response are missing (NaN), the label is set to NaN rather than defaulting to `0` (which would falsely indicate zero risk).\n",
         "\n",
         "---\n",
         "\n",
         "### 2. Specific Target Aggregators in the Sweep\n",
         "\n",
         "The hyperparameter sweep swept over three clinical target endpoints. The specific summation logic for each target is defined below:\n",
         "\n",
         "| Target Outcome | Swept Aggregator Value | Targeted Question IDs | Mathematical Logic | Clinical Mapping |\n",
         "| :--- | :--- | :--- | :--- | :--- |\n",
         "| **Suicide Risk** | `suicide_risk_regression` | `[2, 3, 5, 7]` | $\\sum_{q \\in \\{2,3,5,7\\}} (\\text{Score}_q - 1)$ | Sums Likert-scale questions regarding active suicidal ideation and self-harm desires, excluding binary/conditional questions. |\n",
         "| **Self-Harm** | `self_harm_regression` | `[9]` | $(\\text{Score}_9 - 1)$ | Sums Likert-scale questions explicitly measuring recent self-harm frequencies or behaviors. |\n",
         "| **Positive Emotion** | `positive_emotion_regression` | `[21, 22, 37]` | $\\sum_{q \\in \\{21,22,37\\}} (\\text{Score}_q - 1)$ | Sums Likert-scale mood descriptors capturing positive affect/emotion to model protective factors. |"
        ]
       },
       {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import os\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import rootutils\n",
        "\n",
        "# Set styling\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
        "plt.rcParams[\"font.size\"] = 12\n",
        "palette = sns.color_palette(\"coolwarm\", as_cmap=False)\n",
        "\n",
        "# Load data\n",
        "root_dir = rootutils.setup_root(os.getcwd(), indicator=\".project-root\", pythonpath=True)\n",
        "csv_path = os.path.join(root_dir, \"notebooks\", \"sweep_u929626r_data.csv\")\n",
        "df = pd.read_csv(csv_path)\n",
        "\n",
        "agg_col = \"config/data/aggregator\"\n",
        "mse_col = \"metric/val/mse\"\n",
        "\n",
        "df_suicide = df[df[agg_col] == \"suicide_risk_regression\"].copy()\n",
        "df_suicide[mse_col] = pd.to_numeric(df_suicide[mse_col], errors='coerce')\n",
        "df_suicide = df_suicide.dropna(subset=[mse_col])\n",
        "print(f\"Loaded {len(df_suicide)} runs for suicide_risk_regression.\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 1. Main Effect: OS Filter\n",
        "\n",
        "Filtering the cohort to only include **iOS** users results in a huge performance boost compared to using **both** iOS and Android data. This is likely due to platform differences in sensor data coverage (e.g., steps). "
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "plt.figure(figsize=(7, 4.5))\n",
        "sns.boxplot(data=df_suicide, x=\"config/data.os_filter\", y=\"metric/val/mse\", palette=\"Set2\")\n",
        "plt.title(\"Validation MSE by OS Filter (Suicide Risk Regression)\", weight='bold', pad=15)\n",
        "plt.xlabel(\"OS Filter\")\n",
        "plt.ylabel(\"Validation MSE\")\n",
        "plt.tight_layout()\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 2. Main Effect: Collapse Strategy (iOS Runs Only)\n",
        "\n",
        "Taking the daily **mean** of survey responses (when a user submits multiple responses in one day) yields significantly lower validation MSE than taking the **first** response or doing **no collapse** at all."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "df_ios = df_suicide[df_suicide[\"config/data.os_filter\"] == \"ios\"]\n",
        "plt.figure(figsize=(7, 4.5))\n",
        "sns.barplot(data=df_ios, x=\"config/data.collapse_strategy\", y=\"metric/val/mse\", estimator=np.mean, errorbar=\"sd\", palette=\"Set3\")\n",
        "plt.title(\"Mean Validation MSE by Collapse Strategy (iOS only)\", weight='bold', pad=15)\n",
        "plt.xlabel(\"Collapse Strategy\")\n",
        "plt.ylabel(\"Mean Validation MSE (+/- SD)\")\n",
        "plt.tight_layout()\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 3. Interaction Effect: Scaler x Sampler (iOS + Mean Collapse runs only)\n",
        "\n",
        "Isolating runs to the best settings (**iOS** and **Mean Collapse**), we look at how the Scaler and Sampler interact.\n",
        "\n",
        "There is a **strong interaction**: `subject_standard` standardisation works exceptionally well when paired with the `block` sampler (mean MSE **`0.5632`**), but is the *worst* configuration when paired with the `offset` sampler (mean MSE **`0.5896`**)."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "df_best = df_ios[df_ios[\"config/data.collapse_strategy\"] == \"mean\"]\n",
        "interaction = df_best.groupby([\"config/data/scaler\", \"config/data/sampler\"])[mse_col].mean().unstack()\n",
        "\n",
        "plt.figure(figsize=(7, 4.5))\n",
        "sns.heatmap(interaction, annot=True, fmt=\".4f\", cmap=\"YlGnBu\", cbar_kws={'label': 'Mean Val MSE'}, annot_kws={\"size\": 12})\n",
        "plt.title(\"Interaction: Scaler x Sampler (Mean Val MSE)\", weight='bold', pad=15)\n",
        "plt.xlabel(\"Sampler Choice\")\n",
        "plt.ylabel(\"Scaler Choice\")\n",
        "plt.tight_layout()\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 4. Interaction Effect: Resample Freq x Start Offset Hours\n",
        "\n",
        "Analyzing how sampling frequency and start offsets interact. Resampling at a `6h` or `12h` frequency works best. A granular `4h` resampling rate performs worst across all offsets."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "interaction_freq = df_best.groupby([\"config/++data.sampler.resample_freq\", \"config/++data.sampler.start_offset_hours\"])[mse_col].mean().unstack()\n",
        "\n",
        "plt.figure(figsize=(8, 5))\n",
        "sns.heatmap(interaction_freq, annot=True, fmt=\".4f\", cmap=\"coolwarm\", cbar_kws={'label': 'Mean Val MSE'}, annot_kws={\"size\": 11})\n",
        "plt.title(\"Interaction: Resample Freq x Start Offset (Mean Val MSE)\", weight='bold', pad=15)\n",
        "plt.xlabel(\"Start Offset Hours\")\n",
        "plt.ylabel(\"Resample Frequency\")\n",
        "plt.tight_layout()\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 5. Non-Minimum Values Performance (val/mae_non_min)\n",
        "\n",
        "The target variable (suicide risk) typically has a high proportion of minimum values (e.g., zeros). Analyzing the predicting non-minimum values performance (`val/mae_non_min`) allows us to see how well the model predicts actual clinical elevations rather than just predicting the baseline baseline score.\n",
        "\n",
        "Let's look at how the main filters and hyperparameters affect predicting non-minimum values."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "mae_non_min_col = \"metric/val/mae_non_min\"\n",
        "df_suicide_mae = df_suicide.dropna(subset=[mae_non_min_col]).copy()\n",
        "df_suicide_mae[mae_non_min_col] = pd.to_numeric(df_suicide_mae[mae_non_min_col], errors='coerce')\n",
        "\n",
        "# Plot OS Filter Impact on val/mae_non_min\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "sns.boxplot(data=df_suicide_mae, x=\"config/data.os_filter\", y=mae_non_min_col, palette=\"Set2\", ax=axes[0])\n",
        "axes[0].set_title(\"val/mae_non_min by OS Filter\", weight='bold', pad=15)\n",
        "axes[0].set_xlabel(\"OS Filter\")\n",
        "axes[0].set_ylabel(\"val/mae_non_min\")\n",
        "\n",
        "# Plot Collapse Strategy Impact on val/mae_non_min (iOS only)\n",
        "df_ios_mae = df_suicide_mae[df_suicide_mae[\"config/data.os_filter\"] == \"ios\"]\n",
        "sns.barplot(data=df_ios_mae, x=\"config/data.collapse_strategy\", y=mae_non_min_col, estimator=np.mean, errorbar=\"sd\", palette=\"Set3\", ax=axes[1])\n",
        "axes[1].set_title(\"Mean val/mae_non_min by Collapse Strategy (iOS only)\", weight='bold', pad=15)\n",
        "axes[1].set_xlabel(\"Collapse Strategy\")\n",
        "axes[1].set_ylabel(\"Mean val/mae_non_min (+/- SD)\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Scaler x Sampler Interaction for val/mae_non_min (iOS + Mean Collapse runs only)\n",
        "\n",
        "Let's see if the same `subject_standard` + `block` interaction holds for predicting non-minimum values."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "df_best_mae = df_ios_mae[df_ios_mae[\"config/data.collapse_strategy\"] == \"mean\"]\n",
        "interaction_mae = df_best_mae.groupby([\"config/data/scaler\", \"config/data/sampler\"])[mae_non_min_col].mean().unstack()\n",
        "\n",
        "plt.figure(figsize=(7, 4.5))\n",
        "sns.heatmap(interaction_mae, annot=True, fmt=\".4f\", cmap=\"YlGnBu\", cbar_kws={'label': 'Mean val/mae_non_min'}, annot_kws={\"size\": 12})\n",
        "plt.title(\"Interaction: Scaler x Sampler (Mean val/mae_non_min)\", weight='bold', pad=15)\n",
        "plt.xlabel(\"Sampler Choice\")\n",
        "plt.ylabel(\"Scaler Choice\")\n",
        "plt.tight_layout()\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Top 10 Configurations for val/mae_non_min (iOS + Mean Collapse)\n",
        "\n",
        "Here are the top-performing run configurations specifically optimizing for non-minimum value prediction (`val/mae_non_min`)."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "cols_to_print_mae = [\"run_name\", mae_non_min_col, \"metric/val/mse\", \"config/data/scaler\", \"config/data/sampler\", \"config/++data.sampler.lookback_days\", \"config/++data.sampler.resample_freq\"]\n",
        "df_top_mae = df_best_mae.sort_values(by=mae_non_min_col)[cols_to_print_mae].head(10).copy()\n",
        "df_top_mae.columns = [c.split(\"/\")[-1].split(\".\")[-1] for c in cols_to_print_mae]\n",
        "df_top_mae"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 6. Top 10 Configurations for val/mse (iOS + Mean Collapse)\n",
        "\n",
        "Here are the absolute top-performing run configurations optimizing for overall MSE (`metric/val/mse`)."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "cols_to_print = [\"run_name\", mse_col, \"metric/val/mae_non_min\", \"config/data/scaler\", \"config/data/sampler\", \"config/++data.sampler.lookback_days\", \"config/++data.sampler.resample_freq\", \"config/++data.sampler.start_offset_hours\"]\n",
        "df_top = df_best.sort_values(by=mse_col)[cols_to_print].head(10).copy()\n",
        "df_top.columns = [c.split(\"/\")[-1].split(\".\")[-1] for c in cols_to_print]\n",
        "df_top"
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
       "codemirror_mode": {
        "name": "ipython",
        "version": 3
       },
       "file_extension": ".py",
       "mimetype": "text/x-python",
       "name": "python",
       "nbconvert_exporter": "python",
       "pygments_lexer": "ipython3",
       "version": "3.12.0"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 2
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "sweep_visualization.ipynb")
    with open(output_path, "w") as f:
        json.dump(notebook_content, f, indent=1)
    print(f"Generated notebook at {output_path}")

if __name__ == "__main__":
    main()
