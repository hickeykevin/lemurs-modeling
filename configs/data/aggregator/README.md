# 📊 Data Aggregator Configurations

In longitudinal health studies, surveys often contain multiple questions. An **Aggregator** defines the mathematical or clinical rules used to collapse multiple survey responses into a single target label: either a binary classification label ($0$ or $1$) or a continuous regression score (e.g. summing Likert scale questions) for our models to predict.

---

## ⚙️ Rule-Based Aggregations

We use the `RuleBasedAggregator` class to translate clinical criteria into logical targets.

### Available Operators
*   `ge`: Greater than or equal to (e.g., at least one answer in the set is $\ge X$).
*   `any_eq`: Any equal to (e.g., at least one answer is exactly $X$).
*   `any_gt`: Any greater than (e.g., at least one answer in the set is $> X$).
*   `sum`: Sums up scores of all questions in the set and compares to a threshold.
*   `mean`: Computes the mean score of questions and compares to a threshold.

### Available Presets
Swapping targets is done by setting `data/aggregator=preset_name`:

*   `suicide_risk.yaml`: Targets questions related to self-harm and active ideation.
*   `social_stress.yaml`: Targets questions regarding interpersonal friction and conflict.
*   `emotion_regulation.yaml`: Targets coping strategy questions.
*   `minority_stress.yaml`: Targets questions measuring identity-based discrimination.
*   `positive_emotion.yaml` / `negative_emotion.yaml`: Group mood descriptors.

---

## 📈 Regression-Based Aggregations

We use the `RegressionAggregator` class to aggregate survey answers by summing Likert scale questions to produce a continuous target score.

### Configuration Properties
*   `likert_ids` (list[int]): List of question IDs to target and sum.
*   `shift_likert` (bool): Defaults to `true`. If `true`, shifts 1-indexed Likert scales to start at 0 by subtracting 1.
    *   **Auto-Index Verification**: The aggregator automatically checks if any target answer is already `0` (i.e. already 0-indexed in the dataset). If so, it dynamically skips the subtracting-by-1 step and logs a warning to prevent double-shifting.

### Available Presets
Swapping to a regression target is done by setting `data/aggregator=preset_name_regression`:

*   `suicide_risk_regression.yaml`: Sums Likert-scale questions regarding self-harm/ideation (ignores binary yes/no or conditional questions).
*   `self_harm_regression.yaml`: Sums Likert-scale questions regarding self-harm.
*   `minority_stress_regression.yaml`: Sums Likert-scale questions measuring identity-based discrimination.
*   `social_connection_regression.yaml`: Sums Likert-scale questions regarding social support/connection.
*   `social_stress_regression.yaml`: Sums Likert-scale questions regarding interpersonal stress/friction.
*   `emotion_regulation_regression.yaml`: Sums Likert-scale questions regarding emotional coping/regulation.
*   `positive_emotion_regression.yaml` / `negative_emotion_regression.yaml`: Sums Likert-scale positive / negative mood descriptors.

---

## ⚡ CLI Override Scenarios

### Scenario A: High-Severity Target Filtering
By default, the `suicide_risk` aggregator uses lower thresholds to capture mild risk cases. To retarget the model to predict high-severity ideation:
```bash
# Set threshold for the rule group 'rule_1' (Questions 2, 3, 7) to at least 4
uv run src/train.py data/aggregator=suicide_risk data.aggregator.rules.rule_1.val=4
```

### Scenario B: Restrictive And/Or Logic
To require **all** rules in a preset config to evaluate to True before assignment of label $1$:
```bash
uv run src/train.py data/aggregator=minority_stress data.aggregator.combination_logic=all
```

### Scenario C: Custom Numeric Threshold splits
To bypass clinical rules and construct a target mapping using a simple average of questions 1, 2, and 3:
```bash
uv run src/train.py data/aggregator=suicide_risk \
  data.aggregator._target_=src.data.components.label_aggregators.MeanAggregator \
  data.aggregator.threshold=0.5 \
  "data.aggregator.question_ids=[1,2,3]"
```

---

## 🛠️ How to Create a New Aggregator

1. Create a YAML config inside this directory (e.g., `configs/data/aggregator/anxiety.yaml`).
2. Point `_target_` to the `RuleBasedAggregator`.
3. Add your rules dictionary:

```yaml
_target_: src.data.components.label_aggregators.RuleBasedAggregator
combination_logic: "any" # "any" (OR) or "all" (AND)
rules:
  rule_1:
    ids: [15, 16] # Question IDs to evaluate
    op: "ge"       # Operator
    val: 3         # Target threshold
```

Run training targeting this aggregator:
```bash
uv run src/train.py data/aggregator=anxiety
```

### Scenario D: Custom Regression Target Questions
To bypass the preset questions and sum a custom list of question IDs for regression:
```bash
uv run src/train.py experiment=regression_health \
  data/aggregator=suicide_risk_regression \
  "data.aggregator.likert_ids=[2,3,5]"
```

