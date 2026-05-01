# Data Aggregator Configurations

This directory contains configurations for **Label Aggregation** strategies. Since surveys often contain multiple questions, an aggregator defines the logic used to collapse those answers into a single target label (usually 0 or 1) for the model to predict.

## Rule-Based Aggregation

Most configs in this directory use the `RuleBasedAggregator`. This allows you to define complex clinical logic using simple rules.

### How Rules Work:
A rule targets specific `ids` (Question IDs) and applies an `op` (Operator):
- `ge`: "Greater than or equal to" (e.g., *at least one* answer in the group is $\ge X$)
- `any_eq`: "Any equal to" (e.g., *at least one* answer is exactly $X$)
- `sum`: Calculates the total score of the group and compares to a threshold.
- `mean`: Calculates the average score and compares to a threshold.

### Available Aggregators:
- `suicide_risk.yaml`: Targets questions related to self-harm and ideation.
- `social_stress.yaml`: Targets interpersonal conflict questions.
- `emotion_regulation.yaml`: Targets questions about coping mechanisms.
- *(See the full list of .yaml files for other domains)*

## Common Modeling Scenarios

### Scenario A: "Modeling High-Severity Suicide Risk"
By default, the `suicide_risk` aggregator uses a low threshold. You can make it stricter to target only high-severity cases.
```bash
# Change rule 0 (Question IDs 2, 3, 7) to require a value of at least 4
python src/train.py data/aggregator=suicide_risk "data.aggregator.rules[0].val=4"
```

### Scenario B: "Predicting Sustained Minority Stress"
If you want to define risk as "experiencing multiple symptoms at once," change the `combination_logic` to `all`.
```bash
# Requires ALL rules in the minority_stress config to be True
python src/train.py data/aggregator=minority_stress data.aggregator.combination_logic=all
```

### Scenario C: "Binarizing based on Mean Score"
Instead of clinical rules, you might want a simple binary split based on the average of all questions.
```bash
# Use the MeanAggregator with a 0.5 threshold on questions 1, 2, and 3
python src/train.py data/aggregator=suicide_risk \
  data.aggregator._target_=src.data.components.label_aggregators.MeanAggregator \
  data.aggregator.threshold=0.5 \
  "data.aggregator.question_ids=[1,2,3]"
```

---

## How to use in Experiments

### Basic Swap
To change what the model is trying to predict, swap the aggregator:
```bash
# Predict suicide risk
python src/train.py data/aggregator=suicide_risk

# Predict social stress instead
python src/train.py data/aggregator=social_stress
```

### Overriding Logic on the Fly
You can change the logic of an aggregator without creating a new file. For example, to make the `suicide_risk` criteria stricter:
```bash
# Change the 'val' (threshold) of the first rule (index 0) to 3
python src/train.py data/aggregator=suicide_risk "data.aggregator.rules[0].val=3"
```

## Creating a New Aggregator
1. Create a new `.yaml` file in this directory.
2. Set `_target_` to `src.data.components.label_aggregators.RuleBasedAggregator`.
3. Define your `rules` list and `combination_logic` ("any" for OR, "all" for AND).

**Example `new_phenotype.yaml`:**
```yaml
_target_: src.data.components.label_aggregators.RuleBasedAggregator
combination_logic: "any"
rules:
  - ids: [10, 11]  # Questions 10 and 11
    op: "ge"
    val: 2         # If either is >= 2, label becomes 1
```
