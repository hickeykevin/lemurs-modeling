import numpy as np
import pytest

from src.data.components.tabular_features import SummaryStatsFeaturizer

#testing whether the featurizer calculates the statistics correctly.
def test_summary_stats_shape_and_values():
    # 2 samples, 4 time steps, 2 modalities (step, calorie)
    x = np.array([
        [  #step #cal
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
        ],
        [
            [5, 50],
            [6, 60],
            [7, 70],
            [8, 80],
        ],
    ])

    featurizer = SummaryStatsFeaturizer()
    output = featurizer.transform(x)

    # 2 samples x (2 modalities x 5 stats)
    assert output.shape == (2, 10)

    names = featurizer.feature_names(["step", "calorie"])
    values = dict(zip(names, output[0]))
    # sample 0's step column is [1,2,3,4]
    assert values["step__mean"] == pytest.approx(2.5)
    assert values["step__median"] == pytest.approx(2.5)
    assert values["step__min"] == pytest.approx(1.0)
    assert values["step__max"] == pytest.approx(4.0)
    assert values["step__std"] == pytest.approx(np.std([1, 2, 3, 4]))

#test Whether the number of feature names matches the number of output features.
def test_feature_names_match_output_width():
    x = np.random.randn(3, 5, 2)
    featurizer = SummaryStatsFeaturizer()
    output = featurizer.transform(x)
    names = featurizer.feature_names(["step", "calorie"])
    assert len(names) == output.shape[1]

#Whether the featurizer correctly removes the last 4 columns.
def test_exclude_last_n_cols_drops_trailing_columns():
    # e.g. 1 modality + 4 cyclic time-feature columns, as RollingSampler produces
    x = np.random.randn(4, 8, 5)
    featurizer = SummaryStatsFeaturizer(stats=["mean"], exclude_last_n_cols=4)
    output = featurizer.transform(x)
    assert output.shape == (4, 1)

def test_exclude_last_n_cols_auto_defaults_to_zero():
    x = np.random.randn(4, 8, 3)
    featurizer = SummaryStatsFeaturizer(stats=["mean"], exclude_last_n_cols="auto")
    output = featurizer.transform(x)
    assert output.shape == (4, 3)


#Can the featurizer work with one sample even when there is no batch dimension?
def test_single_sample_input_is_squeezed():
    x = np.random.randn(6, 2)  # [Time, Features], no batch dim
    featurizer = SummaryStatsFeaturizer()
    output = featurizer.transform(x)
    assert output.shape == (2 * 5,)  # 1D, no batch dim

#Does it catch typos or unsupported statistics instead of silently failing?
def test_unknown_stat_raises():
    with pytest.raises(ValueError):
        SummaryStatsFeaturizer(stats=["mean", "not_a_real_stat"])

#This tests what happens if there are zero time steps:
def test_empty_time_dimension_returns_zeros():
    x = np.zeros((3, 0, 2))
    featurizer = SummaryStatsFeaturizer(stats=["mean", "std"])
    output = featurizer.transform(x)
    assert output.shape == (3, 4)
    assert np.all(output == 0.0)


def test_sampler_num_time_features():
    from src.data.components.samplers import (
        RollingSampler, OffsetSampler, BlockSampler, IntervalAwareSampler, LagSampler
    )
    assert RollingSampler(include_time_features=True).num_time_features == 4
    assert RollingSampler(include_time_features=False).num_time_features == 0
    assert OffsetSampler(include_time_features=True).num_time_features == 2
    assert OffsetSampler(include_time_features=False).num_time_features == 0
    assert BlockSampler(include_time_features=True).num_time_features == 4
    assert BlockSampler(include_time_features=False).num_time_features == 0
    assert IntervalAwareSampler(include_time_features=True).num_time_features == 4
    assert IntervalAwareSampler(include_time_features=False).num_time_features == 0
    assert LagSampler().num_time_features == 0

