import numpy as np
import pytest

from src.data.components.catch22_features import Catch22Featurizer

def test_catch22_matches_direct_pycatch22_call():
    """Known-value check for catch22: unlike mean/std, catch22's 22 features
    aren't hand-computable by simple arithmetic, so "known value" here means
    matching a direct pycatch22.catch22_all() call on the same raw data --
    confirms the wrapper (batching, column selection, naming) doesn't distort
    what catch22 itself computes."""
    import pycatch22

    # 2 samples, 4 time steps, 2 modalities (step, calorie) -- same array as
    # test_summary_stats_shape_and_values in test_tabular_features.py
    x = np.array([
        [  # step  cal
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

    featurizer = Catch22Featurizer()
    output = featurizer.transform(x)

    # 2 samples x (2 modalities x 22 catch22 features)
    assert output.shape == (2, 44)

    names = featurizer.feature_names(["step", "calorie"])
    values = dict(zip(names, output[0]))

    # sample 0's step column is [1,2,3,4] -- cross-check every catch22
    # feature against calling pycatch22 directly on that same column.
    direct = pycatch22.catch22_all([1.0, 2.0, 3.0, 4.0])
    print()
    print(f"{'feature':38s} {'ours':>12s} {'direct pycatch22':>18s}")
    for stat_name, direct_val in zip(direct["names"], direct["values"]):
        ours = values[f"step__{stat_name}"]
        print(f"{stat_name:38s} {ours:12.4f} {direct_val:18.4f}")
        if np.isfinite(direct_val):
            assert ours == pytest.approx(direct_val), stat_name
        else:
            # catch22 itself returned NaN/inf for this feature on this short
            # series -- our fill_value (0.0 default) should have replaced it.
            assert np.isfinite(ours), stat_name
