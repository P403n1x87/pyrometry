from random import gauss

import numpy as np
from scipy.stats import ttest_ind

from pyrometry.analysis import compare
from pyrometry.flamegraph import FlameGraph
from pyrometry.stats import hotelling_two_sample_test


def test_hotelling_single_feature_is_student_t() -> None:
    x = [gauss(110, 5) for _ in range(20)]
    y = [gauss(100, 5) for _ in range(15)]

    delta, _, p, m = hotelling_two_sample_test(
        np.array(x)[:, None], np.array(y)[:, None]
    )

    # With one feature, Hotelling's T^2 is the square of Student's t.
    assert np.isclose(p, ttest_ind(x, y, equal_var=True).pvalue)
    assert np.isclose(delta[0], np.mean(x) - np.mean(y))
    assert len(m) == 1


def test_compare_single_stack() -> None:
    x = [FlameGraph({"a": gauss(200, 1)}) for _ in range(10)]
    y = [FlameGraph({"a": gauss(100, 1)}) for _ in range(10)]

    delta, _, p = compare(x, y)

    assert p < 0.01
    assert set(delta) == {"a"}
