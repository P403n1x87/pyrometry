import typing as t
from collections import Counter
from itertools import chain

import numpy as np

from pyrometry.flamegraph import FlameGraph
from pyrometry.stats import hotelling_two_sample_test


def compare(
    x: t.List[FlameGraph],
    y: t.List[FlameGraph],
    threshold: t.Optional[float] = None,
) -> t.Tuple[FlameGraph, float, float]:
    domain = list(set().union(*(_.supp() for _ in chain(x, y))))

    if threshold is not None:
        c: t.Counter[t.Any] = Counter()
        for _ in chain(x, y):
            c.update(_.supp())
        domain = sorted([k for k, v in c.items() if v >= threshold])

    X = np.array([[f(v) for v in domain] for f in x], dtype=np.int32)
    Y = np.array([[f(v) for v in domain] for f in y], dtype=np.int32)

    d, f, p, m = hotelling_two_sample_test(X, Y)

    delta = FlameGraph({k: v for k, v, a in zip(domain, d, m) if v and a})

    return delta, f, p


def decompose_2way(
    x: t.List[FlameGraph],
    y: t.List[FlameGraph],
    threshold: t.Optional[float] = None,
) -> t.Tuple[FlameGraph, FlameGraph]:
    """Decompose the difference X - Y into positive and negative parts."""
    assert x, "x must not be empty"
    assert y, "y must not be empty"

    fg_class = type(x[0])

    delta, _, _ = compare(x, y, threshold)
    return (
        fg_class({k: v for k, v in delta.items() if v > 0}),
        fg_class({k: -v for k, v in delta.items() if v < 0}),
    )


def decompose_4way(
    x: t.List[FlameGraph],
    y: t.List[FlameGraph],
    threshold: t.Optional[float] = None,
) -> t.Tuple[FlameGraph, FlameGraph, FlameGraph, FlameGraph]:
    """Decompose the difference X - Y into appeared, disappeared, grown, and shrunk parts."""
    assert x, "x must not be empty"
    assert y, "y must not be empty"

    fg_class = type(x[0])

    x_domain = set().union(*(x.supp() for x in x))
    y_domain = set().union(*(y.supp() for y in y))

    plus, minus = decompose_2way(x, y, threshold)

    appeared = fg_class({k: v for k, v in plus.items() if k not in y_domain})
    disappeared = fg_class({k: v for k, v in minus.items() if k not in x_domain})
    grown = fg_class({k: v for k, v in plus.items() if k in y_domain})
    shrunk = fg_class({k: v for k, v in minus.items() if k in x_domain})

    return appeared, disappeared, grown, shrunk
