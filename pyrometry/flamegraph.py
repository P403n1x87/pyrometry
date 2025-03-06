import typing as t
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np

from pyrometry.stats import hotelling_two_sample_test


class Map(dict):
    """Element of a free module over a ring."""

    def __call__(self, x):
        # TODO: 0 is not the generic element of the ring
        return self.get(x, 0)

    def __add__(self, other):
        m = self.__class__(self)
        for k, v in other.items():
            n = m.setdefault(k, v.__class__()) + v
            if not n and k in m:
                del m[k]
                continue
            m[k] = n
        return m

    def __mul__(self, other):
        m = self.__class__(self)
        for k, v in self.items():
            n = v * other
            if not n and k in m:
                del m[k]
                continue
            m[k] = n
        return m

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self * (1 / other)

    def __rtruediv__(self, other):
        return self.__div__(other)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        m = Map(self)
        for k, v in m.items():
            m[k] = -v
        return m

    def supp(self):
        return set(self.keys())


class FlameGraph(Map):
    def norm(self):
        return sum(abs(v) for v in self.values())

    @classmethod
    def parse(cls, file: Path) -> "FlameGraph":
        fg = cls()

        for line in (_.strip() for _ in file.open()):
            stack, _, m = line.rpartition(" ")

            fg += cls({stack: int(m)})

        return fg

    def __str__(self):
        return "\n".join(f"{k} {v}" for k, v in self.items())


def compare(
    x: list[FlameGraph],
    y: list[FlameGraph],
    threshold: t.Optional[float] = None,
) -> t.Tuple[FlameGraph, float, float]:
    domain = list(set().union(*(_.supp() for _ in chain(x, y))))

    if threshold is not None:
        c = Counter()
        for _ in chain(x, y):
            c.update(_.supp())
        domain = sorted([k for k, v in c.items() if v >= threshold])

    X = np.array([[f(v) for v in domain] for f in x], dtype=np.int32)
    Y = np.array([[f(v) for v in domain] for f in y], dtype=np.int32)

    d, f, p, m = hotelling_two_sample_test(X, Y)

    delta = FlameGraph({k: v for k, v, a in zip(domain, d, m) if v and a})

    return delta, f, p


def decompose_2way(
    x: list[FlameGraph],
    y: list[FlameGraph],
    threshold: t.Optional[float] = None,
) -> tuple[FlameGraph, FlameGraph]:
    """Decompose the difference X - Y into positive and negative parts."""
    delta, _, _ = compare(x, y, threshold)
    return (
        FlameGraph({k: v for k, v in delta.items() if v > 0}),
        FlameGraph({k: -v for k, v in delta.items() if v < 0}),
    )


def decompose_4way(
    x: list[FlameGraph], y: list[FlameGraph]
) -> tuple[FlameGraph, FlameGraph, FlameGraph, FlameGraph]:
    """Decompose the difference X - Y into appeared, disappeared, grown, and shrunk parts."""
    x_domain = set().union(*(x.supp() for x in x))
    y_domain = set().union(*(y.supp() for y in y))

    plus, minus = decompose_2way(x, y)

    appeared = FlameGraph({k: v for k, v in plus.items() if k not in y_domain})
    disappeared = FlameGraph({k: v for k, v in minus.items() if k not in x_domain})
    grown = FlameGraph({k: v for k, v in plus.items() if k in y_domain})
    shrunk = FlameGraph({k: v for k, v in minus.items() if k in x_domain})

    return appeared, disappeared, grown, shrunk
