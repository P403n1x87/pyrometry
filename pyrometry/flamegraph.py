import typing as t
from pathlib import Path

D = t.TypeVar("D")
R = float


class Map(dict, t.Generic[D]):
    """Element of a free module over a ring."""

    def __call__(self, x: D) -> R:
        """Return the value of the map at x."""
        # TODO: 0 is not the generic element of the ring
        return self.get(x, 0)

    def __add__(self, other: "Map") -> "Map":
        m: Map = self.__class__(self)
        for k, v in other.items():
            n = m.setdefault(k, v.__class__()) + v
            if not n and k in m:
                del m[k]
                continue
            m[k] = n
        return m

    def __mul__(self, other: R) -> "Map":
        m: Map = self.__class__(self)
        for k, v in self.items():
            n = v * other
            if not n and k in m:
                del m[k]
                continue
            m[k] = n
        return m

    def __rmul__(self, other: R) -> "Map":
        return self.__mul__(other)

    def __truediv__(self, other: R) -> "Map":
        return self * (1 / other)

    def __rtruediv__(self, other: R) -> "Map":
        return self.__truediv__(other)

    def __sub__(self, other: "Map") -> "Map":
        return self + (-other)

    def __neg__(self) -> "Map":
        return type(self)({k: -v for k, v in self.items()})

    def supp(self) -> t.Set[D]:
        """Support of the map."""
        return set(self.keys())


class FlameGraph(Map):
    """Flame graph class.

    Flame graph is a map from stack traces to their magnitudes. This
    implementation can parse Brendan Gregg's flame graph collapsed stack format
    to generate a flame graph in the sense of https://arxiv.org/abs/2301.08941.
    """

    def norm(self) -> float:
        """Norm of the flame graph.

        This is the cumulative size of the root element of the flame graph.
        """
        return sum(abs(v) for v in self.values())

    @classmethod
    def loads(cls, text: str) -> "FlameGraph":
        """Parse a flame graph from a file."""
        fg: Map = cls()

        for line in text.splitlines():
            stack, _, m = line.rpartition(" ")

            fg += cls({stack: float(m)})

        return t.cast(FlameGraph, fg)

    @classmethod
    def load(cls, file: Path) -> "FlameGraph":
        """Parse a flame graph from a file."""
        return cls.loads(file.read_text())

    def __str__(self) -> str:
        return "\n".join(f"{k} {v}" for k, v in self.items())
