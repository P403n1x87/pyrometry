from pathlib import Path
from random import randint as r

from pyrometry.analysis import decompose_4way
from pyrometry.flamegraph import FlameGraph

TESTS = Path(__file__).parent
DATA = TESTS / "data"


def test_flamegraph_parse() -> None:
    assert FlameGraph.parse(DATA / "flamegraph.fg") == FlameGraph(
        {
            "a": 100000,
            "b": 200000,
            "c": 300000,
            "d": 400000,
            "e": 500000,
            "f": 600000,
            "g": 700000,
        }
    )


def test_decompose_4way() -> None:
    def n() -> int:
        return r(-10, 10)

    x = [
        FlameGraph(
            {
                "a": 100000 + n(),
                "c": 300000 + n(),
                "xa": 400000 + n(),
                "g": 500000 + n(),
                "s": 100000 + n(),
            }
        )
        for _ in range(100)
    ]

    y = [
        FlameGraph(
            {
                "a": 100000 + n(),
                "c": 300000 + n(),
                "yd": 400000 + n(),
                "g": 300000 + n(),
                "s": 400000 + n(),
            }
        )
        for _ in range(100)
    ]

    a, d, g, s = decompose_4way(x, y)

    assert set(a) == {"xa"}
    assert set(d) == {"yd"}
    assert set(g) == {"g"}
    assert set(s) == {"s"}
