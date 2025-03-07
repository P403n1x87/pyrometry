# pyrometry

This library provides a Python implementation of the paper [On the Algebraic
Properties of Flame Graphs][paper] for the comparison and decomposition of
flame graphs.


## Usage

To parse a flame graph from a collapsed stacks format, use `FlameGraph.load`
from `pyrometry.flamegraph`.

To perform the differential analysis of two samples of flame graphs, collect
them in two lists, and then use e.g. `decompose_4way` from `pyrometry.analysis`
to get the flame graphs of appeared, disappeared, grown and shrunk stacks.


[paper]: https://arxiv.org/pdf/2301.08941
