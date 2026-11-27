"""Convert the aligned 5′/3′ short duplex in panel.csv to the shared numeric input."""

from itertools import product

import numpy as np


PAIRS = ("AU", "UA", "CG", "GC", "GU", "UG")
PAIR_CLASS = dict(AU="AU", UA="AU", CG="GC", GC="GC", GU="GU", UG="GU")
CLASSES = ("AU", "GC", "GU")


def stack_key(first, second):
    return min(first + second, second[::-1] + first[::-1])


STACKS = sorted({stack_key(a, b) for a, b in product(PAIRS, repeat=2)})


def features(duplex):
    top, bottom = duplex.split("/")
    pairs = [a + b for a, b in zip(top, bottom, strict=True)]
    kinds = [PAIR_CLASS[pair] for pair in pairs]
    ends = (kinds[0], kinds[-1])
    adjacent = [stack_key(a, b) for a, b in zip(pairs, pairs[1:])]
    values = [len(top), int(top == bottom[::-1])]
    values.extend(kinds.count(k) for k in CLASSES)
    values.extend(ends.count(k) for k in CLASSES)
    values.extend(adjacent.count(stack) for stack in STACKS)
    values.extend([kinds.count("GU") / len(kinds), 1 / len(kinds)])
    terminal_neighbors = [(kinds[0], kinds[1]), (kinds[-1], kinds[-2])]
    values.extend(terminal_neighbors.count(pair) for pair in product(CLASSES, repeat=2))
    return np.array(values, dtype=np.float64)
