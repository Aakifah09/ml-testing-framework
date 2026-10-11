"""Feature extraction: turn program inputs/outputs into numeric vectors for ML."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from mltest.programs.catalog import IntParam, Param

_LIST_WIDTH = 6


def list_features(xs: Sequence[int]) -> list[float]:
    if not xs:
        return [0.0] * _LIST_WIDTH
    n = len(xs)
    ascending = sum(xs[i] <= xs[i + 1] for i in range(n - 1)) / (n - 1) if n > 1 else 1.0
    return [float(n), float(min(xs)), float(max(xs)), sum(xs) / n, ascending, float(n - len(set(xs)))]


def encode_inputs(params: Sequence[Param], inputs: tuple, moduli: Iterable[int] = ()) -> list[float]:
    """Raw values, pairwise differences of ints, and divisibility by code constants."""
    moduli = tuple(moduli)
    features: list[float] = []
    ints: list[int] = []
    for param, value in zip(params, inputs):
        if isinstance(param, IntParam):
            features.append(float(value))
            features.extend(float(value % m == 0) for m in moduli)
            ints.append(value)
        else:
            features.extend(list_features(value))
    for i in range(len(ints)):
        for j in range(i + 1, len(ints)):
            features.append(float(ints[i] - ints[j]))
    return features


def useful_moduli(constants: Iterable[int]) -> tuple[int, ...]:
    """Constants in the code worth testing divisibility against (e.g. 4, 100, 400)."""
    return tuple(sorted({abs(c) for c in constants if 2 <= abs(c) <= 1000}))


class OutputEncoder:
    """Fixed-width encoding of a program output (str labels one-hot encoded)."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def fit(self, outputs: Iterable[Any]) -> "OutputEncoder":
        for output in outputs:
            if isinstance(output, str) and output not in self.vocabulary:
                self.vocabulary[output] = len(self.vocabulary)
        return self

    def transform(self, inputs: tuple, output: Any, error: str | None) -> list[float]:
        is_number = isinstance(output, (int, float)) and not isinstance(output, bool)
        is_list = isinstance(output, list)
        features = [float(error is not None), float(output is None), float(output) if is_number else 0.0]

        one_hot = [0.0] * (len(self.vocabulary) + 1)  # last slot = unseen label
        if isinstance(output, str):
            one_hot[self.vocabulary.get(output, len(self.vocabulary))] = 1.0
        features += one_hot

        numeric_list = is_list and all(isinstance(x, (int, float)) for x in output)
        features += list_features(output) if numeric_list else [0.0] * _LIST_WIDTH
        source = next((x for x in inputs if isinstance(x, list)), None)
        if numeric_list and source is not None:
            features += [float(len(output) - len(source)), float(sum(output) - sum(source))]
        else:
            features += [0.0, 0.0]
        return features
