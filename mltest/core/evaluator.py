"""Evaluation metrics shared by the three testing modules."""
from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

import numpy as np
from sklearn import metrics


class Stopwatch:
    """``with Stopwatch() as sw: ...`` then read ``sw.seconds``."""

    def __enter__(self) -> "Stopwatch":
        self._start = perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = perf_counter() - self._start


class Evaluator:
    @staticmethod
    def accuracy(y_true, y_pred) -> float:
        return float(metrics.accuracy_score(y_true, y_pred))

    @staticmethod
    def precision(y_true, y_pred) -> float:
        return float(metrics.precision_score(y_true, y_pred, zero_division=0))

    @staticmethod
    def recall(y_true, y_pred) -> float:
        return float(metrics.recall_score(y_true, y_pred, zero_division=0))

    @staticmethod
    def f1(y_true, y_pred) -> float:
        return float(metrics.f1_score(y_true, y_pred, zero_division=0))

    @staticmethod
    def roc_auc(y_true, scores) -> float:
        try:
            return float(metrics.roc_auc_score(y_true, scores))
        except ValueError:  # only one class present
            return math.nan

    @staticmethod
    def false_positive_rate(y_true, y_pred) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        negatives = np.sum(y_true == 0)
        return float(np.sum((y_pred == 1) & (y_true == 0)) / negatives) if negatives else 0.0

    @staticmethod
    def false_negative_rate(y_true, y_pred) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        positives = np.sum(y_true == 1)
        return float(np.sum((y_pred == 0) & (y_true == 1)) / positives) if positives else 0.0

    @classmethod
    def classification(cls, y_true, y_pred, scores: Sequence[float] | None = None) -> dict[str, float]:
        report = {
            "accuracy": cls.accuracy(y_true, y_pred),
            "precision": cls.precision(y_true, y_pred),
            "recall": cls.recall(y_true, y_pred),
            "f1": cls.f1(y_true, y_pred),
        }
        if scores is not None:
            report["roc_auc"] = cls.roc_auc(y_true, scores)
        return report

    @staticmethod
    def coverage(covered: int, total: int) -> float:
        return covered / total if total else 1.0

    @staticmethod
    def mutation_score(killed: int, total: int) -> float:
        return killed / total if total else 0.0
