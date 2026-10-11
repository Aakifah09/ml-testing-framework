"""Ensemble pattern - several models behind one ModelAdapter interface.

EnsemblePredictor is a composite: it *is* a ModelAdapter and *contains*
ModelAdapters, so anything that accepts a single model accepts an ensemble.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence

import numpy as np

from mltest.models.adapters import ModelAdapter


def majority_vote(votes: Sequence[Hashable], weights: Sequence[float] | None = None) -> Hashable:
    """Weighted majority; ties go to the option that was voted for first."""
    if not votes:
        raise ValueError("majority_vote needs at least one vote")
    weights = weights if weights is not None else [1.0] * len(votes)
    totals: dict[Hashable, float] = defaultdict(float)
    for vote, weight in zip(votes, weights):
        totals[vote] += weight
    best = max(totals.values())
    return next(vote for vote in votes if totals[vote] == best)


class EnsemblePredictor(ModelAdapter):
    def __init__(
        self,
        members: Sequence[ModelAdapter],
        voting: str = "soft",
        weights: Sequence[float] | None = None,
        name: str = "Ensemble",
    ) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        if voting not in ("soft", "hard"):
            raise ValueError("voting must be 'soft' or 'hard'")
        if weights is not None and len(weights) != len(members):
            raise ValueError("one weight per member is required")
        self.members = list(members)
        self.voting = voting
        self.weights = np.asarray(weights if weights is not None else [1.0] * len(members), dtype=float)
        self.name = name
        self._classes: np.ndarray | None = None

    def fit(self, X, y) -> "EnsemblePredictor":
        for member in self.members:
            member.fit(X, y)
        self._classes = np.unique(np.asarray(y))
        return self

    def _aligned_proba(self, member: ModelAdapter, X) -> np.ndarray:
        """Member probabilities re-ordered to the ensemble's class order."""
        proba = member.predict_proba(X)
        aligned = np.zeros((proba.shape[0], len(self.classes_)))
        for column, label in enumerate(member.classes_):
            aligned[:, int(np.searchsorted(self.classes_, label))] = proba[:, column]
        return aligned

    def predict_proba(self, X) -> np.ndarray:
        if self.voting == "soft":
            stacked = np.stack([self._aligned_proba(m, X) for m in self.members])
            return np.average(stacked, axis=0, weights=self.weights)
        # hard voting: probability = weighted share of votes
        votes = np.stack([m.predict(X) for m in self.members])  # (members, samples)
        shares = np.zeros((votes.shape[1], len(self.classes_)))
        for weight, row in zip(self.weights, votes):
            shares[np.arange(votes.shape[1]), np.searchsorted(self.classes_, row)] += weight
        return shares / self.weights.sum()

    def predict(self, X) -> np.ndarray:
        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return self.combine_predictions([m.predict(X) for m in self.members])

    def combine_predictions(self, predictions: Sequence[np.ndarray]) -> np.ndarray:
        """Hard-vote per-sample over the members' label predictions."""
        stacked = np.stack([np.asarray(p) for p in predictions])
        return np.array([majority_vote(list(column), list(self.weights)) for column in stacked.T])

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("EnsemblePredictor has not been fitted")
        return self._classes
