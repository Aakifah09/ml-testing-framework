"""Strategy pattern (behavioral) - interchangeable defect-prediction approaches.

Both strategies get their models from ModelFactory, so they work with any
registered model (SVM, decision tree, random forest, neural network, torch...).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from mltest.core.factory import DefectStrategyFactory, ModelFactory
from mltest.models.adapters import ModelAdapter
from mltest.models.ensemble import EnsemblePredictor


class DefectPredictionStrategy(ABC):
    name = "base"

    @property
    @abstractmethod
    def model(self) -> ModelAdapter:
        """The underlying (adapted) model."""

    def train(self, X, y) -> "DefectPredictionStrategy":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Probability that each module is defective."""
        return self.model.positive_scores(X, positive_label=1)


@DefectStrategyFactory.register("single_model")
class SingleModelStrategy(DefectPredictionStrategy):
    def __init__(self, model_type: str, seed: int = 42, **model_params: Any) -> None:
        self.name = model_type
        self._model = ModelFactory.create_model(model_type, seed=seed, **model_params)

    @property
    def model(self) -> ModelAdapter:
        return self._model


@DefectStrategyFactory.register("ensemble")
class EnsembleStrategy(DefectPredictionStrategy):
    def __init__(self, members: Sequence[str], voting: str = "soft", weights: Sequence[float] | None = None,
                 seed: int = 42) -> None:
        self.name = f"ensemble_{voting}"
        self._model = EnsemblePredictor(
            [ModelFactory.create_model(m, seed=seed) for m in members], voting=voting, weights=weights,
            name=f"Ensemble({'+'.join(members)})",
        )

    @property
    def model(self) -> ModelAdapter:
        return self._model
