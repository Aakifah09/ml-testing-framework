"""Adapter pattern (structural) - one prediction interface for every ML framework.

scikit-learn estimators and PyTorch modules have different APIs (``fit`` vs a
training loop, ``predict_proba`` vs logits, missing probabilities on SVMs...).
Callers only ever see ``ModelAdapter.fit / predict / predict_proba``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np


class ModelAdapter(ABC):
    name: str = "model"

    @abstractmethod
    def fit(self, X, y) -> "ModelAdapter":
        """Train on features ``X`` and labels ``y``."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predicted labels."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, shape ``(n_samples, n_classes)``, ordered like ``classes_``."""

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """Labels seen during training."""

    def positive_scores(self, X, positive_label: Any = 1) -> np.ndarray:
        """Probability of ``positive_label`` (e.g. 'defective') for each sample."""
        proba = self.predict_proba(X)
        classes = list(self.classes_)
        if positive_label not in classes:
            return np.zeros(len(proba))
        return proba[:, classes.index(positive_label)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


class SklearnAdapter(ModelAdapter):
    """Adapts any scikit-learn classifier."""

    def __init__(self, estimator: Any, name: str | None = None) -> None:
        self.estimator = estimator
        self.name = name or type(estimator).__name__

    def fit(self, X, y) -> "SklearnAdapter":
        self.estimator.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(np.asarray(X))

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        # e.g. SVC(probability=False): turn decision scores into probabilities.
        scores = self.estimator.decision_function(X)
        if scores.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - positive, positive])
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    @property
    def classes_(self) -> np.ndarray:
        return self.estimator.classes_


def build_torch_mlp(n_features: int, n_classes: int, hidden: Sequence[int] = (64, 32), dropout: float = 0.1):
    """Module builder for PyTorchAdapter (module-level so adapters stay picklable)."""
    from torch import nn

    layers: list[Any] = []
    width = n_features
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
        width = size
    layers.append(nn.Linear(width, n_classes))
    return nn.Sequential(*layers)


class PyTorchAdapter(ModelAdapter):
    """Adapts a PyTorch ``nn.Module`` (plus a training loop) to ModelAdapter."""

    def __init__(
        self,
        module_builder: Callable[[int, int], Any] = partial(build_torch_mlp, hidden=(64, 32)),
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 64,
        balance_classes: bool = True,
        seed: int = 42,
        name: str = "PyTorchMLP",
    ) -> None:
        self.module_builder = module_builder
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.balance_classes = balance_classes
        self.seed = seed
        self.name = name
        self.module = None
        self._classes: np.ndarray | None = None

    def fit(self, X, y) -> "PyTorchAdapter":
        import torch

        torch.manual_seed(self.seed)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = np.asarray(y)
        self._classes = np.unique(y)
        y_t = torch.as_tensor(np.searchsorted(self._classes, y), dtype=torch.long)

        self.module = self.module_builder(X_t.shape[1], len(self._classes))
        weight = None
        if self.balance_classes:
            counts = torch.bincount(y_t, minlength=len(self._classes)).float().clamp(min=1)
            weight = counts.sum() / (len(counts) * counts)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        generator = torch.Generator().manual_seed(self.seed)

        self.module.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(X_t), generator=generator)
            for start in range(0, len(X_t), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.module(X_t[batch]), y_t[batch])
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        if self.module is None:
            raise RuntimeError("PyTorchAdapter must be fitted before predicting")
        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.as_tensor(np.asarray(X, dtype=np.float32)))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("PyTorchAdapter has not been fitted")
        return self._classes
