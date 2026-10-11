"""Singleton pattern (creational) - AppConfig and ModelManager.

Every component reads the same configuration (paths, parameters, seed) and
shares one registry of trained models, so both are process-wide singletons.
"""
from __future__ import annotations

import copy
import json
import pickle
import random
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class SingletonMeta(type):
    """Metaclass that hands back the same instance on every instantiation."""

    _instances: dict[type, Any] = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with SingletonMeta._lock:
            if cls not in SingletonMeta._instances:
                SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta._instances[cls]

    def reset_instance(cls) -> None:
        """Forget the cached instance (used by tests to start from a clean state)."""
        with SingletonMeta._lock:
            SingletonMeta._instances.pop(cls, None)


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "output_dir": "results",
    "modules": ["test_generation", "defect_prediction", "test_oracle"],
    "decorators": ["logging", "timing"],
    "observers": ["console", "report"],
    "test_generation": {
        "programs": ["triangle", "days_in_month", "gcd", "insertion_sort"],
        "strategies": ["random", "search", "ml"],
        "budget": 200,
        "strategy_params": {"ml": {"model_type": "decision_tree"}},
        "write_pytest_files": True,
        "show_examples": 3,
    },
    "defect_prediction": {
        "data_source": "synthetic",
        "loader_params": {"n_samples": 2000},
        "test_size": 0.3,
        "models": ["decision_tree", "svm", "random_forest", "neural_network"],
        "ensemble": {"enabled": True, "members": None, "voting": "soft", "weights": None},
        "save_models": False,
        "show_examples": 3,
    },
    "test_oracle": {
        "programs": ["triangle", "days_in_month", "gcd", "insertion_sort"],
        "strategies": ["rule", "metamorphic", "ml", "hybrid"],
        "n_inputs": 60,
        "input_strategy": "search",
        "train_fraction": 0.5,
        "show_examples": 3,
        "strategy_params": {
            "ml": {"model_type": "random_forest"},
            "hybrid": {"members": ["rule", "metamorphic", "ml"], "voting": "majority"},
        },
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class AppConfig(metaclass=SingletonMeta):
    """Shared configuration: testing modules, parameters, dataset, paths and seed."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self._run_dir: Path | None = None

    # -- loading / updating -------------------------------------------------
    def load(self, path: str | Path) -> "AppConfig":
        with open(path, encoding="utf-8") as fh:
            self.update(json.load(fh))
        return self

    def update(self, overrides: dict[str, Any]) -> "AppConfig":
        _deep_merge(self._data, overrides)
        if "output_dir" in overrides:
            self._run_dir = None
        return self

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        *parents, leaf = dotted_key.split(".")
        node = self._data
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
        if dotted_key == "output_dir":
            self._run_dir = None

    def section(self, name: str) -> dict[str, Any]:
        """A copy of one module's settings, so callers cannot mutate shared state."""
        return copy.deepcopy(self._data.get(name, {}))

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- shared values ------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self._data["seed"])

    @property
    def run_dir(self) -> Path:
        """Output folder of the current run, created on first use."""
        if self._run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_dir = Path(self._data["output_dir"]) / f"run_{stamp}"
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir

    def seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        if "torch" in sys.modules:
            sys.modules["torch"].manual_seed(self.seed)


class ModelManager(metaclass=SingletonMeta):
    """Shared registry of trained models, reachable from any module."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(self, key: str, model: Any, **metadata: Any) -> None:
        self._models[key] = model
        self._metadata[key] = metadata

    def get(self, key: str) -> Any:
        try:
            return self._models[key]
        except KeyError:
            raise KeyError(f"No model registered under '{key}'. Known: {sorted(self._models)}") from None

    def metadata(self, key: str) -> dict[str, Any]:
        return dict(self._metadata.get(key, {}))

    def keys(self) -> list[str]:
        return sorted(self._models)

    def save(self, key: str, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (key.replace("/", "__") + ".pkl")
        with open(path, "wb") as fh:
            pickle.dump(self._models[key], fh)
        return path

    def clear(self) -> None:
        self._models.clear()
        self._metadata.clear()
