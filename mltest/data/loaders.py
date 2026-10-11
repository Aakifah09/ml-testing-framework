"""Data loaders, registered with DataLoaderFactory.

* ``csv``       - any defect dataset on disk (e.g. NASA MDP / PROMISE KC1, PC1, JM1 CSVs)
* ``openml``    - downloads a PROMISE dataset from OpenML (needs internet)
* ``synthetic`` - generates NASA-MDP-style McCabe/Halstead metrics offline
* ``programs``  - the subject programs used by test generation and the oracle
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mltest.core.factory import DataLoaderFactory
from mltest.programs.adapter import ProgramAdapter
from mltest.programs.catalog import PROGRAMS, get_program

log = logging.getLogger("mltest.data")

_TRUE_STRINGS = {"true", "t", "yes", "y", "1", "buggy", "defective", "bug"}


@dataclass
class TabularDataset:
    frame: pd.DataFrame
    label: str
    name: str
    source: str

    @property
    def defect_rate(self) -> float:
        return float(self.frame[self.label].mean())


def to_binary_labels(series: pd.Series) -> pd.Series:
    """bool / 'true'/'yes'/'Y' / bug counts -> 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) > 0).astype(int)
    return series.astype(str).str.strip().str.lower().isin(_TRUE_STRINGS).astype(int)


class DataLoader(ABC):
    @abstractmethod
    def load(self) -> Any:
        """Return the loaded data."""


@DataLoaderFactory.register("csv")
class CSVDataLoader(DataLoader):
    def __init__(self, path: str | Path, label_column: str = "defects", name: str | None = None, **_: Any) -> None:
        self.path = Path(path)
        self.label_column = label_column
        self.name = name or self.path.stem

    def load(self) -> TabularDataset:
        frame = pd.read_csv(self.path)
        if self.label_column not in frame.columns:
            raise ValueError(f"Label column '{self.label_column}' not in {self.path} (columns: {list(frame.columns)})")
        frame[self.label_column] = to_binary_labels(frame[self.label_column])
        return TabularDataset(frame, self.label_column, self.name, f"csv:{self.path}")


@DataLoaderFactory.register("synthetic")
class SyntheticDefectDataLoader(DataLoader):
    """Generates module-level metrics shaped like the NASA MDP datasets.

    Size and complexity drive both the metrics and (through a noisy logistic
    model) the defect label, so the data is learnable but not trivially so.
    A few missing values and duplicate rows are injected so the cleaning
    stage has real work to do.
    """

    def __init__(self, n_samples: int = 2000, defect_rate: float = 0.2, missing_rate: float = 0.01,
                 duplicate_rate: float = 0.01, seed: int = 42, **_: Any) -> None:
        self.n_samples = n_samples
        self.defect_rate = defect_rate
        self.missing_rate = missing_rate
        self.duplicate_rate = duplicate_rate
        self.seed = seed

    def load(self) -> TabularDataset:
        rng = np.random.default_rng(self.seed)
        n = self.n_samples
        loc = np.round(rng.lognormal(3.4, 0.9, n)).clip(1, 3000)
        style = rng.gamma(2.0, 0.5, n)  # how branchy this team/module writes code
        v_g = 1 + rng.poisson(loc / 12 * style)
        ev_g = 1 + rng.binomial(v_g - 1, 0.3)
        iv_g = 1 + rng.binomial(v_g - 1, 0.6)
        uniq_op = np.round(rng.normal(8 + 2 * np.log1p(loc), 2)).clip(1)
        uniq_opnd = np.round(rng.normal(5 + 0.5 * loc ** 0.8, 3)).clip(1)
        total_op = np.round(loc * rng.uniform(2, 4, n)).clip(1)
        total_opnd = np.round(loc * rng.uniform(1.2, 2.8, n)).clip(1)
        length = total_op + total_opnd
        volume = length * np.log2((uniq_op + uniq_opnd).clip(2))
        difficulty = (uniq_op / 2) * (total_opnd / uniq_opnd)
        effort = difficulty * volume
        comments = rng.binomial(loc.astype(int), rng.uniform(0.02, 0.3, n))
        blanks = rng.binomial(loc.astype(int), 0.1)

        risk = (0.9 * np.log1p(loc) + 0.08 * v_g + 0.25 * ev_g + 0.35 * np.log1p(difficulty)
                - 2.5 * comments / loc + rng.normal(0, 0.9, n))
        intercept = self._calibrate(risk)
        defects = rng.random(n) < 1 / (1 + np.exp(-(risk + intercept)))

        frame = pd.DataFrame({
            "loc": loc, "v(g)": v_g, "ev(g)": ev_g, "iv(g)": iv_g,
            "n": length, "v": volume.round(2), "l": (1 / difficulty).round(4), "d": difficulty.round(2),
            "i": (volume / difficulty).round(2), "e": effort.round(2), "b": (volume / 3000).round(4),
            "t": (effort / 18).round(2), "lOCode": loc, "lOComment": comments, "lOBlank": blanks,
            "uniq_Op": uniq_op, "uniq_Opnd": uniq_opnd, "total_Op": total_op, "total_Opnd": total_opnd,
            "branchCount": 2 * v_g - 1, "defects": defects.astype(int),
        })
        features = frame.columns.drop("defects")
        mask = rng.random((n, len(features))) < self.missing_rate
        frame[features] = frame[features].mask(mask)
        duplicates = frame.sample(frac=self.duplicate_rate, random_state=self.seed)
        frame = pd.concat([frame, duplicates], ignore_index=True)
        return TabularDataset(frame, "defects", "synthetic-nasa-mdp", "synthetic")

    def _calibrate(self, risk: np.ndarray) -> float:
        """Bisection for the intercept that gives the requested defect rate."""
        low, high = -50.0, 50.0
        for _ in range(60):
            mid = (low + high) / 2
            if np.mean(1 / (1 + np.exp(-(risk + mid)))) < self.defect_rate:
                low = mid
            else:
                high = mid
        return (low + high) / 2


@DataLoaderFactory.register("openml")
class OpenMLDataLoader(DataLoader):
    """PROMISE defect datasets from OpenML: kc1, kc2, pc1, cm1, jm1 ...

    Falls back to the synthetic generator when offline (the dataset's
    ``source`` field says so, and a warning is logged).
    """

    def __init__(self, name: str = "kc1", version: int = 1, label_column: str | None = None,
                 fallback: bool = True, seed: int = 42, **kwargs: Any) -> None:
        self.name = name
        self.version = version
        self.label_column = label_column
        self.fallback = fallback
        self.seed = seed
        self.fallback_params = kwargs

    def load(self) -> TabularDataset:
        try:
            from sklearn.datasets import fetch_openml

            bunch = fetch_openml(name=self.name, version=self.version, as_frame=True, parser="auto")
        except Exception as exc:
            if not self.fallback:
                raise
            log.warning("OpenML dataset '%s' unavailable (%s); using synthetic data", self.name, exc)
            dataset = SyntheticDefectDataLoader(seed=self.seed, **self.fallback_params).load()
            dataset.source = f"synthetic (fallback: openml '{self.name}' unavailable)"
            return dataset
        frame = bunch.frame.copy()
        label = self.label_column or bunch.target_names[0]
        frame[label] = to_binary_labels(frame[label])
        return TabularDataset(frame, label, self.name, f"openml:{self.name}")


@DataLoaderFactory.register("programs")
class ProgramLoader(DataLoader):
    def __init__(self, names: Sequence[str] | None = None, max_steps: int = 5_000, **_: Any) -> None:
        self.names = list(names) if names else list(PROGRAMS)
        self.max_steps = max_steps

    def load(self) -> list[ProgramAdapter]:
        return [ProgramAdapter(get_program(name), max_steps=self.max_steps) for name in self.names]
