"""Strategy pattern (behavioral) - interchangeable test-generation algorithms.

* ``random`` - uniform sampling of the input domain (baseline)
* ``search`` - coverage-guided evolutionary search with constant seeding
* ``ml``     - a classifier learns input -> execution path and steers
               generation toward rare paths and uncertain regions

Every strategy spends at most ``budget`` program executions. Random keeps all
of its inputs; search and ML keep only inputs that reached new branches,
lines or output categories (a minimised suite).
"""
from __future__ import annotations

import copy
import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np

from mltest.core.factory import ModelFactory, TestStrategyFactory
from mltest.programs.adapter import ExecutionResult, ProgramAdapter
from mltest.programs.catalog import IntListParam, IntParam
from mltest.programs.encoding import encode_inputs, useful_moduli


class TestGenerationStrategy(ABC):
    __test__ = False  # not a pytest test class
    name = "base"

    def __init__(self, seed: int = 42, **params: Any) -> None:
        self.seed = seed
        self.params = params

    @abstractmethod
    def generate(self, program: ProgramAdapter, budget: int) -> list[tuple]:
        """Return a list of input tuples for ``program``."""


def output_category(result: ExecutionResult) -> tuple:
    """Coarse output class used as a diversity objective."""
    if not result.ok:
        return ("error", result.error.split(":", 1)[0])
    out = result.output
    if isinstance(out, (str, bool)) or out is None:
        return ("value", out)
    if isinstance(out, int):
        return ("int", out if abs(out) <= 31 else ("big", out > 0))
    if isinstance(out, list):
        return ("list", min(len(out), 3))
    return ("type", type(out).__name__)


class CoverageArchive:
    """Keeps every test that reached a new branch, line, boundary or output category.

    Boundaries are comparison distances (``x - y`` in {<-1, -1, 0, +1, >+1}); they
    reward tests that sit on and just beside each decision boundary, which is
    what kills off-by-one and relational-operator faults.
    """

    def __init__(self, program: ProgramAdapter) -> None:
        self.program = program
        self.branches: set[tuple[int, int]] = set()
        self.lines: set[int] = set()
        self.boundaries: set[tuple[int, int]] = set()
        self.outputs: set[tuple] = set()
        self.tests: list[ExecutionResult] = []

    def offer(self, result: ExecutionResult) -> bool:
        branches = self.program.covered_branches([result]) - self.branches
        lines = result.lines - self.lines
        boundaries = result.boundaries - self.boundaries
        category = output_category(result)
        if branches or lines or boundaries or category not in self.outputs:
            self.branches |= branches
            self.lines |= lines
            self.boundaries |= boundaries
            self.outputs.add(category)
            self.tests.append(result)
            return True
        return False

    def pick_parent(self, rng: random.Random) -> tuple:
        """Favour tests that execute deeper into the program (more lines)."""
        weights = [len(t.lines) ** 2 + 1 for t in self.tests]
        return rng.choices(self.tests, weights=weights)[0].inputs


class InputMutator:
    """Small, type-aware changes to an input tuple (the search operators)."""

    def __init__(self, program: ProgramAdapter, rng: random.Random) -> None:
        self.params = program.params
        self.constants = [c for c in program.constants if c != 0] or [1]
        self.rng = rng

    def random_inputs(self) -> tuple:
        return tuple(p.sample(self.rng) for p in self.params)

    def mutate(self, inputs: tuple, rounds: int | None = None) -> tuple:
        values = list(copy.deepcopy(inputs))
        for _ in range(rounds or self.rng.randint(1, 2)):
            i = self.rng.randrange(len(values))
            param = self.params[i]
            if isinstance(param, IntParam):
                values[i] = self._mutate_int(i, values)
            elif isinstance(param, IntListParam):
                values[i] = self._mutate_list(values[i], param)
        return tuple(values)

    def _mutate_int(self, i: int, values: list) -> int:
        p, v, rng = self.params[i], values[i], self.rng
        others = [values[j] for j, q in enumerate(self.params) if j != i and isinstance(q, IntParam)]
        roll = rng.random()
        if roll < 0.35:
            v += rng.choice((-1, 1)) * rng.randint(1, 3)
        elif roll < 0.5 and others:
            v = rng.choice(others)  # make two parameters equal
        elif roll < 0.62:
            v = rng.choice((p.low, p.high, 0, 1, -1))
        elif roll < 0.85:
            v = self._from_constant(p)  # values mined from the code itself
        else:
            v = p.sample(rng)
        return max(p.low, min(p.high, v))

    def _from_constant(self, p: IntParam) -> int:
        c = self.rng.choice(self.constants)
        if self.rng.random() < 0.5:
            return c + self.rng.choice((-1, 0, 1))
        lo, hi = math.ceil(p.low / c), math.floor(p.high / c)
        if c < 0:
            lo, hi = math.ceil(p.high / c), math.floor(p.low / c)
        return c * self.rng.randint(lo, hi) if lo <= hi else c

    def _mutate_list(self, xs: list[int], p: IntListParam) -> list[int]:
        xs, rng = list(xs), self.rng
        op = rng.choice(("insert", "delete", "change", "swap", "duplicate", "reverse"))
        if op == "insert" and len(xs) < p.max_len:
            xs.insert(rng.randint(0, len(xs)), rng.randint(p.low, p.high))
        elif op == "delete" and len(xs) > p.min_len:
            xs.pop(rng.randrange(len(xs)))
        elif op == "change" and xs:
            xs[rng.randrange(len(xs))] = rng.randint(p.low, p.high)
        elif op == "swap" and len(xs) > 1:
            a, b = rng.sample(range(len(xs)), 2)
            xs[a], xs[b] = xs[b], xs[a]
        elif op == "duplicate" and xs and len(xs) < p.max_len:
            xs.insert(rng.randint(0, len(xs)), rng.choice(xs))
        elif op == "reverse":
            xs.reverse()
        return xs


@TestStrategyFactory.register("random")
class RandomTestStrategy(TestGenerationStrategy):
    name = "random"

    def generate(self, program: ProgramAdapter, budget: int) -> list[tuple]:
        rng = random.Random(self.seed)
        return [program.spec.sample_inputs(rng) for _ in range(budget)]


@TestStrategyFactory.register("search")
class SearchTestStrategy(TestGenerationStrategy):
    name = "search"

    def generate(self, program: ProgramAdapter, budget: int) -> list[tuple]:
        rng = random.Random(self.seed)
        mutator = InputMutator(program, rng)
        archive = CoverageArchive(program)
        initial = max(1, budget // 10)
        for step in range(budget):
            if step < initial or not archive.tests or rng.random() < 0.1:
                candidate = mutator.random_inputs()
            else:
                candidate = mutator.mutate(archive.pick_parent(rng))
            archive.offer(program.run(candidate))
        return [r.inputs for r in archive.tests]


@TestStrategyFactory.register("ml")
class MLTestStrategy(TestGenerationStrategy):
    """Active-learning generation with a path classifier created by ModelFactory."""

    name = "ml"

    def __init__(self, seed: int = 42, model_type: str = "decision_tree", warmup_fraction: float = 0.25,
                 pool_size: int = 60, batch_size: int = 5, **params: Any) -> None:
        super().__init__(seed, **params)
        self.model_type = model_type
        self.warmup_fraction = warmup_fraction
        self.pool_size = pool_size
        self.batch_size = batch_size

    def generate(self, program: ProgramAdapter, budget: int) -> list[tuple]:
        rng = random.Random(self.seed)
        mutator = InputMutator(program, rng)
        archive = CoverageArchive(program)
        moduli = useful_moduli(program.constants)
        features: list[list[float]] = []
        paths: list[int] = []
        path_ids: dict[frozenset, int] = {}

        def execute(inputs: tuple) -> None:
            result = program.run(inputs)
            archive.offer(result)
            features.append(encode_inputs(program.params, inputs, moduli))
            key = frozenset(program.covered_branches([result])) | {output_category(result)}
            paths.append(path_ids.setdefault(key, len(path_ids)))

        warmup = min(budget, max(5, int(budget * self.warmup_fraction)))
        for _ in range(warmup):
            execute(mutator.random_inputs())

        spent = warmup
        while spent < budget:
            pool = [
                mutator.mutate(archive.pick_parent(rng)) if archive.tests and rng.random() < 0.7
                else mutator.random_inputs()
                for _ in range(self.pool_size)
            ]
            order = list(range(len(pool)))
            if len(set(paths)) > 1:
                model = ModelFactory.create_model(self.model_type, seed=self.seed)
                model.fit(features, paths)
                proba = model.predict_proba([encode_inputs(program.params, x, moduli) for x in pool])
                counts = Counter(paths)
                predicted = model.classes_[np.argmax(proba, axis=1)]
                rarity = np.array([1.0 / counts[p] for p in predicted])
                uncertainty = 1.0 - proba.max(axis=1)
                order = list(np.argsort(-(rarity + uncertainty), kind="stable"))
            for index in order[: min(self.batch_size, budget - spent)]:
                execute(pool[index])
                spent += 1
        return [r.inputs for r in archive.tests]
