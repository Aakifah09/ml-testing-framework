"""Strategy pattern (behavioral) - interchangeable test oracles.

An oracle sees an input and the output a (possibly faulty) program version
produced, and must decide PASS or FAIL *without* the expected output:

* ``rule``        - partial correctness properties from the specification
* ``metamorphic`` - re-runs the program on transformed inputs and checks relations
* ``ml``          - a classifier (from ModelFactory) trained on labelled executions
* ``hybrid``      - Ensemble of the above, combined by voting

Every oracle treats a crash or timeout as FAIL (the implicit oracle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from mltest.core.config import ModelManager
from mltest.core.factory import ModelFactory, OracleStrategyFactory
from mltest.models.ensemble import majority_vote
from mltest.programs.adapter import ProgramAdapter
from mltest.programs.encoding import OutputEncoder, encode_inputs, useful_moduli

PASS, FAIL = "PASS", "FAIL"


@dataclass
class Observation:
    """One execution of one program version on one input."""

    program: str
    version: str
    inputs: tuple
    output: Any
    error: str | None
    label: str  # ground truth: FAIL iff behaviour differs from the original program
    adapter: ProgramAdapter = field(repr=False, compare=False)
    expected: Any = None  # the original program's output (never shown to the oracles)


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str = ""


class OracleStrategy(ABC):
    name = "base"

    def __init__(self, seed: int = 42, **_: Any) -> None:
        self.seed = seed

    def train(self, observations: Sequence[Observation]) -> "OracleStrategy":
        return self

    @abstractmethod
    def check(self, observation: Observation) -> Verdict:
        """Decide PASS/FAIL for a single execution."""

    def validate(self, observations: Sequence[Observation]) -> list[Verdict]:
        return [self.check(o) for o in observations]


@OracleStrategyFactory.register("rule")
class RuleBasedOracle(OracleStrategy):
    name = "rule"

    def check(self, obs: Observation) -> Verdict:
        if obs.error is not None:
            return Verdict(FAIL, f"crash: {obs.error}")
        for rule in obs.adapter.spec.rules:
            try:
                holds = bool(rule.check(obs.inputs, obs.output))
            except Exception:
                holds = False
            if not holds:
                return Verdict(FAIL, f"rule violated: {rule.name}")
        return Verdict(PASS)


@OracleStrategyFactory.register("metamorphic")
class MetamorphicOracle(OracleStrategy):
    name = "metamorphic"

    def check(self, obs: Observation) -> Verdict:
        if obs.error is not None:
            return Verdict(FAIL, f"crash: {obs.error}")
        for relation in obs.adapter.spec.relations:
            if not relation.applies(obs.inputs):
                continue
            follow_up = obs.adapter.run(relation.transform(obs.inputs))
            if not follow_up.ok:
                return Verdict(FAIL, f"follow-up crashed ({relation.name}): {follow_up.error}")
            try:
                holds = bool(relation.relation(obs.output, follow_up.output, obs.inputs))
            except Exception:
                holds = False
            if not holds:
                return Verdict(FAIL, f"relation violated: {relation.name}")
        return Verdict(PASS)


@OracleStrategyFactory.register("ml")
class MLOracle(OracleStrategy):
    """Learns PASS/FAIL from (input, output) features; one model per program."""

    name = "ml"

    def __init__(self, seed: int = 42, model_type: str = "random_forest", **_: Any) -> None:
        super().__init__(seed)
        self.model_type = model_type
        self._models: dict[str, Any] = {}
        self._encoders: dict[str, OutputEncoder] = {}
        self._constant: dict[str, str] = {}

    def _features(self, obs: Observation) -> list[float]:
        moduli = useful_moduli(obs.adapter.constants)
        return (encode_inputs(obs.adapter.params, obs.inputs, moduli)
                + self._encoders[obs.program].transform(obs.inputs, obs.output, obs.error))

    def train(self, observations: Sequence[Observation]) -> "MLOracle":
        by_program: dict[str, list[Observation]] = defaultdict(list)
        for obs in observations:
            by_program[obs.program].append(obs)
        for program, group in by_program.items():
            self._encoders[program] = OutputEncoder().fit(o.output for o in group)
            labels = np.array([o.label for o in group])
            if len(set(labels)) < 2:
                self._constant[program] = labels[0]
                continue
            model = ModelFactory.create_model(self.model_type, seed=self.seed)
            model.fit([self._features(o) for o in group], labels)
            self._models[program] = model
            ModelManager().register(f"test_oracle/{self.model_type}/{program}", model)
        return self

    def validate(self, observations: Sequence[Observation]) -> list[Verdict]:
        verdicts: list[Verdict | None] = [None] * len(observations)
        pending: dict[str, list[int]] = defaultdict(list)
        for i, obs in enumerate(observations):
            if obs.error is not None:
                verdicts[i] = Verdict(FAIL, f"crash: {obs.error}")
            elif obs.program in self._models:
                pending[obs.program].append(i)
            else:
                verdicts[i] = Verdict(self._constant.get(obs.program, PASS), "no model for program")
        for program, indices in pending.items():
            model = self._models[program]
            predictions = model.predict([self._features(observations[i]) for i in indices])
            confidence = model.predict_proba([self._features(observations[i]) for i in indices]).max(axis=1)
            for i, label, conf in zip(indices, predictions, confidence):
                verdicts[i] = Verdict(str(label), f"model confidence {conf:.2f}")
        return verdicts  # type: ignore[return-value]

    def check(self, obs: Observation) -> Verdict:
        return self.validate([obs])[0]


@OracleStrategyFactory.register("hybrid")
class HybridOracle(OracleStrategy):
    """Ensemble of oracles: members vote, by majority or 'any FAIL'."""

    name = "hybrid"

    def __init__(self, seed: int = 42, members: Sequence[str] = ("rule", "metamorphic", "ml"),
                 voting: str = "majority", weights: Sequence[float] | None = None,
                 member_params: dict[str, dict] | None = None, **_: Any) -> None:
        super().__init__(seed)
        if voting not in ("majority", "any"):
            raise ValueError("voting must be 'majority' or 'any'")
        member_params = member_params or {}
        self.members = [OracleStrategyFactory.create(m, seed=seed, **member_params.get(m, {})) for m in members]
        self.voting = voting
        self.weights = list(weights) if weights is not None else None

    def train(self, observations: Sequence[Observation]) -> "HybridOracle":
        for member in self.members:
            member.train(observations)
        return self

    def validate(self, observations: Sequence[Observation]) -> list[Verdict]:
        per_member = [member.validate(observations) for member in self.members]
        verdicts = []
        for votes in zip(*per_member):
            statuses = [v.status for v in votes]
            if self.voting == "any":
                status = FAIL if FAIL in statuses else PASS
            else:
                # FAIL listed first so a tie is resolved conservatively as FAIL
                status = majority_vote([FAIL, *statuses], [0.0, *(self.weights or [1.0] * len(statuses))])
            reason = ", ".join(f"{m.name}={s}" for m, s in zip(self.members, statuses))
            verdicts.append(Verdict(status, reason))
        return verdicts

    def check(self, obs: Observation) -> Verdict:
        return self.validate([obs])[0]
