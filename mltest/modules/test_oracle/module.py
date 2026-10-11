"""Test oracle: program/output -> features -> oracle -> PASS/FAIL decision -> evaluation.

Ground truth comes from mutation: each mutant is a faulty version, and an
execution is labelled FAIL when it behaves differently from the original.
Mutants (and inputs) are split so that learned oracles are evaluated on
faults and inputs they never saw during training.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from mltest.core.evaluator import Evaluator, Stopwatch
from mltest.core.factory import DataLoaderFactory, ModuleFactory, OracleStrategyFactory, TestStrategyFactory
from mltest.core.pipeline import PipelineContext
from mltest.modules.base import TestingModule
from mltest.modules.test_oracle.strategies import FAIL, PASS, Observation
from mltest.programs.mutation import mutant_adapters
from mltest.reporting import write_csv


@ModuleFactory.register("test_oracle")
class TestOracleModule(TestingModule):
    __test__ = False
    name = "test_oracle"
    summary_columns = ("oracle", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
                       "faults_detected", "check_time")

    def load(self, ctx: PipelineContext) -> PipelineContext:
        ctx["programs"] = DataLoaderFactory.create("programs", names=self.settings.get("programs")).load()
        self.say(f"loaded {len(ctx['programs'])} programs under test")
        return ctx

    def _inputs_for(self, program, rng: random.Random) -> list[tuple]:
        """Diverse inputs: a search-generated suite topped up with random inputs."""
        n = int(self.settings.get("n_inputs", 60))
        strategy = TestStrategyFactory.create(self.settings.get("input_strategy", "search"), seed=self.seed)
        inputs = strategy.generate(program, n)[:n]
        while len(inputs) < n:
            inputs.append(program.spec.sample_inputs(rng))
        return inputs

    def preprocess(self, ctx: PipelineContext) -> PipelineContext:
        rng = random.Random(self.seed)
        fraction = float(self.settings.get("train_fraction", 0.5))
        train, test = [], []
        for program in ctx["programs"]:
            inputs = self._inputs_for(program, rng)
            rng.shuffle(inputs)
            pairs = mutant_adapters(program)
            ctx.setdefault("bug_descriptions", {}).update({m.id: m.description for m, _ in pairs})
            mutants = [adapter for _, adapter in pairs]
            rng.shuffle(mutants)
            cut_in, cut_mut = int(len(inputs) * fraction), int(len(mutants) * fraction)
            reference = {repr(x): program.run(x) for x in inputs}
            # the original program appears on both sides so both labels are represented
            for versions, xs, bucket in ((mutants[:cut_mut], inputs[:cut_in], train),
                                         (mutants[cut_mut:], inputs[cut_in:], test)):
                for version in [program, *versions]:
                    for x in xs:
                        result = version.run(x)
                        correct = reference[repr(x)]
                        label = PASS if result.signature() == correct.signature() else FAIL
                        bucket.append(Observation(program.name, version.version, x, result.output,
                                                  result.error, label, version, correct.output))
        ctx["train"], ctx["test"] = train, test
        for split, obs in (("train", train), ("test", test)):
            counts = Counter(o.label for o in obs)
            self.say(f"{split}: {len(obs)} executions ({counts[FAIL]} FAIL / {counts[PASS]} PASS)")
        return ctx

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        params = self.settings.get("strategy_params", {})
        runs = {}
        for name in self.settings.get("strategies", ["rule", "metamorphic", "ml", "hybrid"]):
            oracle = OracleStrategyFactory.create(name, seed=self.seed, **params.get(name, {}))
            with Stopwatch() as train_time:
                oracle.train(ctx["train"])
            with Stopwatch() as check_time:
                verdicts = oracle.validate(ctx["test"])
            runs[name] = {"verdicts": verdicts, "train_time": train_time.seconds, "check_time": check_time.seconds}
            fails = sum(v.status == FAIL for v in verdicts)
            self.say(f"{name:<12} flagged {fails}/{len(verdicts)} executions as FAIL")
        ctx["runs"] = runs
        self._show_verdict_examples(ctx)
        return ctx

    def _show_verdict_examples(self, ctx: PipelineContext) -> None:
        """Print a few real executions with every oracle's verdict, to show what is being judged."""
        limit = int(self.settings.get("show_examples", 3))
        if not limit:
            return
        runs, test = ctx["runs"], ctx["test"]
        for program in dict.fromkeys(o.program for o in test):
            func = next(o.adapter.spec.func.__name__ for o in test if o.program == program)
            failing, seen = [], set()
            for i, obs in enumerate(test):  # distinct buggy versions, preferring visible wrong outputs
                if obs.program == program and obs.label == FAIL and obs.error is None and obs.version not in seen:
                    seen.add(obs.version)
                    failing.append(i)
            correct = [i for i, o in enumerate(test) if o.program == program and o.version == "original"][:1]
            self.say(f"{program}: example verdicts (oracles never see the correct output)")
            for i in failing[: max(limit - 1, 1)] + correct:
                obs = test[i]
                args = ", ".join(repr(x) for x in obs.inputs)
                verdicts = "  ".join(f"{name}={run['verdicts'][i].status}" for name, run in runs.items())
                if obs.version == "original":
                    self.say(f"    correct program: {func}({args}) -> {obs.output!r}")
                else:
                    bug = ctx["bug_descriptions"].get(obs.version, obs.version)
                    self.say(f"    bug at {bug}: {func}({args}) -> {obs.output!r} (correct: {obs.expected!r})")
                reason = next((run["verdicts"][i].reason for run in runs.values()
                               if run["verdicts"][i].status == FAIL and "confidence" not in run["verdicts"][i].reason), "")
                self.say(f"        truth={obs.label}  {verdicts}" + (f"  [{reason}]" if reason else ""))

    def _scores(self, observations, verdicts) -> dict:
        y_true = [int(o.label == FAIL) for o in observations]
        y_pred = [int(v.status == FAIL) for v in verdicts]
        faulty = defaultdict(bool)   # version -> has at least one failing execution
        caught = defaultdict(bool)   # version -> oracle flagged one of those
        for obs, truth, pred in zip(observations, y_true, y_pred):
            key = (obs.program, obs.version)
            faulty[key] |= bool(truth)
            caught[key] |= bool(truth and pred)
        detectable = [k for k, v in faulty.items() if v]
        return {
            **Evaluator.classification(y_true, y_pred),
            "false_positive_rate": Evaluator.false_positive_rate(y_true, y_pred),
            "false_negative_rate": Evaluator.false_negative_rate(y_true, y_pred),
            "faults_detected": sum(caught[k] for k in detectable) / len(detectable) if detectable else 0.0,
        }

    def evaluate(self, ctx: PipelineContext) -> PipelineContext:
        records, per_program = [], []
        for name, run in ctx["runs"].items():
            records.append({"name": name, "oracle": name, **self._scores(ctx["test"], run["verdicts"]),
                            "train_time": run["train_time"], "check_time": run["check_time"]})
            groups = defaultdict(list)
            for obs, verdict in zip(ctx["test"], run["verdicts"]):
                groups[obs.program].append((obs, verdict))
            for program, pairs in groups.items():
                obs, verdicts = zip(*pairs)
                per_program.append({"name": f"{name}/{program}", "oracle": name, "program": program,
                                    **self._scores(obs, verdicts)})
        ctx["metrics"], ctx["per_program"] = records, per_program
        self.emit("metrics", records=records)
        best = max(records, key=lambda r: r["f1"])
        self.say(f"best oracle by F1: {best['oracle']} (F1={best['f1']:.3f}, "
                 f"FPR={best['false_positive_rate']:.3f}, faults detected={best['faults_detected']:.1%})")
        return ctx

    def report(self, ctx: PipelineContext) -> PipelineContext:
        out = self.output_dir
        rows = []
        for i, obs in enumerate(ctx["test"]):
            row = {"program": obs.program, "version": obs.version, "inputs": repr(obs.inputs),
                   "output": repr(obs.output), "correct_output": repr(obs.expected), "error": obs.error or "",
                   "bug": ctx["bug_descriptions"].get(obs.version, ""), "expected": obs.label}
            for name, run in ctx["runs"].items():
                verdict = run["verdicts"][i]
                row[name] = verdict.status
                row[f"{name}_reason"] = verdict.reason
            rows.append(row)
        artifacts = [write_csv(out / "oracle_results.csv", rows), write_csv(out / "metrics.csv", ctx["metrics"]),
                     write_csv(out / "metrics_per_program.csv", ctx["per_program"])]
        ctx["artifacts"] = [str(a) for a in artifacts]
        self.say(f"wrote {len(rows)} PASS/FAIL decisions to {out}")
        return ctx
