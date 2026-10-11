"""Framework entry point: builds one Pipeline per selected testing module and runs it.

User / Configuration -> AppConfig (Singleton) -> ModuleFactory creates the module
-> its five stages are wrapped by Decorators -> Observers attach to the Pipeline
-> Pipeline runs Load -> Preprocess -> Execute -> Evaluate -> Report.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from mltest.core.config import AppConfig
from mltest.core.decorators import decorate
from mltest.core.factory import ModuleFactory, ObserverFactory
from mltest.core.observer import Observer
from mltest.core.pipeline import Pipeline, PipelineContext
from mltest.modules.base import TestingModule
from mltest.reporting import markdown_table


def build_pipeline(module: TestingModule, decorators: Iterable[str] = ()) -> Pipeline:
    pipeline = Pipeline(module.name)
    module.bind(pipeline)
    for stage in module.stages():
        pipeline.add_stage(decorate(stage, decorators))
    return pipeline


class TestingFramework:
    __test__ = False  # not a pytest test class

    def __init__(self, config: AppConfig | None = None, observers: Iterable[Observer] | None = None) -> None:
        self.config = config or AppConfig()
        self.observers = list(observers) if observers is not None else [
            ObserverFactory.create(name) for name in self.config.get("observers", [])
        ]

    def run(self, modules: Iterable[str] | None = None) -> dict[str, PipelineContext]:
        self.config.seed_everything()
        run_dir = self.config.run_dir
        handler = self._attach_log_file(run_dir / "run.log")
        (run_dir / "config.json").write_text(json.dumps(self.config.as_dict(), indent=2), encoding="utf-8")
        results: dict[str, PipelineContext] = {}
        try:
            for name in modules or self.config.get("modules"):
                module = ModuleFactory.create(name, config=self.config)
                pipeline = build_pipeline(module, self.config.get("decorators", []))
                for observer in self.observers:
                    pipeline.attach(observer)
                results[name] = pipeline.run(PipelineContext(module=name))
                results[name]["summary_columns"] = module.summary_columns
        finally:
            for observer in self.observers:
                if hasattr(observer, "flush"):
                    observer.flush(run_dir / "events.json")
            logging.getLogger("mltest").removeHandler(handler)
            handler.close()
        self.write_summary(run_dir, results)
        return results

    @staticmethod
    def _attach_log_file(path: Path) -> logging.Handler:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logger = logging.getLogger("mltest")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        return handler

    def write_summary(self, run_dir: Path, results: dict[str, PipelineContext]) -> Path:
        titles = {"test_generation": "1. Test Case Generation", "defect_prediction": "2. Defect Prediction",
                  "test_oracle": "3. Test Oracle"}
        parts = [f"# mltest run summary\n\nSeed: {self.config.seed}  \nOutput folder: `{run_dir}`\n"]
        for name, ctx in results.items():
            parts.append(f"## {titles.get(name, name)}\n")
            if name == "test_generation" and "strategy_summary" in ctx:
                parts.append("Average per strategy:\n\n" + markdown_table(
                    ctx["strategy_summary"],
                    ["strategy", "tests", "line_coverage", "branch_coverage", "boundary_coverage",
                     "mutation_score", "generation_time"],
                ) + "\n\nPer program:\n")
            if name == "defect_prediction" and "dataset" in ctx:
                parts.append(f"Dataset: {ctx['dataset'].name} ({ctx['dataset'].source})\n")
            parts.append(markdown_table(ctx.get("metrics", []), ctx["summary_columns"]) + "\n")
            timings = ", ".join(f"{k} {v:.2f}s" for k, v in ctx.get("timings", {}).items())
            parts.append(f"Stage timings: {timings}\n")
        path = run_dir / "summary.md"
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
