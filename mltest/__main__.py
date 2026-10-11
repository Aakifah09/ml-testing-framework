"""Command line interface: ``python -m mltest --help``."""
from __future__ import annotations

import argparse
import sys

from mltest.core.config import AppConfig
from mltest.core.factory import (DataLoaderFactory, DecoratorFactory, ModelFactory, ModuleFactory,
                                 ObserverFactory, OracleStrategyFactory, TestStrategyFactory)

MODULES = ("test_generation", "defect_prediction", "test_oracle")


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mltest", description="ML-based automated software testing framework")
    parser.add_argument("--config", help="JSON file overriding the default configuration")
    parser.add_argument("--module", "-m", action="append", choices=(*MODULES, "all"),
                        help="module to run (repeatable); default: all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", help="results folder (a run_<timestamp> subfolder is created)")
    parser.add_argument("--budget", type=int, help="test generation: executions per program and strategy")
    parser.add_argument("--strategies", type=_csv, help="test generation strategies, e.g. random,search,ml")
    parser.add_argument("--data-source", choices=("synthetic", "csv", "openml"), help="defect prediction data")
    parser.add_argument("--csv", help="defect dataset CSV (implies --data-source csv)")
    parser.add_argument("--label", help="label column of the CSV/OpenML dataset")
    parser.add_argument("--openml", help="OpenML dataset name, e.g. kc1, pc1 (implies --data-source openml)")
    parser.add_argument("--models", type=_csv, help="defect models, e.g. svm,random_forest,torch_mlp")
    parser.add_argument("--no-ensemble", action="store_true", help="skip the ensemble predictor")
    parser.add_argument("--oracles", type=_csv, help="oracle strategies, e.g. rule,metamorphic,ml,hybrid")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every pipeline event")
    parser.add_argument("--examples", type=int, help="examples printed per program/model (0 = none, default 3)")
    parser.add_argument("--list", action="store_true", help="list everything the factories can create")
    parser.add_argument("--programs", action="store_true",
                        help="show the programs under test: code, input ranges, rules, metamorphic relations")
    return parser.parse_args(argv)


def configure(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    if args.config:
        config.load(args.config)
    if args.module and "all" not in args.module:
        config.set("modules", args.module)
    if args.seed is not None:
        config.set("seed", args.seed)
    if args.output:
        config.set("output_dir", args.output)
    if args.budget:
        config.set("test_generation.budget", args.budget)
    if args.strategies:
        config.set("test_generation.strategies", args.strategies)
    if args.csv:
        config.set("defect_prediction.data_source", "csv")
        config.set("defect_prediction.loader_params", {"path": args.csv})
    if args.openml:
        config.set("defect_prediction.data_source", "openml")
        config.set("defect_prediction.loader_params", {"name": args.openml})
    if args.data_source:
        config.set("defect_prediction.data_source", args.data_source)
    if args.label:
        config.set("defect_prediction.loader_params.label_column", args.label)
    if args.models:
        config.set("defect_prediction.models", args.models)
    if args.no_ensemble:
        config.set("defect_prediction.ensemble.enabled", False)
    if args.oracles:
        config.set("test_oracle.strategies", args.oracles)
    if args.examples is not None:
        for module in MODULES:
            config.set(f"{module}.show_examples", args.examples)
    return config


def show_programs() -> None:
    import inspect

    from mltest.programs.adapter import ProgramAdapter
    from mltest.programs.catalog import PROGRAMS, IntParam
    from mltest.programs.mutation import generate_mutants

    for spec in PROGRAMS.values():
        program = ProgramAdapter(spec)
        print("=" * 78)
        print(f"{spec.name}: {spec.description}")
        print("-" * 78)
        print(inspect.getsource(spec.func).rstrip())
        print("-" * 78)
        domains = [f"{p.name} in [{p.low}, {p.high}]" if isinstance(p, IntParam)
                   else f"{p.name} = list of {p.min_len}-{p.max_len} ints in [{p.low}, {p.high}]" for p in spec.params]
        print("Inputs:          " + "; ".join(domains))
        print(f"Code facts:      {len(program.static.executable_lines)} lines, "
              f"{len(program.decision_lines)} decisions, {len(generate_mutants(spec.func))} mutants (planted bugs)")
        print("Oracle rules:    " + ", ".join(r.name for r in spec.rules))
        print("Metamorphic:     " + ", ".join(r.name for r in spec.relations))
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for factory in (ModuleFactory, DataLoaderFactory, ModelFactory, TestStrategyFactory,
                        OracleStrategyFactory, DecoratorFactory, ObserverFactory):
            print(f"{factory.__name__:<22} {', '.join(factory.available())}")
        return 0
    if args.programs:
        show_programs()
        return 0

    from mltest.app import TestingFramework
    from mltest.core.observer import ConsoleObserver

    config = configure(args)
    framework = TestingFramework(config)
    if args.verbose:
        for observer in framework.observers:
            if isinstance(observer, ConsoleObserver):
                observer.verbose = True
    framework.run()
    print(f"\nResults written to {config.run_dir} (summary.md, events.json, run.log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
