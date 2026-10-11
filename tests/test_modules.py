"""End-to-end runs of the three testing modules (small budgets)."""
import json

from mltest.app import TestingFramework
from mltest.core.config import ModelManager
from mltest.core.factory import TestStrategyFactory
from mltest.core.observer import ReportObserver
from mltest.programs.adapter import ProgramAdapter
from mltest.programs.catalog import TRIANGLE


def _run(config, module):
    report = ReportObserver()
    ctx = TestingFramework(config, observers=[report]).run([module])[module]
    return ctx, report


def test_every_generation_strategy_reaches_full_branch_coverage_on_triangle():
    program = ProgramAdapter(TRIANGLE)
    for name in ("search", "ml"):
        inputs = TestStrategyFactory.create(name, seed=42).generate(program, 200)
        assert program.branch_coverage([program.run(x) for x in inputs]) == 1.0, name


def test_test_generation_module(config):
    ctx, report = _run(config, "test_generation")
    assert {r["strategy"] for r in ctx["metrics"]} == {"random", "search", "ml"}
    for record in ctx["metrics"]:
        assert 0 <= record["mutation_score"] <= 1
        assert record["tests"] > 0
    out = config.run_dir / "test_generation"
    assert (out / "test_cases.json").exists()
    assert (out / "generated_tests" / "test_triangle_search.py").exists()
    assert report.of_type("pipeline_finished")
    assert (config.run_dir / "summary.md").exists()
    assert json.loads((config.run_dir / "events.json").read_text())


def test_defect_prediction_module(config):
    ctx, _ = _run(config, "defect_prediction")
    names = {r["model"] for r in ctx["metrics"]}
    assert names == {"decision_tree", "logistic_regression", "ensemble_soft"}
    assert all(r["roc_auc"] > 0.6 for r in ctx["metrics"])
    assert "defect_prediction/ensemble_soft" in ModelManager().keys()
    assert (config.run_dir / "defect_prediction" / "predictions.csv").exists()


def test_defect_prediction_from_csv(config, tmp_path):
    from mltest.data.loaders import SyntheticDefectDataLoader

    path = tmp_path / "kc_like.csv"
    frame = SyntheticDefectDataLoader(n_samples=300).load().frame
    frame["defects"] = frame["defects"].map({1: "true", 0: "false"})
    frame.to_csv(path, index=False)
    config.update({"defect_prediction": {"data_source": "csv", "loader_params": {"path": str(path)}}})
    ctx, _ = _run(config, "defect_prediction")
    assert ctx["dataset"].source.startswith("csv:")
    assert 0 < ctx["dataset"].defect_rate < 1


def test_test_oracle_module(config):
    ctx, _ = _run(config, "test_oracle")
    by_name = {r["oracle"]: r for r in ctx["metrics"]}
    assert set(by_name) == {"rule", "metamorphic", "ml", "hybrid"}
    assert by_name["rule"]["false_positive_rate"] == 0.0  # rules hold for the correct program
    assert by_name["rule"]["recall"] > 0.5
    assert all(0 <= r["recall"] <= 1 and r["check_time"] >= 0 for r in by_name.values())
    assert any(k.startswith("test_oracle/") for k in ModelManager().keys())  # ML oracle models registered
    assert (config.run_dir / "test_oracle" / "oracle_results.csv").exists()
