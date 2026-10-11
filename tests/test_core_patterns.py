"""Singleton, Factory, Pipeline, Decorator and Observer."""
import logging

import pytest

from mltest.core.config import AppConfig, ModelManager
from mltest.core.decorators import LoggingDecorator, TimingDecorator, decorate
from mltest.core.factory import Factory, ModelFactory
from mltest.core.observer import Observer, ReportObserver
from mltest.core.pipeline import FunctionStage, Pipeline, PipelineContext


# -- Singleton ---------------------------------------------------------------
def test_app_config_is_a_singleton(config):
    assert AppConfig() is config
    config.set("seed", 7)
    assert AppConfig().seed == 7


def test_model_manager_is_a_singleton(config):
    ModelManager().register("m", object())
    assert "m" in ModelManager().keys()


def test_config_deep_merge_and_dotted_access(config):
    config.update({"defect_prediction": {"test_size": 0.2}})
    assert config.get("defect_prediction.test_size") == 0.2
    assert config.get("defect_prediction.models") == ["decision_tree", "logistic_regression"]  # kept
    assert config.get("does.not.exist", "fallback") == "fallback"


def test_section_is_a_copy(config):
    config.section("test_generation")["budget"] = 1
    assert config.get("test_generation.budget") == 60


# -- Factory -----------------------------------------------------------------
def test_factory_unknown_key_lists_available():
    with pytest.raises(ValueError, match="Available: .*random_forest"):
        ModelFactory.create_model("no_such_model")


def test_factory_registration_is_per_subclass():
    class ShapeFactory(Factory):
        kind = "shape"

    @ShapeFactory.register("square")
    class Square:
        def __init__(self, side=1):
            self.side = side

    assert ShapeFactory.create("square", side=3).side == 3
    assert ShapeFactory.available() == ["square"]
    assert "square" not in ModelFactory.available()


# -- Pipeline + Observer -----------------------------------------------------
class Recorder(Observer):
    def __init__(self):
        self.types = []

    def update(self, event):
        self.types.append((event.type, event.payload.get("stage")))


def test_pipeline_runs_stages_in_order_and_notifies():
    order = []
    pipeline = Pipeline("demo", [FunctionStage(n, lambda ctx, n=n: order.append(n)) for n in ("a", "b", "c")])
    recorder = Recorder()
    pipeline.attach(recorder)
    pipeline.run()
    assert order == ["a", "b", "c"]
    assert recorder.types[0] == ("pipeline_started", None)
    assert ("stage_finished", "b") in recorder.types
    assert recorder.types[-1] == ("pipeline_finished", None)


def test_pipeline_reports_failures():
    def boom(ctx):
        raise RuntimeError("nope")

    pipeline = Pipeline("demo", [FunctionStage("bad", boom)])
    report = ReportObserver()
    pipeline.attach(report)
    with pytest.raises(RuntimeError):
        pipeline.run()
    assert report.of_type("stage_failed")[0]["payload"]["stage"] == "bad"


# -- Decorator ---------------------------------------------------------------
def test_timing_decorator_records_time():
    stage = TimingDecorator(FunctionStage("work", lambda ctx: None))
    ctx = stage.run(PipelineContext())
    assert ctx["timings"]["work"] >= 0
    assert stage.name == "work"


def test_logging_decorator_logs(caplog):
    stage = LoggingDecorator(FunctionStage("work", lambda ctx: None))
    with caplog.at_level(logging.INFO, logger="mltest.pipeline"):
        stage.run(PipelineContext(module="m"))
    assert "entering stage 'work'" in caplog.text


def test_decorators_stack():
    stage = decorate(FunctionStage("s", lambda ctx: None), ["logging", "timing"])
    assert isinstance(stage, TimingDecorator) and isinstance(stage.inner, LoggingDecorator)
