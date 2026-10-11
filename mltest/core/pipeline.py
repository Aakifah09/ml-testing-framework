"""Pipeline pattern (workflow) - Load -> Preprocess -> Execute -> Evaluate -> Report.

A Pipeline runs an ordered list of Stages, passing one shared context from
stage to stage. It is also the Subject that observers subscribe to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Callable, Iterable

from mltest.core.observer import Subject


class PipelineContext(dict):
    """State handed from one stage to the next."""


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Do this stage's work and return the (updated) context."""


class FunctionStage(Stage):
    """Turns a plain callable ``f(context) -> context | None`` into a Stage."""

    def __init__(self, name: str, func: Callable[[PipelineContext], PipelineContext | None]):
        self.name = name
        self._func = func

    def run(self, context: PipelineContext) -> PipelineContext:
        result = self._func(context)
        return context if result is None else result


class Pipeline(Subject):
    def __init__(self, name: str, stages: Iterable[Stage] = ()) -> None:
        super().__init__()
        self.name = name
        self._stages: list[Stage] = list(stages)

    def add_stage(self, stage: Stage) -> "Pipeline":
        self._stages.append(stage)
        return self

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(self._stages)

    def run(self, context: PipelineContext | None = None) -> PipelineContext:
        ctx = context if context is not None else PipelineContext()
        ctx.setdefault("timings", {})
        self.notify("pipeline_started", stages=[s.name for s in self._stages])
        start = perf_counter()
        for stage in self._stages:
            self.notify("stage_started", stage=stage.name)
            try:
                ctx = stage.run(ctx)
            except Exception as exc:
                self.notify("stage_failed", stage=stage.name, error=repr(exc))
                raise
            self.notify("stage_finished", stage=stage.name, seconds=ctx["timings"].get(stage.name))
        self.notify("pipeline_finished", seconds=perf_counter() - start)
        return ctx
