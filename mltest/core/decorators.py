"""Decorator pattern (structural) - add behaviour to stages without changing them.

A StageDecorator *is* a Stage and *wraps* a Stage, so decorators stack:
``TimingDecorator(LoggingDecorator(load_stage))``.
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Iterable

from mltest.core.factory import DecoratorFactory
from mltest.core.pipeline import PipelineContext, Stage


class StageDecorator(Stage):
    def __init__(self, inner: Stage) -> None:
        self._inner = inner

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._inner.name

    @property
    def inner(self) -> Stage:
        return self._inner

    def run(self, context: PipelineContext) -> PipelineContext:
        return self._inner.run(context)


@DecoratorFactory.register("logging")
class LoggingDecorator(StageDecorator):
    """Logs entry, exit and failures of the wrapped stage."""

    def __init__(self, inner: Stage, logger: logging.Logger | None = None) -> None:
        super().__init__(inner)
        self.logger = logger or logging.getLogger("mltest.pipeline")

    def run(self, context: PipelineContext) -> PipelineContext:
        module = context.get("module", "?")
        self.logger.info("[%s] entering stage '%s'", module, self.name)
        try:
            result = super().run(context)
        except Exception:
            self.logger.exception("[%s] stage '%s' failed", module, self.name)
            raise
        self.logger.info("[%s] leaving stage '%s'", module, self.name)
        return result


@DecoratorFactory.register("timing")
class TimingDecorator(StageDecorator):
    """Measures wall-clock time and records it in ``context['timings']``."""

    def run(self, context: PipelineContext) -> PipelineContext:
        start = perf_counter()
        result = super().run(context)
        result.setdefault("timings", {})[self.name] = perf_counter() - start
        return result


def decorate(stage: Stage, names: Iterable[str]) -> Stage:
    """Wrap ``stage`` in the named decorators (first name = innermost)."""
    for name in names:
        stage = DecoratorFactory.create(name, inner=stage)
    return stage
