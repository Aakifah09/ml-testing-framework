"""Common shape of a testing module: the five pipeline steps."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mltest.core.config import AppConfig
from mltest.core.observer import Subject
from mltest.core.pipeline import FunctionStage, PipelineContext, Stage

STAGE_NAMES = ("load", "preprocess", "execute", "evaluate", "report")


class TestingModule(ABC):
    """Load -> Preprocess -> Execute -> Evaluate -> Report for one testing scenario."""

    __test__ = False  # not a pytest test class
    name: str = "module"
    #: Columns of ``context['metrics']`` shown in the run summary.
    summary_columns: tuple[str, ...] = ()

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.settings: dict[str, Any] = self.config.section(self.name)
        self.seed = self.config.seed
        self._subject: Subject | None = None

    def bind(self, subject: Subject) -> None:
        """Publish this module's events through ``subject`` (its pipeline)."""
        self._subject = subject

    def emit(self, event_type: str, **payload: Any) -> None:
        if self._subject is not None:
            self._subject.notify(event_type, **payload)

    def say(self, message: str) -> None:
        self.emit("notification", message=message)

    @property
    def output_dir(self) -> Path:
        path = self.config.run_dir / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stages(self) -> list[Stage]:
        return [FunctionStage(name, getattr(self, name)) for name in STAGE_NAMES]

    @abstractmethod
    def load(self, ctx: PipelineContext) -> PipelineContext: ...

    @abstractmethod
    def preprocess(self, ctx: PipelineContext) -> PipelineContext: ...

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext: ...

    @abstractmethod
    def evaluate(self, ctx: PipelineContext) -> PipelineContext: ...

    @abstractmethod
    def report(self, ctx: PipelineContext) -> PipelineContext: ...
