"""Observer pattern (behavioral) - pipelines publish events, observers react.

The pipeline does not know who is listening: console output, the JSON event
log and (optionally) MLflow tracking are all just observers.
"""
from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from mltest.core.factory import ObserverFactory


@dataclass(frozen=True)
class Event:
    type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))


class Observer(ABC):
    @abstractmethod
    def update(self, event: Event) -> None:
        """React to an event published by a Subject."""


class Subject:
    """Keeps a list of observers and notifies all of them."""

    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def attach(self, observer: Observer) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    @property
    def observers(self) -> tuple[Observer, ...]:
        return tuple(self._observers)

    def notify(self, event_type: str, **payload: Any) -> Event:
        event = Event(event_type, getattr(self, "name", type(self).__name__), payload)
        for observer in list(self._observers):
            observer.update(event)
        return event


@ObserverFactory.register("console")
class ConsoleObserver(Observer):
    """Prints a short, human-readable line for the interesting events."""

    def __init__(self, stream: TextIO | None = None, verbose: bool = False) -> None:
        self.stream = stream or sys.stdout
        self.verbose = verbose

    def update(self, event: Event) -> None:
        line = self._format(event)
        if line:
            print(line, file=self.stream, flush=True)

    def _format(self, event: Event) -> str | None:
        p, src = event.payload, event.source
        if event.type == "pipeline_started":
            return f"\n[{src}] pipeline started: {' -> '.join(p['stages'])}"
        if event.type == "stage_finished":
            seconds = p.get("seconds")
            took = f" ({seconds:.2f}s)" if seconds is not None else ""
            return f"[{src}]   {p['stage']:<10} done{took}"
        if event.type == "stage_failed":
            return f"[{src}]   {p['stage']} FAILED: {p['error']}"
        if event.type == "pipeline_finished":
            return f"[{src}] pipeline finished in {p['seconds']:.2f}s"
        if event.type == "notification":
            return f"[{src}]     {p['message']}"
        if self.verbose:
            return f"[{src}] {event.type}: {p}"
        return None


@ObserverFactory.register("report")
class ReportObserver(Observer):
    """Collects every event so the whole run can be written to events.json."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def update(self, event: Event) -> None:
        self.events.append(asdict(event))

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]

    def flush(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.events, indent=2, default=str), encoding="utf-8")
        return path


@ObserverFactory.register("mlflow")
class MLflowObserver(Observer):
    """Logs every metrics record as an MLflow run (requires ``pip install mlflow``)."""

    def __init__(self, experiment: str = "mltest") -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("The 'mlflow' observer needs MLflow: pip install mlflow") from exc
        self._mlflow = mlflow
        mlflow.set_experiment(experiment)

    def update(self, event: Event) -> None:
        if event.type != "metrics":
            return
        for record in event.payload.get("records", []):
            numeric = {
                k: float(v)
                for k, v in record.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            with self._mlflow.start_run(run_name=f"{event.source}:{record.get('name')}"):
                self._mlflow.log_metrics(numeric)
