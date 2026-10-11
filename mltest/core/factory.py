"""Factory pattern (creational) - centralised, registry-based object creation.

Clients ask a factory for an object by name (``ModelFactory.create_model("svm")``)
and never import concrete classes. New models, loaders or strategies are added
by registering them; no client code changes.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable


class Factory:
    """Base class: every subclass gets its own registry of named creators."""

    kind = "object"
    #: Modules imported lazily on first use so their built-in creators register.
    builtins: tuple[str, ...] = ()

    _creators: dict[str, Callable[..., Any]]
    _loaded: bool

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._creators = {}
        cls._loaded = False

    @classmethod
    def register(cls, key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a class or builder function under ``key``."""

        def decorator(creator: Callable[..., Any]) -> Callable[..., Any]:
            cls._creators[key] = creator
            return creator

        return decorator

    @classmethod
    def _ensure_builtins(cls) -> None:
        if not cls._loaded:
            cls._loaded = True
            for module in cls.builtins:
                importlib.import_module(module)

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> Any:
        cls._ensure_builtins()
        try:
            creator = cls._creators[key]
        except KeyError:
            raise ValueError(
                f"Unknown {cls.kind} '{key}'. Available: {', '.join(cls.available())}"
            ) from None
        return creator(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        cls._ensure_builtins()
        return sorted(cls._creators)


class DataLoaderFactory(Factory):
    kind = "data loader"
    builtins = ("mltest.data.loaders",)


class ModelFactory(Factory):
    """Creates ML models, always wrapped in a ModelAdapter."""

    kind = "model"
    builtins = ("mltest.models.catalog",)

    @classmethod
    def create_model(cls, model_type: str, **params: Any) -> Any:
        return cls.create(model_type, **params)


class TestStrategyFactory(Factory):
    __test__ = False  # not a pytest test class
    kind = "test-generation strategy"
    builtins = ("mltest.modules.test_generation.strategies",)


class DefectStrategyFactory(Factory):
    kind = "defect-prediction strategy"
    builtins = ("mltest.modules.defect_prediction.strategies",)


class OracleStrategyFactory(Factory):
    kind = "oracle strategy"
    builtins = ("mltest.modules.test_oracle.strategies",)


class ModuleFactory(Factory):
    kind = "testing module"
    builtins = (
        "mltest.modules.test_generation.module",
        "mltest.modules.defect_prediction.module",
        "mltest.modules.test_oracle.module",
    )


class ObserverFactory(Factory):
    kind = "observer"
    builtins = ("mltest.core.observer",)


class DecoratorFactory(Factory):
    kind = "stage decorator"
    builtins = ("mltest.core.decorators",)
