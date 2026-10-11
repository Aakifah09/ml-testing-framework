"""Adapter pattern (structural) for programs under test.

ProgramAdapter turns a plain Python function into the interface the testing
strategies need: run an input with line/branch tracing, an execution-step
limit (so mutants with infinite loops cannot hang), and static facts about
the code (executable lines, decision points, literal constants).
"""
from __future__ import annotations

import ast
import copy
import dis
import inspect
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable

from mltest.programs.catalog import ProgramSpec

EXIT = -1  # pseudo line number for "left the function"


class StepLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    inputs: tuple
    output: Any
    error: str | None
    lines: frozenset[int]
    arcs: frozenset[tuple[int, int]]
    steps: int
    #: (comparison site, clamp(left - right, -2, 2)) pairs; only recorded on the original program
    boundaries: frozenset[tuple[int, int]] = frozenset()

    @property
    def ok(self) -> bool:
        return self.error is None

    def signature(self) -> tuple[str, str]:
        """Behaviour fingerprint: two runs behave the same iff signatures match."""
        if self.error is not None:
            return ("error", self.error.split(":", 1)[0])
        return ("ok", repr(self.output))


@dataclass(frozen=True)
class StaticInfo:
    executable_lines: frozenset[int]
    decision_lines: frozenset[int]
    constants: tuple[int, ...]


def analyse_function(func: Callable[..., Any]) -> StaticInfo:
    code = func.__code__
    lines = {ln for _, ln in dis.findlinestarts(code) if ln is not None} - {code.co_firstlineno}
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    offset = code.co_firstlineno - 1
    decisions = {
        node.lineno + offset for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.For))
    }
    constants = sorted({
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) is int
    })
    return StaticInfo(frozenset(lines), frozenset(decisions), tuple(constants))


_COMPARE_FUNCS = {
    "Lt": lambda a, b: a < b, "LtE": lambda a, b: a <= b, "Gt": lambda a, b: a > b,
    "GtE": lambda a, b: a >= b, "Eq": lambda a, b: a == b, "NotEq": lambda a, b: a != b,
}
_RECORDER = "__mltest_compare"


class _CompareInstrumenter(ast.NodeTransformer):
    """Rewrites ``x < y`` into ``__mltest_compare(site, x, y, 'Lt')``."""

    def __init__(self) -> None:
        self.sites = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        op = type(node.ops[0]).__name__
        if len(node.ops) != 1 or op not in _COMPARE_FUNCS:
            return node
        call = ast.Call(ast.Name(_RECORDER, ast.Load()),
                        [ast.Constant(self.sites), node.left, node.comparators[0], ast.Constant(op)], [])
        self.sites += 1
        return ast.copy_location(call, node)


def _distance_class(left: Any, right: Any) -> int:
    try:
        return int(max(-2, min(2, left - right)))
    except TypeError:
        return 0 if left == right else 3


class ProgramAdapter:
    def __init__(
        self,
        spec: ProgramSpec,
        func: Callable[..., Any] | None = None,
        version: str = "original",
        max_steps: int = 5_000,
    ) -> None:
        self.spec = spec
        self.func = func or spec.func
        self.version = version
        self.max_steps = max_steps
        self._boundaries: set[tuple[int, int]] | None = None
        self.comparison_sites = 0
        self._runnable = self._instrument() if self.func is spec.func else self.func

    def _instrument(self) -> Callable[..., Any]:
        """Copy of the original function that also records comparison distances."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.func)))
        ast.increment_lineno(tree, self.func.__code__.co_firstlineno - 1)  # keep real line numbers
        instrumenter = _CompareInstrumenter()
        tree = ast.fix_missing_locations(instrumenter.visit(tree))
        self.comparison_sites = instrumenter.sites

        def record(site: int, left: Any, right: Any, op: str) -> bool:
            if self._boundaries is not None:
                self._boundaries.add((site, _distance_class(left, right)))
            return _COMPARE_FUNCS[op](left, right)

        namespace = {**self.func.__globals__, _RECORDER: record}
        exec(compile(tree, self.func.__code__.co_filename, "exec"), namespace)
        return namespace[self.func.__name__]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def params(self):
        return self.spec.params

    @cached_property
    def static(self) -> StaticInfo:
        # Mutants have no source file; static facts always come from the original.
        return analyse_function(self.spec.func)

    @property
    def decision_lines(self) -> frozenset[int]:
        return self.static.decision_lines

    @property
    def constants(self) -> tuple[int, ...]:
        return self.static.constants

    def with_function(self, func: Callable[..., Any], version: str) -> "ProgramAdapter":
        return ProgramAdapter(self.spec, func=func, version=version, max_steps=self.max_steps)

    # -- execution ----------------------------------------------------------
    def run(self, inputs: tuple) -> ExecutionResult:
        code = self._runnable.__code__
        lines: set[int] = set()
        arcs: set[tuple[int, int]] = set()
        state = {"prev": None, "steps": 0}
        max_steps = self.max_steps

        def local_trace(frame, event, arg):
            if event == "line":
                state["steps"] += 1
                if state["steps"] > max_steps:
                    raise StepLimitExceeded(f"more than {max_steps} steps")
                line = frame.f_lineno
                lines.add(line)
                if state["prev"] is not None:
                    arcs.add((state["prev"], line))
                state["prev"] = line
            elif event == "return" and state["prev"] is not None:
                arcs.add((state["prev"], EXIT))
            return local_trace

        def global_trace(frame, event, arg):
            return local_trace if frame.f_code is code else None

        args = copy.deepcopy(tuple(inputs))
        boundaries: set[tuple[int, int]] = set()
        self._boundaries = boundaries
        previous = sys.gettrace()
        sys.settrace(global_trace)
        try:
            output, error = self._runnable(*args), None
        except StepLimitExceeded:
            output, error = None, "StepLimitExceeded"
        except Exception as exc:  # a crash is a legitimate test outcome
            output, error = None, f"{type(exc).__name__}: {exc}"
        finally:
            sys.settrace(previous)
            self._boundaries = None
        return ExecutionResult(tuple(inputs), output, error, frozenset(lines), frozenset(arcs), state["steps"],
                               frozenset(boundaries))

    # -- coverage -----------------------------------------------------------
    def covered_branches(self, results: Iterable[ExecutionResult]) -> set[tuple[int, int]]:
        decisions = self.decision_lines
        return {arc for r in results for arc in r.arcs if arc[0] in decisions}

    def line_coverage(self, results: Iterable[ExecutionResult]) -> float:
        executable = self.static.executable_lines
        covered = set().union(*(r.lines for r in results)) & executable if executable else set()
        return len(covered) / len(executable) if executable else 1.0

    def branch_coverage(self, results: Iterable[ExecutionResult]) -> float:
        """Each decision (if/while/for) has two outcomes; count distinct successors seen."""
        successors: dict[int, set[int]] = defaultdict(set)
        for source, target in self.covered_branches(results):
            successors[source].add(target)
        total = 2 * len(self.decision_lines)
        covered = sum(min(2, len(successors[d])) for d in self.decision_lines)
        return covered / total if total else 1.0

    def boundary_coverage(self, results: Iterable[ExecutionResult]) -> float:
        """Share of (comparison, distance in {<-1, -1, 0, +1, >+1}) combinations exercised.

        Some combinations are infeasible (e.g. ``x % 4`` is never negative), so 1.0
        is an upper bound that real programs rarely reach.
        """
        sites = self.comparison_sites
        seen = {b for r in results for b in r.boundaries}
        return len(seen) / (5 * sites) if sites else 1.0

    def __repr__(self) -> str:
        return f"ProgramAdapter({self.name}, version={self.version})"
