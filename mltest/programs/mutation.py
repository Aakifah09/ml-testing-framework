"""AST-based mutation testing: generate faulty versions of a function.

Each mutant changes exactly one operator or literal (``<`` -> ``<=``,
``and`` -> ``or``, ``+`` -> ``-``, ``4`` -> ``5`` ...). A test suite *kills*
a mutant when some test behaves differently on it than on the original.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from mltest.programs.adapter import ExecutionResult, ProgramAdapter

_COMPARE_SWAPS = {
    ast.Lt: (ast.LtE, ast.Gt), ast.LtE: (ast.Lt, ast.GtE), ast.Gt: (ast.GtE, ast.Lt),
    ast.GtE: (ast.Gt, ast.LtE), ast.Eq: (ast.NotEq,), ast.NotEq: (ast.Eq,),
    ast.In: (ast.NotIn,), ast.NotIn: (ast.In,),
}
_BINOP_SWAPS = {
    ast.Add: (ast.Sub,), ast.Sub: (ast.Add,), ast.Mult: (ast.Add,),
    ast.Div: (ast.Mult,), ast.FloorDiv: (ast.Mod,), ast.Mod: (ast.FloorDiv,),
}
_BOOL_SWAPS = {ast.And: ast.Or, ast.Or: ast.And}


@dataclass(frozen=True)
class Mutant:
    id: str
    description: str
    func: Callable[..., Any]


# The outermost of these around a mutation is shown as its context, e.g. "b <= 0  ->  b >= 0".
_CONTEXT_NODES = (ast.Compare, ast.BinOp, ast.Call, ast.Subscript, ast.Return, ast.Assign, ast.AugAssign)


class _Mutator(ast.NodeTransformer):
    """Walks the AST numbering every possible mutation; keeps only the one at ``target``."""

    def __init__(self, target: int | None = None, line_offset: int = 0) -> None:
        self.target = target
        self.line_offset = line_offset
        self.descriptions: list[str] = []
        self._context: ast.AST | None = None

    def visit(self, node: ast.AST) -> ast.AST:
        is_context = self._context is None and isinstance(node, _CONTEXT_NODES)
        if is_context:
            self._context = node
        try:
            return super().visit(node)
        finally:
            if is_context:
                self._context = None

    def _offer(self, node: ast.AST, apply: Callable[[], None], undo: Callable[[], None]) -> None:
        """Record the mutation as 'code before -> code after'; undo it unless it is the target."""
        if self.target is not None and len(self.descriptions) > self.target:
            self.descriptions.append("")  # target already applied: only keep the numbering in step
            return
        context = self._context or node
        before = ast.unparse(context)
        apply()
        self.descriptions.append(f"line {node.lineno + self.line_offset}: {before}  ->  {ast.unparse(context)}")
        if len(self.descriptions) - 1 != self.target:
            undo()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        for statement in node.body:  # body only: leave the signature/defaults alone
            self.visit(statement)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        self.generic_visit(node)
        for i, op in enumerate(list(node.ops)):
            for replacement in _COMPARE_SWAPS.get(type(op), ()):
                self._offer(node, lambda: node.ops.__setitem__(i, replacement()),
                            lambda: node.ops.__setitem__(i, op))
        return node

    def _swap_operator(self, node: ast.BinOp | ast.AugAssign | ast.BoolOp, replacements) -> None:
        original = node.op
        for replacement in replacements:
            self._offer(node, lambda: setattr(node, "op", replacement()), lambda: setattr(node, "op", original))

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        self.generic_visit(node)
        self._swap_operator(node, _BINOP_SWAPS.get(type(node.op), ()))
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        self.generic_visit(node)
        self._swap_operator(node, _BINOP_SWAPS.get(type(node.op), ()))
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        self.generic_visit(node)
        self._swap_operator(node, (_BOOL_SWAPS[type(node.op)],))
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if type(node.value) is int:
            value = node.value
            self._offer(node, lambda: setattr(node, "value", value + 1), lambda: setattr(node, "value", value))
        return node


def generate_mutants(func: Callable[..., Any]) -> list[Mutant]:
    source = textwrap.dedent(inspect.getsource(func))
    offset = func.__code__.co_firstlineno - 1
    probe = _Mutator(line_offset=offset)
    probe.visit(ast.parse(source))

    mutants = []
    for index, description in enumerate(probe.descriptions):
        tree = ast.parse(source)
        _Mutator(target=index, line_offset=offset).visit(tree)
        ast.fix_missing_locations(tree)
        namespace = dict(func.__globals__)
        exec(compile(tree, f"<mutant {func.__name__}#{index}>", "exec"), namespace)
        mutants.append(Mutant(f"{func.__name__}_m{index:03d}", description, namespace[func.__name__]))
    return mutants


def mutant_adapters(program: ProgramAdapter) -> list[tuple[Mutant, ProgramAdapter]]:
    return [(m, program.with_function(m.func, m.id)) for m in generate_mutants(program.spec.func)]


def killing_test(mutant: ProgramAdapter, reference_results: Iterable[ExecutionResult]
                 ) -> tuple[ExecutionResult, ExecutionResult] | None:
    """First test that behaves differently on the mutant: (original run, mutant run), else None."""
    for ref in reference_results:
        observed = mutant.run(ref.inputs)
        if observed.signature() != ref.signature():
            return ref, observed
    return None


def is_killed(mutant: ProgramAdapter, reference_results: Iterable[ExecutionResult]) -> bool:
    """True if any test behaves differently on the mutant than on the original."""
    return killing_test(mutant, reference_results) is not None


def describe_call(func_name: str, result: ExecutionResult) -> str:
    """``triangle(3, 3, 3) -> 'equilateral'`` (or the error)."""
    args = ", ".join(repr(x) for x in result.inputs)
    outcome = repr(result.output) if result.ok else f"raises {result.error.split(':', 1)[0]}"
    return f"{func_name}({args}) -> {outcome}"
