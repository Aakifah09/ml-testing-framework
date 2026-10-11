"""Subject specs, ProgramAdapter tracing and the mutation engine."""
import random

import pytest

from mltest.programs.adapter import ProgramAdapter
from mltest.programs.catalog import PROGRAMS, TRIANGLE
from mltest.programs.mutation import generate_mutants, is_killed, mutant_adapters


@pytest.mark.parametrize("spec", PROGRAMS.values(), ids=list(PROGRAMS))
def test_original_programs_satisfy_their_rules_and_relations(spec):
    """Otherwise the rule/metamorphic oracles would raise false alarms."""
    rng = random.Random(1)
    for _ in range(300):
        inputs = spec.sample_inputs(rng)
        output = spec.func(*inputs)
        for rule in spec.rules:
            assert rule.check(inputs, output), (rule.name, inputs, output)
        for relation in spec.relations:
            if relation.applies(inputs):
                follow = spec.func(*relation.transform(inputs))
                assert relation.relation(output, follow, inputs), (relation.name, inputs)


def test_adapter_traces_coverage():
    program = ProgramAdapter(TRIANGLE)
    suite = [program.run(x) for x in [(0, 1, 1), (1, 1, 5), (3, 3, 3), (3, 3, 4), (3, 4, 5)]]
    assert [r.output for r in suite] == ["invalid", "invalid", "equilateral", "isosceles", "scalene"]
    assert program.line_coverage(suite) == 1.0
    assert program.branch_coverage(suite) == 1.0
    assert program.branch_coverage(suite[:1]) < 1.0
    assert 0 < program.boundary_coverage(suite) < 1.0


def test_instrumented_original_matches_plain_function():
    program = ProgramAdapter(TRIANGLE)
    rng = random.Random(3)
    for _ in range(100):
        x = TRIANGLE.sample_inputs(rng)
        assert program.run(x).output == TRIANGLE.func(*x)


def test_step_limit_stops_infinite_loops():
    def spin(n):
        while n >= 0:
            n += 1
        return n

    from dataclasses import replace
    spec = replace(TRIANGLE, func=spin)
    result = ProgramAdapter(spec, func=spin, version="spin", max_steps=100).run((1,))
    assert result.error == "StepLimitExceeded"


def test_mutants_change_behaviour_and_can_be_killed():
    mutants = generate_mutants(TRIANGLE.func)
    assert len(mutants) > 20
    assert len({m.description for m in mutants}) > 10
    program = ProgramAdapter(TRIANGLE)
    suite = [program.run(x) for x in [(0, 1, 1), (1, 1, 2), (3, 3, 3), (3, 3, 4), (3, 4, 5), (1, 2, 2)]]
    killed = sum(is_killed(adapter, suite) for _, adapter in mutant_adapters(program))
    assert killed / len(mutants) > 0.5
