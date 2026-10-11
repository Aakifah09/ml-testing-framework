"""Specifications of the subject programs: input domains, rules, metamorphic relations.

Rules are partial correctness properties (they do not re-implement the
program). Metamorphic relations relate the outputs of a source input and a
transformed follow-up input, so no expected output is needed at all.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from mltest.programs import subjects


@dataclass(frozen=True)
class IntParam:
    name: str
    low: int
    high: int

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)


@dataclass(frozen=True)
class IntListParam:
    name: str
    min_len: int
    max_len: int
    low: int
    high: int

    def sample(self, rng: random.Random) -> list[int]:
        return [rng.randint(self.low, self.high) for _ in range(rng.randint(self.min_len, self.max_len))]


Param = Union[IntParam, IntListParam]


@dataclass(frozen=True)
class Rule:
    name: str
    check: Callable[[tuple, Any], bool]


def _always(_inputs: tuple) -> bool:
    return True


@dataclass(frozen=True)
class MetamorphicRelation:
    name: str
    transform: Callable[[tuple], tuple]
    relation: Callable[[Any, Any, tuple], bool]  # (source output, follow-up output, source inputs)
    applies: Callable[[tuple], bool] = _always


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    func: Callable[..., Any]
    params: tuple[Param, ...]
    rules: tuple[Rule, ...] = ()
    relations: tuple[MetamorphicRelation, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def sample_inputs(self, rng: random.Random) -> tuple:
        return tuple(p.sample(rng) for p in self.params)


def _same(out1: Any, out2: Any, _inputs: tuple) -> bool:
    return out1 == out2


def _is_sorted(xs) -> bool:
    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))


_TRIANGLE_LABELS = {"invalid", "equilateral", "isosceles", "scalene"}

TRIANGLE = ProgramSpec(
    name="triangle",
    func=subjects.triangle,
    params=(IntParam("a", -3, 20), IntParam("b", -3, 20), IntParam("c", -3, 20)),
    description="Classify a triangle from its three side lengths.",
    rules=(
        Rule("known_label", lambda i, o: o in _TRIANGLE_LABELS),
        Rule("non_positive_side_is_invalid", lambda i, o: min(i) > 0 or o == "invalid"),
        Rule("inequality_violation_is_invalid",
             lambda i, o: (i[0] + i[1] > i[2] and i[0] + i[2] > i[1] and i[1] + i[2] > i[0]) or o == "invalid"),
        Rule("equilateral_has_equal_sides", lambda i, o: o != "equilateral" or i[0] == i[1] == i[2]),
        Rule("isosceles_has_two_equal_sides", lambda i, o: o != "isosceles" or len(set(i)) == 2),
        Rule("scalene_has_distinct_sides", lambda i, o: o != "scalene" or len(set(i)) == 3),
    ),
    relations=(
        MetamorphicRelation("rotate_sides", lambda i: (i[1], i[2], i[0]), _same),
        MetamorphicRelation("swap_first_two", lambda i: (i[1], i[0], i[2]), _same),
        MetamorphicRelation("scale_by_two", lambda i: tuple(2 * x for x in i), _same),
    ),
)

DAYS_IN_MONTH = ProgramSpec(
    name="days_in_month",
    func=subjects.days_in_month,
    params=(IntParam("year", 1600, 2400), IntParam("month", -1, 14)),
    description="Number of days in a month (-1 for an invalid month).",
    rules=(
        Rule("known_value", lambda i, o: o in (-1, 28, 29, 30, 31)),
        Rule("invalid_month_iff_minus_one", lambda i, o: (i[1] < 1 or i[1] > 12) == (o == -1)),
        Rule("february_has_28_or_29", lambda i, o: i[1] != 2 or o in (28, 29)),
        Rule("leap_day_needs_year_divisible_by_4", lambda i, o: o != 29 or i[0] % 4 == 0),
        Rule("thirty_day_months", lambda i, o: i[1] not in (4, 6, 9, 11) or o == 30),
    ),
    relations=(
        MetamorphicRelation("shift_400_years", lambda i: (i[0] + 400, i[1]), _same),
        MetamorphicRelation("next_year_outside_february", lambda i: (i[0] + 1, i[1]), _same,
                            applies=lambda i: i[1] != 2),
        MetamorphicRelation("february_never_longer_in_non_leap_year",
                            lambda i: (i[0] - i[0] % 4 + 1, i[1]),
                            lambda o1, o2, i: o2 <= o1,
                            applies=lambda i: i[1] == 2),
    ),
)

GCD = ProgramSpec(
    name="gcd",
    func=subjects.gcd,
    params=(IntParam("a", -60, 60), IntParam("b", -60, 60)),
    description="Greatest common divisor of two integers.",
    rules=(
        Rule("non_negative", lambda i, o: o >= 0),
        Rule("divides_both", lambda i, o: o == 0 or (i[0] % o == 0 and i[1] % o == 0)),
        Rule("zero_iff_both_zero", lambda i, o: (o == 0) == (i[0] == 0 and i[1] == 0)),
        Rule("bounded_by_largest_input", lambda i, o: o <= max(abs(i[0]), abs(i[1]))),
    ),
    relations=(
        MetamorphicRelation("commutative", lambda i: (i[1], i[0]), _same),
        MetamorphicRelation("scale_by_two", lambda i: (2 * i[0], 2 * i[1]), lambda o1, o2, i: o2 == 2 * o1),
        MetamorphicRelation("add_first_to_second", lambda i: (i[0], i[1] + i[0]), _same),
        MetamorphicRelation("negate_first", lambda i: (-i[0], i[1]), _same),
    ),
)

INSERTION_SORT = ProgramSpec(
    name="insertion_sort",
    func=subjects.insertion_sort,
    params=(IntListParam("xs", 0, 8, -20, 20),),
    description="Sort a list of integers in ascending order.",
    rules=(
        Rule("same_length", lambda i, o: len(o) == len(i[0])),
        Rule("ascending", lambda i, o: _is_sorted(o)),
        Rule("permutation_of_input", lambda i, o: Counter(o) == Counter(i[0])),
    ),
    relations=(
        MetamorphicRelation("reverse_input", lambda i: (list(reversed(i[0])),), _same),
        MetamorphicRelation("rotate_input", lambda i: (i[0][1:] + i[0][:1],), _same),
        MetamorphicRelation("append_new_maximum",
                            lambda i: (i[0] + [max(i[0], default=0) + 1],),
                            lambda o1, o2, i: o2 == list(o1) + [max(i[0], default=0) + 1]),
        MetamorphicRelation("negate_elements",
                            lambda i: ([-x for x in i[0]],),
                            lambda o1, o2, i: list(o2) == [-x for x in reversed(o1)]),
    ),
)

PROGRAMS: dict[str, ProgramSpec] = {spec.name: spec for spec in (TRIANGLE, DAYS_IN_MONTH, GCD, INSERTION_SORT)}


def get_program(name: str) -> ProgramSpec:
    try:
        return PROGRAMS[name]
    except KeyError:
        raise ValueError(f"Unknown program '{name}'. Available: {', '.join(PROGRAMS)}") from None
