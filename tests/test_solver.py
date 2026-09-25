"""Unit tests for the interval-DP adjudicator.

Includes a brute-force cross-check: for short sequences every non-crossing
structure is enumerated, and the solver's score, primary structure and witness
logic must match the exact optimum.
"""

from __future__ import annotations

import random
from functools import lru_cache

from app.solver import (
    ALLOWED_PAIRS,
    MIN_PAIR_DISTANCE,
    adjudicate,
    bracket_key,
    pairs_from_structure,
)


def brute_force(sequence: str, forced: frozenset[int], forbidden: frozenset[int]):
    """Enumerate all feasible non-crossing structures; return the optimal set."""
    n = len(sequence)
    partners = [[] for _ in range(n)]
    for i in range(n):
        if i in forbidden:
            continue
        for r in range(i + MIN_PAIR_DISTANCE, n):
            if r not in forbidden and (sequence[i], sequence[r]) in ALLOWED_PAIRS:
                partners[i].append(r)

    @lru_cache(maxsize=None)
    def gen(i: int, j: int) -> tuple[frozenset, ...]:
        if i >= j:
            return (frozenset(),)
        out = []
        if i not in forced:
            out.extend(gen(i + 1, j))
        for r in partners[i]:
            if r < j:
                for inside in gen(i + 1, r):
                    for outside in gen(r + 1, j):
                        out.append(inside | outside | {(i, r)})
        return tuple(out)

    structures = gen(0, n)
    if not structures:
        return None
    best = max(
        structures,
        key=lambda s: (len(s), sum((i + 1, j - 1) in s for i, j in s)),
    )
    best_score = (len(best), sum((i + 1, j - 1) in best for i, j in best))
    optimal = [
        s
        for s in structures
        if (len(s), sum((i + 1, j - 1) in s for i, j in s)) == best_score
    ]
    return best_score, optimal


def to_dotbracket(n: int, pairs: frozenset) -> str:
    out = ["."] * n
    for i, j in pairs:
        out[i] = "("
        out[j] = ")"
    return "".join(out)


def score_of(structure: str) -> tuple[int, int]:
    pairs = pairs_from_structure(structure)
    pair_set = {tuple(p) for p in pairs}
    stacks = sum((i + 1, j - 1) in pair_set for i, j in pair_set)
    return len(pairs), stacks


def test_min_pair_distance_enforced():
    # G..C exactly 4 apart pairs up; 3 apart does not.
    assert adjudicate("GAAC").pairs == 0
    assert adjudicate("GAAAC").pairs == 1
    assert adjudicate("GAAAC").primary == "(...)"


def test_only_allowed_pairs():
    assert adjudicate("AAAAAAAAAAAAAAAAAAAA").pairs == 0
    assert adjudicate("CCCCCCCCCCCCCCCCCCCC").pairs == 0
    # GU wobble pair is allowed.
    assert adjudicate("GAAAU").pairs == 1


def test_stacking_is_second_level():
    # Two nested pairs with one stack beat two unstacked alternatives.
    result = adjudicate("GGAAAACC")
    assert result.pairs == 2
    assert result.stacks == 1
    assert result.primary == "((....))"


def test_forced_position_must_pair():
    result = adjudicate("GAAAC", forced_positions={4})
    assert result.feasible
    assert result.primary == "(...)"


def test_forced_position_infeasible():
    # Position 0 is 'A' and no U exists to pair with.
    result = adjudicate("AAAAAAAAAAAAAAAAAAAA", forced_positions={0})
    assert not result.feasible


def test_forbidden_position_stays_unpaired():
    result = adjudicate("GAAAC", forbidden_positions={4})
    assert result.feasible
    assert result.pairs == 0
    assert result.primary == "....."


def test_forced_and_forbidden_conflict_is_infeasible():
    result = adjudicate("GAAAC", forced_positions={0}, forbidden_positions={0})
    assert not result.feasible


def test_unique_optimum_has_no_witness():
    result = adjudicate("GGGGGAAAACCCCCAAAAA")
    assert result.feasible
    assert result.primary == "(((((....)))))....."
    assert result.pairs == 5
    assert result.stacks == 4
    assert result.witness is None


def test_multiple_optima_witness():
    # Verified against full enumeration: exactly 4 optimal structures, (6, 4).
    result = adjudicate("CUUAAGGGUUAAGUAAGUGU")
    assert result.feasible
    assert (result.pairs, result.stacks) == (6, 4)
    assert result.primary == "(((((...)))))(....)."
    assert result.witness == ".((((...))))((....))"
    # Primary is the smallest under '(' < '.' < ')'.
    assert bracket_key(result.primary) < bracket_key(result.witness)
    # Both reach the same two-level score.
    assert score_of(result.primary) == score_of(result.witness)
    assert score_of(result.primary) == (result.pairs, result.stacks)


def test_deterministic_repeat_calls():
    args = ("AUGCAUGCAUGCAUGCAUGCAUGC", {1, 5}, {10})
    first = adjudicate(*args)
    second = adjudicate(*args)
    assert first == second


def test_solver_matches_brute_force():
    rng = random.Random(20260920)
    for trial in range(300):
        n = rng.randint(5, 14)
        seq = "".join(rng.choice("ACGU") for _ in range(n))
        forced = frozenset(i for i in range(n) if rng.random() < 0.15)
        forbidden = frozenset(i for i in range(n) if rng.random() < 0.15)
        expected = brute_force(seq, forced, forbidden)
        result = adjudicate(seq, forced, forbidden)
        if expected is None:
            assert not result.feasible, (seq, forced, forbidden)
            continue
        (exp_pairs, exp_stacks), optimal = expected
        assert result.feasible, (seq, forced, forbidden)
        assert (result.pairs, result.stacks) == (exp_pairs, exp_stacks)
        optimal_db = [to_dotbracket(n, s) for s in optimal]
        assert result.primary == min(optimal_db, key=bracket_key)
        largest = max(optimal_db, key=bracket_key)
        if largest == result.primary:
            assert result.witness is None
        else:
            assert result.witness == largest
        # Reported structures really attain the advertised score.
        assert score_of(result.primary) == (exp_pairs, exp_stacks)
        if result.witness is not None:
            assert score_of(result.witness) == (exp_pairs, exp_stacks)
