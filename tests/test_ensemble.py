"""Unit tests for the exact ensemble counter.

Includes a brute-force cross-check: for short sequences every legal
structure is enumerated, and the counter's total, per-position unpaired
count and per-partner counts must match the enumeration exactly.
"""

from __future__ import annotations

import random
from functools import lru_cache

from app.ensemble import count_ensemble
from app.solver import ALLOWED_PAIRS, MIN_PAIR_DISTANCE


def brute_force_structures(
    sequence: str, forced: frozenset[int], forbidden: frozenset[int]
) -> tuple[frozenset, ...]:
    """Enumerate every legal structure as a frozenset of (i, j) pairs."""
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

    return gen(0, n)


def legal_partners(sequence: str, p: int, forbidden: frozenset[int]) -> list[int]:
    """Every position p may legally pair with (base/distance/forbidden)."""
    n = len(sequence)
    out = []
    for q in range(n):
        if q == p or p in forbidden or q in forbidden:
            continue
        a, b = min(p, q), max(p, q)
        if b - a >= MIN_PAIR_DISTANCE and (sequence[a], sequence[b]) in ALLOWED_PAIRS:
            out.append(q)
    return out


def test_single_pair_counted():
    # Only (0, 4) is legal: structures are {} and {(0, 4)}.
    result = count_ensemble("GAAAC", positions=[0, 2, 4])
    assert result.total == 2
    by_pos = {d.position: d for d in result.distributions}
    assert by_pos[0].unpaired == 1
    assert by_pos[0].pairs == ((4, 1),)
    assert by_pos[2].unpaired == 2
    assert by_pos[2].pairs == ()
    assert by_pos[4].unpaired == 1
    assert by_pos[4].pairs == ((0, 1),)


def test_no_legal_pairs_counts_empty_structure_only():
    result = count_ensemble("A" * 20, positions=[0, 19])
    assert result.total == 1
    for dist in result.distributions:
        assert dist.unpaired == 1
        assert dist.pairs == ()


def test_nested_and_disjoint_structures():
    # Legal pairs: (0,6), (0,7), (1,6), (1,7); non-crossing subsets give
    # {}, four singletons and the nested {(0,7),(1,6)}: 6 structures.
    result = count_ensemble("GGAAAACC", positions=[0, 1, 6, 7])
    assert result.total == 6
    by_pos = {d.position: d for d in result.distributions}
    assert by_pos[0].unpaired == 3
    assert by_pos[0].pairs == ((6, 1), (7, 2))
    assert by_pos[1].unpaired == 3
    assert by_pos[1].pairs == ((6, 2), (7, 1))
    assert by_pos[6].unpaired == 3
    assert by_pos[6].pairs == ((0, 1), (1, 2))
    assert by_pos[7].unpaired == 3
    assert by_pos[7].pairs == ((0, 2), (1, 1))


def test_forced_position_never_unpaired():
    result = count_ensemble("GGAAAACC", forced_positions={0}, positions=[0, 1])
    # Only structures with 0 paired: {(0,6)}, {(0,7)}, {(0,7),(1,6)}.
    assert result.total == 3
    by_pos = {d.position: d for d in result.distributions}
    assert by_pos[0].unpaired == 0
    assert by_pos[0].pairs == ((6, 1), (7, 2))


def test_forbidden_position_has_empty_pairing():
    result = count_ensemble("GGAAAACC", forbidden_positions={7}, positions=[7, 0])
    # Pair (x, 7) removed: {}, {(0,6)}, {(1,6)}.
    assert result.total == 3
    by_pos = {d.position: d for d in result.distributions}
    assert by_pos[7].unpaired == 3
    assert by_pos[7].pairs == ()
    assert by_pos[0].pairs == ((6, 1),)  # forbidden 7 is not a legal partner


def test_infeasible_instance_yields_zero_and_empty_distributions():
    result = count_ensemble("A" * 20, forced_positions={0}, positions=[0, 5, 19])
    assert result.total == 0
    for dist in result.distributions:
        assert dist.unpaired == 0
        assert dist.pairs == ()


def test_forced_forbidden_conflict_is_infeasible():
    result = count_ensemble(
        "GAAAC", forced_positions={0}, forbidden_positions={0}, positions=[0]
    )
    assert result.total == 0
    assert result.distributions[0].pairs == ()


def test_partners_listed_ascending():
    result = count_ensemble("GAAACAAACAAAGAAACAAU", positions=[0])
    partners = [q for q, _ in result.distributions[0].pairs]
    assert partners == sorted(partners)


def test_deterministic_repeat_calls():
    args = ("AUGCAUGCAUGCAUGCAUGCAUGC", {1, 5}, {10}, [0, 3, 7, 11])
    first = count_ensemble(args[0], args[1], args[2], args[3])
    second = count_ensemble(args[0], args[1], args[2], args[3])
    assert first == second


def test_ensemble_matches_brute_force():
    rng = random.Random(20260925)
    for _ in range(300):
        n = rng.randint(4, 14)
        seq = "".join(rng.choice("ACGU") for _ in range(n))
        forced = frozenset(i for i in range(n) if rng.random() < 0.12)
        forbidden = frozenset(i for i in range(n) if rng.random() < 0.12)
        positions = rng.sample(range(n), rng.randint(1, min(5, n)))
        structures = brute_force_structures(seq, forced, forbidden)
        result = count_ensemble(seq, forced, forbidden, positions)
        assert result.total == len(structures), (seq, forced, forbidden)
        for dist in result.distributions:
            p = dist.position
            if result.total == 0:
                assert dist.unpaired == 0 and dist.pairs == ()
                continue
            # Every legal partner is listed, ascending.
            assert [q for q, _ in dist.pairs] == legal_partners(seq, p, forbidden)
            expected_unpaired = sum(
                1 for s in structures if all(p not in pair for pair in s)
            )
            assert dist.unpaired == expected_unpaired
            expected_pairs: dict[int, int] = {}
            for s in structures:
                for a, b in s:
                    if a == p:
                        expected_pairs[b] = expected_pairs.get(b, 0) + 1
                    elif b == p:
                        expected_pairs[a] = expected_pairs.get(a, 0) + 1
            for q, count in dist.pairs:
                assert count == expected_pairs.get(q, 0)
            # Conservation: the distribution partitions the ensemble.
            assert dist.unpaired + sum(c for _, c in dist.pairs) == result.total


def test_conservation_on_dense_max_length_instances():
    rng = random.Random(97)
    for _ in range(10):
        seq = "".join(rng.choice("ACGU") for _ in range(120))
        positions = rng.sample(range(120), 12)
        result = count_ensemble(seq, positions=positions)
        assert result.total > 0
        for dist in result.distributions:
            assert dist.unpaired + sum(c for _, c in dist.pairs) == result.total
