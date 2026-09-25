"""Exact ensemble counting over all legal RNA secondary structures.

Where :mod:`app.solver` adjudicates a single optimal structure, this module
counts *every* legal structure of a constrained instance (allowed base pairs,
minimum pair distance, no pseudoknots, forced/forbidden position constraints)
without enumerating structures, and reports for each requested review position
how many structures leave it unpaired or pair it with each possible partner.

The recurrences are the counting analogues of the adjudication DP:

    C[i][j] = number of legal structures on interval [i, j)
            = (C[i+1][j] if i is not forced)
              + sum_{r legal partner of i, r < j} C[i+1][r] * C[r+1][j]

    O[i][j] = number of legal structures on the complement of [i, j),
              i.e. on positions [0, i) ∪ [j, n).  Computed over strictly
              larger holes by expanding position i-1 (unpaired, paired with
              a position left of it, or paired with a position at/after j).

The structures containing a given pair (p, q) factorize into an independent
inside and outside part, so count(p~q) = C[p+1][q] * O[p][q+1].  The unpaired
count of a position is the total minus its paired counts, which keeps every
reported distribution exactly conserved (the parts always sum to the total).
All arithmetic is exact: Python integers grow to whatever size the counts need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.solver import _partners


@dataclass(frozen=True)
class PositionCounts:
    """Ensemble distribution for one review position.

    ``partners`` holds ``(partner_position, count)`` entries for every legal
    pairing partner, sorted by partner position ascending; counts may be zero
    when constraints prevent an otherwise legal pair from appearing.
    """

    position: int
    unpaired: int
    partners: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EnsembleCounts:
    """Exact counts over all legal structures of one constrained instance."""

    total: int
    distributions: tuple[PositionCounts, ...]


def _left_partners(partners: list[list[int]], n: int) -> list[list[int]]:
    """For each position, the sorted list of legal partners to its left."""
    left: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for r in partners[i]:
            left[r].append(i)
    return left


def _inside_table(
    n: int, forced: frozenset[int], partners: list[list[int]]
) -> list[list[int]]:
    """C[i][j] = number of legal structures on interval [i, j)."""
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][i] = 1  # empty interval
    for i in range(n - 1, -1, -1):
        Ci = C[i]
        Ci1 = C[i + 1]
        i_forced = i in forced
        for j in range(i + 1, n + 1):
            # Position i stays unpaired (dropped when i is forced) ...
            total = 0 if i_forced else Ci1[j]
            # ... or pairs with some r inside the interval.
            for r in partners[i]:
                if r >= j:
                    break
                total += Ci1[r] * C[r + 1][j]
            Ci[j] = total
    return C


def _outside_table(
    n: int,
    forced: frozenset[int],
    partners: list[list[int]],
    left: list[list[int]],
    C: list[list[int]],
) -> list[list[int]]:
    """O[i][j] = number of legal structures on the complement of [i, j).

    Computed from the largest hole [0, n) down to empty holes; every
    dependency is a strictly larger hole, so the order is exact.
    """
    O = [[0] * (n + 1) for _ in range(n + 1)]
    for size in range(n, -1, -1):
        for i in range(0, n - size + 1):
            j = i + size
            if i == 0:
                # Complement is the single interval [j, n).
                O[0][j] = C[j][n]
                continue
            k = i - 1  # rightmost complement position left of the hole
            total = 0
            if k not in forced:
                total += O[k][j]  # k stays unpaired
            for l in left[k]:
                # k pairs with l < k: inside [l+1, k), outside complement of [l, j).
                total += C[l + 1][k] * O[l][j]
            Ok = O[k]
            Cj = C[j]
            for r in partners[k]:
                if r >= j:
                    # k pairs with r >= j: the pair encloses the hole plus the
                    # segment [j, r); outside is the complement of [k, r+1).
                    total += Cj[r] * Ok[r + 1]
            O[i][j] = total
    return O


def count_ensemble(
    sequence: str,
    forced_positions: Iterable[int] = (),
    forbidden_positions: Iterable[int] = (),
    positions: Iterable[int] = (),
) -> EnsembleCounts:
    """Count all legal structures and per-position pairing distributions.

    ``positions`` are the review positions; each receives the number of
    structures leaving it unpaired plus one count per legal pairing partner.
    When no legal structure exists the total is zero and every distribution
    is empty (zero unpaired, no partner entries).
    """
    n = len(sequence)
    forced = frozenset(forced_positions)
    forbidden = frozenset(forbidden_positions)
    partners = _partners(sequence, forbidden)
    left = _left_partners(partners, n)
    C = _inside_table(n, forced, partners)
    total = C[0][n]
    if total == 0:
        return EnsembleCounts(
            total=0,
            distributions=tuple(
                PositionCounts(position=p, unpaired=0, partners=()) for p in positions
            ),
        )
    O = _outside_table(n, forced, partners, left, C)
    distributions = []
    for p in positions:
        counts: list[tuple[int, int]] = []
        paired = 0
        for l in left[p]:
            # Pair (l, p): inside [l+1, p), outside complement of [l, p+1).
            c = C[l + 1][p] * O[l][p + 1]
            counts.append((l, c))
            paired += c
        for r in partners[p]:
            # Pair (p, r): inside [p+1, r), outside complement of [p, r+1).
            c = C[p + 1][r] * O[p][r + 1]
            counts.append((r, c))
            paired += c
        counts.sort()
        distributions.append(
            PositionCounts(
                position=p,
                unpaired=total - paired,
                partners=tuple(counts),
            )
        )
    return EnsembleCounts(total=total, distributions=tuple(distributions))
