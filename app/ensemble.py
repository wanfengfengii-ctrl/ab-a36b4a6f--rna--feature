"""Exact ensemble counting over all legal RNA secondary structures.

Companion to :mod:`app.solver`: instead of adjudicating one optimal
structure, this module counts *every* legal structure under the exact same
legality rules (allowed base pairs, minimum loop distance, no pseudoknots,
forced positions paired, forbidden positions unpaired) and derives
per-position pairing distributions — all with an inside/outside dynamic
program, never enumerating structures.

Tables
------
* ``C[i][j]`` — number of legal structures on the interval ``[i, j)``.
* ``O[i][j]`` — number of legal structures on the outside of ``[i, j)``,
  i.e. on positions ``[0, i) ∪ [j, n)``; pairs may nest around the gap.

Both recursions decompose by the state of one frontier position, which is
unambiguous, so every structure is counted exactly once::

    C[i][j] = (C[i+1][j]                 if i may stay unpaired)
            + Σ_{r ∈ partners(i), r < j} C[i+1][r] · C[r+1][j]

    O[i][j] = (O[i-1][j]                 if i-1 may stay unpaired)
            + Σ_{r ∈ partners(i-1), r ≥ j} O[i-1][r+1] · C[j][r]
            + Σ_{k : (k, i-1) legal}       O[k][j]     · C[k+1][i-1]

with ``C[i][i] = 1`` and ``O[0][j] = C[j][n]``.

For a reviewed position ``p`` the ensemble splits into disjoint cases —
``p`` unpaired, or ``p`` paired with exactly one partner ``q`` — hence the
counts below are exact and always sum to the grand total ``C[0][n]``::

    unpaired(p)         = C[p][p+1] · O[p][p+1]
    paired(p, q), p < q = C[p+1][q] · O[p][q+1]
    paired(p, q), q < p = C[q+1][p] · O[q][p+1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.solver import _partners


@dataclass(frozen=True)
class PositionCounts:
    """Exact ensemble distribution for one reviewed position.

    ``pairs`` holds ``(partner, count)`` entries for every legal pairing
    partner, sorted by partner position ascending; ``unpaired`` plus the
    sum of all pair counts equals the ensemble total.
    """

    position: int
    unpaired: int
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EnsembleCounts:
    """Exact counts over all legal structures for one constrained instance."""

    total: int
    distributions: tuple[PositionCounts, ...]


def _inside_table(
    sequence: str, forced: frozenset[int], forbidden: frozenset[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Fill the inside table C and return it with the partner lists."""
    n = len(sequence)
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][i] = 1  # empty interval: exactly one (empty) structure
    partners = _partners(sequence, forbidden)
    for i in range(n - 1, -1, -1):
        Ci = C[i]
        Ci1 = C[i + 1]
        if i not in forced:
            # Position i stays unpaired (branch removed when i is forced).
            Ci[i + 1 :] = Ci1[i + 1 :]
        for r in partners[i]:
            inside = Ci1[r]  # structures on [i+1, r), inside pair (i, r)
            if inside == 0:
                continue
            Cr1 = C[r + 1]
            for j in range(r + 1, n + 1):
                Ci[j] += inside * Cr1[j]
    return C, partners


def _outside_table(
    C: list[list[int]],
    partners: list[list[int]],
    forced: frozenset[int],
    n: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Fill the outside table O; also return left-partner lists."""
    left: list[list[int]] = [[] for _ in range(n)]
    for i, rs in enumerate(partners):
        for r in rs:
            left[r].append(i)

    O = [[0] * (n + 1) for _ in range(n + 1)]
    # Row 0: the outside of [0, j) is [j, n), counted by the inside table.
    O[0] = [C[j][n] for j in range(n + 1)]
    for i in range(1, n + 1):
        Oi = O[i]
        Oim1 = O[i - 1]
        p = i - 1  # frontier position entering the outside region
        if p not in forced:
            # p stays unpaired (branch removed when p is forced).
            for j in range(i, n + 1):
                Oi[j] = Oim1[j]
        # p pairs with r to the right (r >= j, possibly across the gap).
        for r in partners[p]:
            outside = Oim1[r + 1]
            if outside == 0:
                continue
            for j in range(i, r + 1):
                Oi[j] += outside * C[j][r]
        # p pairs with k to the left; the pair nests around [k+1, p).
        for k in left[p]:
            inside = C[k + 1][p]
            if inside == 0:
                continue
            Ok = O[k]
            for j in range(i, n + 1):
                Oi[j] += Ok[j] * inside
    return O, left


def count_ensemble(
    sequence: str,
    forced_positions: Iterable[int] = (),
    forbidden_positions: Iterable[int] = (),
    positions: Iterable[int] = (),
) -> EnsembleCounts:
    """Count all legal structures and per-position pairing distributions.

    Returns the exact total number of legal structures and, for each
    reviewed position, how many structures leave it unpaired and how many
    pair it with every legal partner (ascending by partner position).
    """
    n = len(sequence)
    forced = frozenset(forced_positions)
    forbidden = frozenset(forbidden_positions)
    C, partners = _inside_table(sequence, forced, forbidden)
    O, left = _outside_table(C, partners, forced, n)
    total = C[0][n]

    distributions: list[PositionCounts] = []
    for p in positions:
        if total == 0:
            # No legal structure: the distribution is empty by definition.
            distributions.append(PositionCounts(p, 0, ()))
            continue
        unpaired = C[p][p + 1] * O[p][p + 1]
        counts: list[tuple[int, int]] = []
        for q in sorted(left[p] + partners[p]):
            if q < p:
                counts.append((q, C[q + 1][p] * O[q][p + 1]))
            else:
                counts.append((q, C[p + 1][q] * O[p][q + 1]))
        distributions.append(PositionCounts(p, unpaired, tuple(counts)))
    return EnsembleCounts(total=total, distributions=tuple(distributions))
