"""Deterministic RNA secondary-structure adjudication by interval dynamic programming.

Model
-----
* Positions are 0-based indices into the sequence.
* A pair (i, j) is legal iff ``j - i >= MIN_PAIR_DISTANCE``, the bases are one of
  AU, UA, CG, GC, GU, UG, and neither position is forbidden (must stay unpaired).
* A structure pairs each position at most once and contains no pseudoknots
  (crossing pairs).  Forced positions must be paired.
* Objectives are lexicographic: first maximize the number of pairs, then the
  number of adjacent stacked pairs ((i, j) and (i+1, j-1) both paired).

The two-level score is packed into a single integer ``pairs * _PAIR + stacks``
(``_PAIR`` exceeds any attainable stack count), so the classical Nussinov-style
interval recursion optimizes both levels exactly without enumerating structures:

    A[i][j] = best score over interval [i, j)
            = max( A[i+1][j]                      (i unpaired; dropped if forced),
                   max_r B[i][r] + A[r+1][j] )    (i pairs with r)
    B[i][r] = _PAIR + max( A[i+1][r],             (inside of pair (i, r))
                           B[i+1][r-1] + _STACK ) (stacked on pair (i+1, r-1))

Traceback reconstructs, among all optimal structures, the lexicographically
smallest dot-bracket string under the order '(' < '.' < ')' (the primary
result) and the largest one (used as the second witness whenever the optimum
is not unique).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable

sys.setrecursionlimit(100000)

#: Allowed base pairs (Watson-Crick plus wobble GU/UG).
ALLOWED_PAIRS = frozenset(
    {("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"), ("U", "G")}
)

#: Paired positions must be at least this far apart (j - i >= 4).
MIN_PAIR_DISTANCE = 4

# Score packing: pairs dominate stacks; _PAIR exceeds any possible stack count.
_PAIR = 1 << 20
_STACK = 1
_NEG = -(1 << 60)

# Custom dot-bracket order '(' < '.' < ')' implemented as a translation so that
# native string comparison yields the required ordering.
_KEY_TRANS = str.maketrans({"(": "\x00", ".": "\x01", ")": "\x02"})


def bracket_key(structure: str) -> str:
    """Order key realizing '(' < '.' < ')' for dot-bracket strings."""
    return structure.translate(_KEY_TRANS)


@dataclass(frozen=True)
class Adjudication:
    """Outcome of one deterministic adjudication."""

    feasible: bool
    pairs: int = 0
    stacks: int = 0
    primary: str | None = None
    # A second, distinct optimal structure; present iff the optimum is not unique.
    witness: str | None = None


def _partners(sequence: str, forbidden: frozenset[int]) -> list[list[int]]:
    """For each position, the sorted list of legal pairing partners."""
    n = len(sequence)
    partners: list[list[int]] = [[] for _ in range(n)]
    for i, si in enumerate(sequence):
        if i in forbidden:
            continue
        for r in range(i + MIN_PAIR_DISTANCE, n):
            if r in forbidden:
                continue
            if (si, sequence[r]) in ALLOWED_PAIRS:
                partners[i].append(r)
    return partners


def _compute_tables(
    sequence: str, forced: frozenset[int], forbidden: frozenset[int]
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Fill the interval DP tables A (general) and B (endpoints paired)."""
    n = len(sequence)
    A = [[_NEG] * (n + 1) for _ in range(n + 1)]
    B = [[_NEG] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        A[i][i] = 0  # empty interval
    partners = _partners(sequence, forbidden)
    for i in range(n - 1, -1, -1):
        Ai = A[i]
        Bi = B[i]
        Ai1 = A[i + 1]
        Bi1 = B[i + 1]
        for r in partners[i]:
            a = Ai1[r]  # inside [i+1, r) not closed by a pair
            b = Bi1[r - 1] + _STACK  # inside closed by stacked pair (i+1, r-1)
            Bi[r] = _PAIR + (a if a >= b else b)
        if i in forced:
            # Position i must pair: the "leave i unpaired" branch is removed.
            for j in range(i + 1, n + 1):
                Ai[j] = _NEG
        else:
            Ai[i + 1 :] = Ai1[i + 1 :]
        for r in partners[i]:
            base = Bi[r]
            Ar1 = A[r + 1]
            for j in range(r + 1, n + 1):
                v = base + Ar1[j]
                if v > Ai[j]:
                    Ai[j] = v
    return A, B, partners


def _traceback(
    A: list[list[int]],
    B: list[list[int]],
    partners: list[list[int]],
    forced: frozenset[int],
    n: int,
    minimize: bool,
) -> str:
    """Reconstruct the lexicographically extreme optimal dot-bracket structure.

    ``minimize=True`` yields the smallest structure under '(' < '.' < ')',
    ``False`` the largest.  Each memoized state returns the dot-bracket string
    together with its order key so comparisons never re-translate.
    """
    memo_a: dict[tuple[int, int], tuple[str, str]] = {}
    memo_b: dict[tuple[int, int], tuple[str, str]] = {}

    def solve_a(i: int, j: int) -> tuple[str, str]:
        """Extreme optimal structure for interval [i, j)."""
        if i >= j:
            return ("", "")
        hit = memo_a.get((i, j))
        if hit is not None:
            return hit
        target = A[i][j]
        best: tuple[str, str] | None = None
        if i not in forced and A[i + 1][j] == target:
            s, k = solve_a(i + 1, j)
            best = ("." + s, "\x01" + k)
        for r in partners[i]:
            if r >= j:
                break
            if B[i][r] + A[r + 1][j] == target:
                sb, kb = solve_b(i, r)
                sa, ka = solve_a(r + 1, j)
                cand = (sb + sa, kb + ka)
                if best is None or (cand[1] < best[1] if minimize else cand[1] > best[1]):
                    best = cand
        assert best is not None, "traceback on an infeasible state"
        memo_a[(i, j)] = best
        return best

    def solve_b(i: int, r: int) -> tuple[str, str]:
        """Extreme optimal structure with positions i and r paired."""
        hit = memo_b.get((i, r))
        if hit is not None:
            return hit
        a_val = A[i + 1][r]
        b_val = B[i + 1][r - 1] + _STACK
        if b_val > a_val:
            s, k = solve_b(i + 1, r - 1)
        elif a_val > b_val:
            s, k = solve_a(i + 1, r)
        else:
            sa, ka = solve_a(i + 1, r)
            sb, kb = solve_b(i + 1, r - 1)
            if minimize:
                s, k = (sa, ka) if ka <= kb else (sb, kb)
            else:
                s, k = (sa, ka) if ka >= kb else (sb, kb)
        res = ("(" + s + ")", "\x00" + k + "\x02")
        memo_b[(i, r)] = res
        return res

    return solve_a(0, n)[0]


def adjudicate(
    sequence: str,
    forced_positions: Iterable[int] = (),
    forbidden_positions: Iterable[int] = (),
) -> Adjudication:
    """Run one deterministic adjudication for the given constrained instance."""
    n = len(sequence)
    forced = frozenset(forced_positions)
    forbidden = frozenset(forbidden_positions)
    A, B, partners = _compute_tables(sequence, forced, forbidden)
    best = A[0][n]
    if best < 0:
        return Adjudication(feasible=False)
    primary = _traceback(A, B, partners, forced, n, minimize=True)
    largest = _traceback(A, B, partners, forced, n, minimize=False)
    return Adjudication(
        feasible=True,
        pairs=best // _PAIR,
        stacks=best % _PAIR,
        primary=primary,
        witness=None if largest == primary else largest,
    )


def pairs_from_structure(structure: str) -> list[list[int]]:
    """Extract the sorted pair table [[i, j], ...] from a dot-bracket string."""
    stack: list[int] = []
    pairs: list[list[int]] = []
    for idx, ch in enumerate(structure):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            pairs.append([stack.pop(), idx])
    pairs.sort()
    return pairs
