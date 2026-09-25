"""One-shot acceptance service for the RNA folding adjudication API.

Runs an end-to-end, black-box suite against a running API instance (located
through ``RNA_API_BASE_URL``) and exits with status 0 only when every check
passes.  The checks cover the acceptance contract:

* health endpoint and versioned routing;
* deterministic adjudication (unique optimum / multiple optima / infeasible);
* lexicographic primary selection under '(' < '.' < ')' and a distinct witness;
* two-level scoring (pairs, then adjacent stacked pairs);
* dot-bracket / pair-table consistency and every structural legality rule;
* forced/forbidden constraint satisfaction;
* exact ensemble counting: constrained totals, per-position distributions,
  distribution conservation, decimal-string counts, cross-checked against
  an independent brute-force enumeration;
* rejection of malformed input with HTTP 422 before any solving happens.

Run locally:  python -m acceptance.run
In compose:  docker compose --profile acceptance up acceptance
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx

API_BASE = os.environ.get("RNA_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
FOLD_PATH = "/api/v1/fold"
ENSEMBLE_PATH = "/api/v1/fold/ensemble"
TIMEOUT = float(os.environ.get("RNA_API_TIMEOUT", "15"))

# Bases that may pair with each other.
ALLOWED_PAIRS = {
    ("A", "U"),
    ("U", "A"),
    ("C", "G"),
    ("G", "C"),
    ("G", "U"),
    ("U", "G"),
}
MIN_PAIR_DISTANCE = 4
# Custom character order '(' < '.' < ')'.
ORDER = {"(": 0, ".": 1, ")": 2}


def bracket_key(structure: str) -> tuple[int, ...]:
    return tuple(ORDER[c] for c in structure)


def enumerate_structures(
    sequence: str, forced: frozenset[int], forbidden: frozenset[int]
) -> tuple[frozenset, ...]:
    """Reference enumeration of every legal structure (sets of (i, j) pairs).

    Independent of the service implementation; used to cross-check the
    ensemble endpoint's exact counts on small instances.
    """
    n = len(sequence)
    partners: list[list[int]] = [[] for _ in range(n)]
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
        out: list[frozenset] = []
        if i not in forced:
            out.extend(gen(i + 1, j))
        for r in partners[i]:
            if r < j:
                for inside in gen(i + 1, r):
                    for outside in gen(r + 1, j):
                        out.append(inside | outside | {(i, r)})
        return tuple(out)

    return gen(0, n)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class CheckFailed(AssertionError):
    """Raised by check helpers when an expectation is not met."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def expect_eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise CheckFailed(f"{label}: expected {expected!r}, got {actual!r}")


def parse_pairs(structure: str) -> list[tuple[int, int]]:
    """Parse a dot-bracket string into its pair list (raises if unbalanced)."""
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for idx, ch in enumerate(structure):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            expect(bool(stack), f"unbalanced dot-bracket: {structure!r}")
            pairs.append((stack.pop(), idx))
        else:
            expect(ch == ".", f"illegal bracket character {ch!r}")
    expect(not stack, f"unbalanced dot-bracket: {structure!r}")
    return pairs


def validate_structure(
    structure: str,
    sequence: str,
    length: int,
    forced: list[int],
    forbidden: list[int],
    expected_pairs: int | None = None,
    expected_stacks: int | None = None,
) -> tuple[int, int]:
    """Validate every structural rule; return the (pairs, stacks) score."""
    expect_eq(len(structure), length, "structure length")
    pairs = parse_pairs(structure)

    # Each position pairs at most once.
    endpoints: list[int] = [p for pair in pairs for p in pair]
    expect(
        len(endpoints) == len(set(endpoints)),
        "a position participates in more than one pair",
    )

    stack_count = 0
    pair_set = set(pairs)
    for i, j in pairs:
        # Minimum loop distance.
        expect(
            j - i >= MIN_PAIR_DISTANCE,
            f"pair ({i},{j}) closer than {MIN_PAIR_DISTANCE} apart",
        )
        # Only legal base combinations.
        expect(
            (sequence[i], sequence[j]) in ALLOWED_PAIRS,
            f"illegal base pairing ({i}:{sequence[i]},{j}:{sequence[j]})",
        )
        # No pseudoknots: nested or disjoint, never crossing.
        for k, l in pairs:
            expect(
                i == k or i < k < l < j or k < i < j < l or j <= k or l <= i,
                f"pseudoknot between ({i},{j}) and ({k},{l})",
            )
        if (i + 1, j - 1) in pair_set:
            stack_count += 1

    paired = set(endpoints)
    for pos in forced:
        expect(pos in paired, f"forced position {pos} is unpaired")
    for pos in forbidden:
        expect(pos not in paired, f"forbidden position {pos} is paired")

    if expected_pairs is not None:
        expect_eq(len(pairs), expected_pairs, "pair count")
    if expected_stacks is not None:
        expect_eq(stack_count, expected_stacks, "stack count")
    return len(pairs), stack_count


def validate_view(
    view: dict,
    sequence: str,
    length: int,
    forced: list[int],
    forbidden: list[int],
    expected_pairs: int,
    expected_stacks: int,
) -> tuple[int, int]:
    expect_eq(set(view.keys()), {"structure", "pairs", "score"}, "view fields")
    score = view["score"]
    expect_eq(set(score.keys()), {"pairs", "stacks"}, "score fields")
    structure = view["structure"]
    computed = validate_structure(
        structure, sequence, length, forced, forbidden,
        expected_pairs, expected_stacks,
    )
    expect_eq(score["pairs"], expected_pairs, "score.pairs")
    expect_eq(score["stacks"], expected_stacks, "score.stacks")

    # Pair table must match the dot-bracket string exactly (sorted by opener).
    parsed = sorted(list(p) for p in parse_pairs(structure))
    expect_eq(view["pairs"], parsed, "pair table vs dot-bracket")
    return computed


class Acceptance:
    def __init__(self) -> None:
        self.client = httpx.Client(base_url=API_BASE, timeout=TIMEOUT)

    def close(self) -> None:
        self.client.close()

    def fold(self, payload: dict) -> httpx.Response:
        return self.client.post(FOLD_PATH, json=payload)

    def ensemble(self, payload: dict) -> httpx.Response:
        return self.client.post(ENSEMBLE_PATH, json=payload)

    # -- individual checks -------------------------------------------------

    def check_health(self) -> None:
        resp = self.client.get("/health")
        expect_eq(resp.status_code, 200, "health status")
        expect_eq(resp.json(), {"status": "ok"}, "health body")

    def check_versioned_routing(self) -> None:
        resp = self.client.get(FOLD_PATH)
        expect(resp.status_code == 405, f"GET on fold endpoint should be 405, got {resp.status_code}")
        resp = self.client.post("/api/v0/fold", json={"sequence": "A" * 20})
        expect(resp.status_code == 404, f"unknown API version should be 404, got {resp.status_code}")

    def check_unique_optimum(self) -> None:
        seq = "G" + "A" * 18 + "C"
        resp = self.fold({"sequence": seq})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["status"], "OPTIMAL", "status field")
        expect_eq(body["length"], 20, "length")
        expect_eq(body["sequence"], seq, "echoed sequence")
        expect(body["unique"] is True, "should be a unique optimum")
        expect(body["witness"] is None, "witness must be null for unique optimum")
        primary = body["primary"]
        expect_eq(primary["structure"], "(..................)", "dot-bracket")
        validate_view(primary, seq, 20, [], [], expected_pairs=1, expected_stacks=0)
        expect_eq(primary["pairs"], [[0, 19]], "pair table")

    def check_no_legal_pairs(self) -> None:
        seq = "A" * 20
        resp = self.fold({"sequence": seq})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["status"], "OPTIMAL", "status field")
        expect(body["unique"] is True, "empty pairing should be unique")
        expect(body["witness"] is None, "witness must be null")
        primary = body["primary"]
        expect_eq(primary["structure"], "." * 20, "all-unpaired structure")
        validate_view(primary, seq, 20, [], [], expected_pairs=0, expected_stacks=0)

    def check_wobble_and_multiple_optima(self) -> None:
        # G0 can pair U19, but interior A's can also pair U19: several optima.
        seq = "G" + "A" * 18 + "U"
        resp = self.fold({"sequence": seq})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["status"], "OPTIMAL", "status field")
        expect(body["unique"] is False, "optimum is not unique")
        primary = body["primary"]
        witness = body["witness"]
        expect(witness is not None, "witness must be returned for multiple optima")
        expect(primary["structure"] != witness["structure"], "witness must differ")
        validate_view(primary, seq, 20, [], [], 1, 0)
        validate_view(witness, seq, 20, [], [], 1, 0)
        # Primary is the character-order minimum: '(' < '.' < ')'.
        expect(
            bracket_key(primary["structure"]) < bracket_key(witness["structure"]),
            "primary must be the smallest structure under '(' < '.' < ')'",
        )
        expect_eq(primary["structure"], "(..................)", "lexicographic minimum")

    def check_stacking_is_second_objective(self) -> None:
        seq = "GGAAAACC" + "A" * 12
        resp = self.fold({"sequence": seq})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect(body["unique"] is True, "stacked optimum should be unique")
        primary = body["primary"]
        expect_eq(primary["structure"], "((....))" + "." * 12, "stacked structure")
        validate_view(primary, seq, 20, [], [], expected_pairs=2, expected_stacks=1)

    def check_lexicographic_witness_pair(self) -> None:
        seq = "CUUAAGGGUUAAGUAAGUGU"
        resp = self.fold({"sequence": seq})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect(body["unique"] is False, "fixture has multiple optima")
        primary_s = "(((((...)))))(....)."
        witness_s = ".((((...))))((....))"
        primary = body["primary"]
        witness = body["witness"]
        expect_eq(primary["structure"], primary_s, "primary dot-bracket")
        expect_eq(witness["structure"], witness_s, "witness dot-bracket")
        validate_view(primary, seq, 20, [], [], 6, 4)
        validate_view(witness, seq, 20, [], [], 6, 4)
        expect(bracket_key(primary_s) < bracket_key(witness_s), "char-order selection")
        expect_eq(primary["score"], witness["score"], "both witnesses same two-level score")

    def check_infeasible_forced(self) -> None:
        seq = "A" * 20
        resp = self.fold({"sequence": seq, "forced_positions": [0]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["status"], "INFEASIBLE", "status field")
        expect(body["primary"] is None, "no structure on infeasible")
        expect(body["witness"] is None, "no witness on infeasible")
        expect_eq(body["length"], 20, "length still reported")

    def check_infeasible_forced_forbidden_conflict(self) -> None:
        payload = {"sequence": "G" + "A" * 18 + "C",
                   "forced_positions": [0], "forbidden_positions": [0]}
        resp = self.fold(payload)
        expect_eq(resp.status_code, 200, "status")
        expect_eq(resp.json()["status"], "INFEASIBLE", "status field")

    def check_forbidden_constraint_satisfied(self) -> None:
        seq = "G" + "A" * 18 + "C"
        resp = self.fold({"sequence": seq, "forbidden_positions": [19]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["status"], "OPTIMAL", "status field")
        validate_view(body["primary"], seq, 20, [], [19], expected_pairs=0,
                      expected_stacks=0)

    def check_forced_constraint_satisfied(self) -> None:
        seq = "GGAAAACC" + "A" * 12
        resp = self.fold({"sequence": seq, "forced_positions": [0, 6]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        validate_view(body["primary"], seq, 20, [0, 6], [],
                      expected_pairs=2, expected_stacks=1)

    def check_lowercase_normalized(self) -> None:
        resp = self.fold({"sequence": "acgu" + "a" * 16})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["sequence"], "ACGU" + "A" * 16, "normalized sequence echo")

    def check_determinism(self) -> None:
        payload = {"sequence": "CUUAAGGGUUAAGUAAGUGU"}
        first = self.fold(payload).json()
        second = self.fold(payload).json()
        expect_eq(first, second, "identical requests must yield identical verdicts")

    def check_max_length_performance(self) -> None:
        payload = {"sequence": "GGCC" * 60}
        start = time.perf_counter()
        resp = self.fold(payload)
        elapsed = time.perf_counter() - start
        expect_eq(resp.status_code, 200, "status")
        expect(elapsed < 10.0, f"n=240 adjudication took {elapsed:.2f}s (>10s)")
        body = resp.json()
        expect_eq(body["length"], 240, "length")
        validate_view(body["primary"], "GGCC" * 60, 240, [], [],
                      body["primary"]["score"]["pairs"],
                      body["primary"]["score"]["stacks"])
        print(f"\n    n=240 adjudication: {elapsed * 1000:.1f} ms")

    def check_rejections(self) -> None:
        bad_payloads = [
            ("length below minimum", {"sequence": "A" * 19}),
            ("length above maximum", {"sequence": "A" * 241}),
            ("empty sequence", {"sequence": ""}),
            ("illegal base", {"sequence": "N" + "A" * 19}),
            ("dna base T", {"sequence": "T" + "A" * 19}),
            ("position out of range", {"sequence": "A" * 20, "forced_positions": [20]}),
            ("negative position", {"sequence": "A" * 20, "forbidden_positions": [-1]}),
            ("unknown field", {"sequence": "A" * 20, "extra": 1}),
            ("wrong element type", {"sequence": "A" * 20, "forced_positions": ["0"]}),
            ("sequence wrong type", {"sequence": 12345}),
            ("missing sequence", {"forced_positions": []}),
            ("malformed JSON", b"{not json"),
        ]
        for label, payload in bad_payloads:
            if isinstance(payload, bytes):
                resp = self.client.post(
                    FOLD_PATH, content=payload,
                    headers={"content-type": "application/json"},
                )
            else:
                resp = self.fold(payload)
            expect(
                resp.status_code == 422,
                f"{label}: expected 422 rejection, got {resp.status_code} {resp.text[:120]}",
            )

    # -- ensemble counting checks ------------------------------------------

    @staticmethod
    def validate_distribution(dist: dict, total: int) -> None:
        """Enforce the ensemble distribution contract for one position."""
        expect_eq(set(dist.keys()), {"position", "unpaired", "pairs"}, "distribution fields")
        expect(
            isinstance(dist["unpaired"], str) and dist["unpaired"].isdigit(),
            f"unpaired must be a decimal string, got {dist['unpaired']!r}",
        )
        partners = []
        subtotal = int(dist["unpaired"])
        for pair in dist["pairs"]:
            expect_eq(set(pair.keys()), {"position", "count"}, "pair entry fields")
            expect(
                isinstance(pair["count"], str) and pair["count"].isdigit(),
                f"pair count must be a decimal string, got {pair['count']!r}",
            )
            partners.append(pair["position"])
            subtotal += int(pair["count"])
        expect_eq(partners, sorted(partners), "pair entries sorted ascending")
        expect_eq(
            subtotal, total,
            f"distribution conservation for position {dist['position']}",
        )

    def check_ensemble_basic_count(self) -> None:
        seq = "G" + "A" * 18 + "C"
        resp = self.ensemble({"sequence": seq, "positions": [0, 5, 19]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["sequence"], seq, "echoed sequence")
        expect_eq(body["length"], 20, "length")
        expect_eq(body["total"], "2", "total count")
        positions = body["positions"]
        expect_eq([p["position"] for p in positions], [0, 5, 19], "echoed positions")
        # Exactly two structures: all-unpaired and {(0, 19)}.
        expect_eq(positions[0], {"position": 0, "unpaired": "1",
                                 "pairs": [{"position": 19, "count": "1"}]},
                  "distribution for position 0")
        expect_eq(positions[1], {"position": 5, "unpaired": "2", "pairs": []},
                  "distribution for position 5")
        expect_eq(positions[2], {"position": 19, "unpaired": "1",
                                 "pairs": [{"position": 0, "count": "1"}]},
                  "distribution for position 19")

    def check_ensemble_constrained_count(self) -> None:
        # Legal pairs (0,6),(0,7),(1,6),(1,7) give six non-crossing structures.
        seq = "GGAAAACC" + "A" * 12
        resp = self.ensemble({"sequence": seq, "positions": [0, 1, 6, 7]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["total"], "6", "total count")
        by_pos = {p["position"]: p for p in body["positions"]}
        expect_eq(by_pos[0], {"position": 0, "unpaired": "3", "pairs": [
            {"position": 6, "count": "1"}, {"position": 7, "count": "2"}]},
            "distribution for position 0")
        expect_eq(by_pos[7], {"position": 7, "unpaired": "3", "pairs": [
            {"position": 0, "count": "2"}, {"position": 1, "count": "1"}]},
            "distribution for position 7")
        for dist in body["positions"]:
            self.validate_distribution(dist, 6)

    def check_ensemble_forced_constraint(self) -> None:
        seq = "GGAAAACC" + "A" * 12
        resp = self.ensemble({"sequence": seq, "forced_positions": [0],
                              "positions": [0, 1]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        # Only structures with 0 paired: {(0,6)}, {(0,7)}, {(0,7),(1,6)}.
        expect_eq(body["total"], "3", "total count")
        first = body["positions"][0]
        expect_eq(first["unpaired"], "0", "forced position never unpaired")
        expect_eq(first["pairs"], [{"position": 6, "count": "1"},
                                   {"position": 7, "count": "2"}],
                  "forced position partners")
        for dist in body["positions"]:
            self.validate_distribution(dist, 3)

    def check_ensemble_forbidden_constraint(self) -> None:
        seq = "GGAAAACC" + "A" * 12
        resp = self.ensemble({"sequence": seq, "forbidden_positions": [7],
                              "positions": [7, 0]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        # Pairs into 7 removed: {}, {(0,6)}, {(1,6)}.
        expect_eq(body["total"], "3", "total count")
        forbidden, opener = body["positions"]
        expect_eq(forbidden, {"position": 7, "unpaired": "3", "pairs": []},
                  "forbidden position distribution")
        expect_eq(opener["pairs"], [{"position": 6, "count": "1"}],
                  "forbidden partner not listed")
        for dist in body["positions"]:
            self.validate_distribution(dist, 3)

    def check_ensemble_infeasible(self) -> None:
        resp = self.ensemble({"sequence": "A" * 20, "forced_positions": [0],
                              "positions": [0, 5, 19]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["total"], "0", "total count")
        expect_eq(body["positions"], [
            {"position": 0, "unpaired": "0", "pairs": []},
            {"position": 5, "unpaired": "0", "pairs": []},
            {"position": 19, "unpaired": "0", "pairs": []},
        ], "empty distributions on infeasible")

    def check_ensemble_brute_force_crosscheck(self) -> None:
        seq = "AUGCAUGCAUGCAUGCAUGC"
        forced = frozenset({3})
        forbidden = frozenset({16})
        positions = [0, 3, 8, 16, 19]
        structures = enumerate_structures(seq, forced, forbidden)
        expect(0 < len(structures) < 100000, "fixture should stay enumerable")
        resp = self.ensemble({"sequence": seq, "forced_positions": [3],
                              "forbidden_positions": [16], "positions": positions})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        total = len(structures)
        expect_eq(body["total"], str(total), "total vs brute-force enumeration")
        expect(isinstance(body["total"], str), "total must be a decimal string")
        for dist in body["positions"]:
            p = dist["position"]
            exp_unpaired = sum(
                1 for s in structures if all(p not in pair for pair in s)
            )
            expect_eq(int(dist["unpaired"]), exp_unpaired,
                      f"unpaired count for position {p}")
            exp_pairs: dict[int, int] = {}
            for s in structures:
                for a, b in s:
                    if a == p:
                        exp_pairs[b] = exp_pairs.get(b, 0) + 1
                    elif b == p:
                        exp_pairs[a] = exp_pairs.get(a, 0) + 1
            for pair in dist["pairs"]:
                expect_eq(int(pair["count"]), exp_pairs.get(pair["position"], 0),
                          f"pair count ({p},{pair['position']})")
            self.validate_distribution(dist, total)

    def check_ensemble_max_length(self) -> None:
        seq = "GC" * 60
        positions = list(range(12))
        start = time.perf_counter()
        resp = self.ensemble({"sequence": seq, "positions": positions})
        elapsed = time.perf_counter() - start
        expect_eq(resp.status_code, 200, "status")
        expect(elapsed < 10.0, f"n=120 ensemble count took {elapsed:.2f}s (>10s)")
        body = resp.json()
        expect_eq(body["length"], 120, "length")
        total = int(body["total"])
        expect(total > 0, "dense sequence must have legal structures")
        for dist in body["positions"]:
            self.validate_distribution(dist, total)
        print(f"\n    n=120 ensemble count: {elapsed * 1000:.1f} ms "
              f"({len(body['total'])}-digit total)")

    def check_ensemble_determinism(self) -> None:
        payload = {"sequence": "AUGCAUGCAUGCAUGCAUGCAUGC",
                   "forced_positions": [1, 5], "forbidden_positions": [10],
                   "positions": [0, 3, 7, 11]}
        first = self.ensemble(payload).json()
        second = self.ensemble(payload).json()
        expect_eq(first, second, "identical requests must yield identical counts")

    def check_ensemble_rejections(self) -> None:
        bad_payloads = [
            ("length below mode minimum", {"sequence": "A" * 19, "positions": [0]}),
            ("length above mode maximum", {"sequence": "A" * 121, "positions": [0]}),
            ("adjudication-only length", {"sequence": "A" * 240, "positions": [0]}),
            ("illegal base", {"sequence": "N" + "A" * 19, "positions": [0]}),
            ("no positions", {"sequence": "A" * 20, "positions": []}),
            ("positions missing", {"sequence": "A" * 20}),
            ("too many positions",
             {"sequence": "A" * 20, "positions": list(range(13))}),
            ("duplicate positions", {"sequence": "A" * 20, "positions": [3, 7, 3]}),
            ("position out of range", {"sequence": "A" * 20, "positions": [20]}),
            ("negative position", {"sequence": "A" * 20, "positions": [-1]}),
            ("position wrong type", {"sequence": "A" * 20, "positions": ["3"]}),
            ("forced out of range",
             {"sequence": "A" * 20, "positions": [0], "forced_positions": [20]}),
            ("forbidden negative",
             {"sequence": "A" * 20, "positions": [0], "forbidden_positions": [-1]}),
            ("unknown field",
             {"sequence": "A" * 20, "positions": [0], "extra": 1}),
        ]
        for label, payload in bad_payloads:
            resp = self.ensemble(payload)
            expect(
                resp.status_code == 422,
                f"{label}: expected 422 rejection, got {resp.status_code} {resp.text[:120]}",
            )
        resp = self.client.get(ENSEMBLE_PATH)
        expect(resp.status_code == 405,
               f"GET on ensemble endpoint should be 405, got {resp.status_code}")

    def run(self) -> list[CheckResult]:
        checks = [
            ("health endpoint", self.check_health),
            ("versioned routing", self.check_versioned_routing),
            ("unique optimum verdict", self.check_unique_optimum),
            ("structure with no legal pairs", self.check_no_legal_pairs),
            ("wobble pair and multiple optima", self.check_wobble_and_multiple_optima),
            ("stacking as second objective", self.check_stacking_is_second_objective),
            ("lexicographic primary/witness selection", self.check_lexicographic_witness_pair),
            ("infeasible forced pairing", self.check_infeasible_forced),
            ("infeasible forced+forbidden conflict",
             self.check_infeasible_forced_forbidden_conflict),
            ("forbidden constraint satisfied", self.check_forbidden_constraint_satisfied),
            ("forced constraint satisfied", self.check_forced_constraint_satisfied),
            ("lowercase normalization", self.check_lowercase_normalized),
            ("deterministic repeated verdicts", self.check_determinism),
            ("n=240 performance", self.check_max_length_performance),
            ("illegal input never reaches solver", self.check_rejections),
            ("ensemble basic count", self.check_ensemble_basic_count),
            ("ensemble constrained count", self.check_ensemble_constrained_count),
            ("ensemble forced constraint", self.check_ensemble_forced_constraint),
            ("ensemble forbidden constraint", self.check_ensemble_forbidden_constraint),
            ("ensemble infeasible yields zero", self.check_ensemble_infeasible),
            ("ensemble brute-force crosscheck", self.check_ensemble_brute_force_crosscheck),
            ("ensemble n=120 performance", self.check_ensemble_max_length),
            ("ensemble deterministic repeats", self.check_ensemble_determinism),
            ("ensemble illegal input rejected", self.check_ensemble_rejections),
        ]
        results: list[CheckResult] = []
        for name, fn in checks:
            try:
                fn()
            except CheckFailed as exc:
                results.append(CheckResult(name, False, str(exc)))
            except Exception as exc:  # noqa: BLE001 - report any runner error
                results.append(CheckResult(name, False, f"runner error: {exc!r}"))
            else:
                results.append(CheckResult(name, True))
        return results


def main() -> int:
    print(f"RNA folding adjudication acceptance suite")
    print(f"Target: {API_BASE} ({FOLD_PATH}, {ENSEMBLE_PATH})\n")
    acceptance = Acceptance()
    try:
        results = acceptance.run()
    finally:
        acceptance.close()

    width = max(len(r.name) for r in results)
    passed = 0
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        line = f"  [{marker}] {result.name.ljust(width)}"
        if result.detail:
            line += f"  -- {result.detail}"
        print(line)
        passed += result.passed

    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("ACCEPTANCE FAILED")
        return 1
    print("ACCEPTANCE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
