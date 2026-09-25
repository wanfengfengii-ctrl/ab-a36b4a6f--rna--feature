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
* exact ensemble counting over all legal structures, checked against
  independent brute-force enumeration, with forced/forbidden constraints;
* per-position distribution conservation (parts always sum to the total);
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


def brute_ensemble(
    sequence: str,
    forced: set[int],
    forbidden: set[int],
    positions: list[int],
) -> tuple[int, dict[int, tuple[int, dict[int, int]]]]:
    """Independently enumerate all legal structures; tally per-position counts."""
    n = len(sequence)
    partners: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        if i in forbidden:
            continue
        for r in range(i + MIN_PAIR_DISTANCE, n):
            if r not in forbidden and (sequence[i], sequence[r]) in ALLOWED_PAIRS:
                partners[i].append(r)

    @lru_cache(maxsize=None)
    def gen(i: int, j: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        if i >= j:
            return ((),)
        out: list[tuple[tuple[int, int], ...]] = []
        if i not in forced:
            out.extend(gen(i + 1, j))
        for r in partners[i]:
            if r < j:
                for inside in gen(i + 1, r):
                    for outside in gen(r + 1, j):
                        out.append(inside + outside + ((i, r),))
        return tuple(out)

    structures = gen(0, n)
    dists: dict[int, tuple[int, dict[int, int]]] = {}
    for p in positions:
        unpaired = 0
        counts: dict[int, int] = {}
        for structure in structures:
            mate = None
            for a, b in structure:
                if a == p:
                    mate = b
                    break
                if b == p:
                    mate = a
                    break
            if mate is None:
                unpaired += 1
            else:
                counts[mate] = counts.get(mate, 0) + 1
        dists[p] = (unpaired, counts)
    return len(structures), dists


def is_decimal(value: object) -> bool:
    """True iff the value is a non-empty ASCII decimal string."""
    return isinstance(value, str) and bool(value) and all(c in "0123456789" for c in value)


def validate_distribution(dist: dict, total: str, length: int) -> None:
    """Validate one per-position distribution, including conservation."""
    expect_eq(set(dist.keys()), {"position", "unpaired", "partners"}, "distribution fields")
    position = dist["position"]
    expect(isinstance(position, int) and 0 <= position < length,
           "review position must be an in-range integer")
    expect(is_decimal(dist["unpaired"]), "unpaired must be a decimal string")
    partner_positions: list[int] = []
    subtotal = int(dist["unpaired"])
    for entry in dist["partners"]:
        expect_eq(set(entry.keys()), {"position", "count"}, "partner entry fields")
        expect(isinstance(entry["position"], int) and 0 <= entry["position"] < length,
               "partner position must be an in-range integer")
        expect(is_decimal(entry["count"]), "partner count must be a decimal string")
        partner_positions.append(entry["position"])
        subtotal += int(entry["count"])
    expect(partner_positions == sorted(partner_positions),
           "partners must be sorted by position ascending")
    expect_eq(subtotal, int(total), "unpaired + partner counts must equal the total")


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

    def check_ensemble_exact_counting(self) -> None:
        seq = "CUUAAGGGUUAAGUAAGUGU"
        positions = [0, 5, 11, 19]
        resp = self.ensemble({"sequence": seq, "positions": positions})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["status"], "FEASIBLE", "status field")
        expect_eq(body["sequence"], seq, "echoed sequence")
        expect_eq(body["length"], 20, "length")
        total, dists = brute_ensemble(seq, set(), set(), positions)
        expect_eq(total, 4076, "fixture total (independent enumeration)")
        expect_eq(body["total"], str(total), "ensemble total")
        expect(is_decimal(body["total"]), "total must be a decimal string")
        expect_eq([d["position"] for d in body["positions"]], positions,
                  "one distribution per requested position, ascending")
        for dist in body["positions"]:
            validate_distribution(dist, body["total"], 20)
            unpaired, counts = dists[dist["position"]]
            expect_eq(dist["unpaired"], str(unpaired), "unpaired count")
            got = {e["position"]: e["count"] for e in dist["partners"]
                   if e["count"] != "0"}
            expect_eq(got, {q: str(c) for q, c in counts.items()}, "partner counts")

    def check_ensemble_constrained_counting(self) -> None:
        seq = "CUUAAGGGUUAAGUAAGUGU"
        positions = [0, 7, 19]
        resp = self.ensemble({
            "sequence": seq,
            "positions": positions,
            "forced_positions": [0],
            "forbidden_positions": [19],
        })
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        total, dists = brute_ensemble(seq, {0}, {19}, positions)
        expect(0 < total < 4076, "constraints must strictly shrink the ensemble")
        expect_eq(body["total"], str(total), "constrained ensemble total")
        by_pos = {d["position"]: d for d in body["positions"]}
        # The forced position is paired in every counted structure.
        expect_eq(by_pos[0]["unpaired"], "0", "forced position never unpaired")
        # The forbidden position is unpaired everywhere and has no partners.
        expect_eq(by_pos[19]["unpaired"], str(total), "forbidden position always unpaired")
        expect_eq(by_pos[19]["partners"], [], "forbidden position has no partners")
        for dist in body["positions"]:
            validate_distribution(dist, body["total"], 20)
            unpaired, counts = dists[dist["position"]]
            expect_eq(dist["unpaired"], str(unpaired), "unpaired count")
            got = {e["position"]: e["count"] for e in dist["partners"]
                   if e["count"] != "0"}
            expect_eq(got, {q: str(c) for q, c in counts.items()}, "partner counts")

    def check_ensemble_conservation_large(self) -> None:
        seq = "AUGC" * 30  # n = 120, the maximum for this mode
        positions = list(range(12))
        start = time.perf_counter()
        resp = self.ensemble({"sequence": seq, "positions": positions})
        elapsed = time.perf_counter() - start
        expect_eq(resp.status_code, 200, "status")
        expect(elapsed < 10.0, f"n=120 ensemble took {elapsed:.2f}s (>10s)")
        body = resp.json()
        expect_eq(body["length"], 120, "length")
        expect(is_decimal(body["total"]), "total must be a decimal string")
        expect(int(body["total"]) > 0, "dense instance should have many structures")
        expect_eq([d["position"] for d in body["positions"]], positions,
                  "one distribution per requested position, ascending")
        for dist in body["positions"]:
            validate_distribution(dist, body["total"], 120)
        print(f"\n    n=120 ensemble counting: {elapsed * 1000:.1f} ms")

    def check_ensemble_infeasible(self) -> None:
        resp = self.ensemble({"sequence": "A" * 20, "forced_positions": [0],
                              "positions": [0, 5]})
        expect_eq(resp.status_code, 200, "status")
        body = resp.json()
        expect_eq(body["status"], "INFEASIBLE", "status field")
        expect_eq(body["total"], "0", "zero total")
        expect_eq(body["positions"], [
            {"position": 0, "unpaired": "0", "partners": []},
            {"position": 5, "unpaired": "0", "partners": []},
        ], "empty distributions")

    def check_ensemble_determinism(self) -> None:
        payload = {"sequence": "CUUAAGGGUUAAGUAAGUGU", "positions": [0, 5, 19],
                   "forced_positions": [0]}
        first = self.ensemble(payload).json()
        second = self.ensemble(payload).json()
        expect_eq(first, second, "identical requests must yield identical ensembles")

    def check_ensemble_fold_feasibility_consistency(self) -> None:
        payloads = [
            {"sequence": "CUUAAGGGUUAAGUAAGUGU", "positions": [0]},
            {"sequence": "A" * 20, "forced_positions": [0], "positions": [0]},
        ]
        for payload in payloads:
            fold_payload = {k: v for k, v in payload.items() if k != "positions"}
            fold_status = self.fold(fold_payload).json()["status"]
            ensemble_status = self.ensemble(payload).json()["status"]
            expect_eq(
                ensemble_status,
                "FEASIBLE" if fold_status == "OPTIMAL" else "INFEASIBLE",
                "fold/ensemble feasibility agreement",
            )

    def check_ensemble_rejections(self) -> None:
        bad_payloads = [
            ("length below minimum", {"sequence": "A" * 19, "positions": [0]}),
            ("length above mode maximum", {"sequence": "A" * 121, "positions": [0]}),
            ("empty sequence", {"sequence": "", "positions": [0]}),
            ("illegal base", {"sequence": "N" + "A" * 19, "positions": [0]}),
            ("missing positions", {"sequence": "A" * 20}),
            ("empty positions", {"sequence": "A" * 20, "positions": []}),
            ("too many positions", {"sequence": "A" * 20, "positions": list(range(13))}),
            ("duplicate review positions", {"sequence": "A" * 20, "positions": [3, 3]}),
            ("review position out of range", {"sequence": "A" * 20, "positions": [20]}),
            ("negative review position", {"sequence": "A" * 20, "positions": [-1]}),
            ("forced out of range", {"sequence": "A" * 20, "positions": [0],
                                     "forced_positions": [20]}),
            ("forbidden negative", {"sequence": "A" * 20, "positions": [0],
                                    "forbidden_positions": [-1]}),
            ("review position wrong type", {"sequence": "A" * 20, "positions": ["0"]}),
            ("unknown field", {"sequence": "A" * 20, "positions": [0], "extra": 1}),
            ("sequence wrong type", {"sequence": 123, "positions": [0]}),
        ]
        for label, payload in bad_payloads:
            resp = self.ensemble(payload)
            expect(
                resp.status_code == 422,
                f"{label}: expected 422 rejection, got {resp.status_code} {resp.text[:120]}",
            )

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
            ("ensemble exact counting vs enumeration", self.check_ensemble_exact_counting),
            ("ensemble forced/forbidden counting", self.check_ensemble_constrained_counting),
            ("ensemble distribution conservation (n=120)",
             self.check_ensemble_conservation_large),
            ("ensemble infeasible has zero total", self.check_ensemble_infeasible),
            ("ensemble deterministic repeats", self.check_ensemble_determinism),
            ("fold/ensemble feasibility agreement",
             self.check_ensemble_fold_feasibility_consistency),
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
    print(f"Target: {API_BASE}{FOLD_PATH}\n")
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
