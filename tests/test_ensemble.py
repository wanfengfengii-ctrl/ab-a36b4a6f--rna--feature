"""Tests for exact ensemble counting and the /api/v1/fold/ensemble endpoint.

Solver-level tests cross-check every reported count against exhaustive
enumeration of all legal structures on random short instances.  HTTP tests
cover the response contract, distribution conservation, rejection of illegal
input with 422 before solving, and determinism.
"""

from __future__ import annotations

import random
from functools import lru_cache

from fastapi.testclient import TestClient

from app.ensemble import count_ensemble
from app.main import app
from app.solver import ALLOWED_PAIRS, MIN_PAIR_DISTANCE

client = TestClient(app)

ENSEMBLE_PATH = "/api/v1/fold/ensemble"


def enumerate_structures(sequence: str, forced: frozenset[int], forbidden: frozenset[int]):
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


def brute_distribution(
    sequence: str,
    forced: frozenset[int],
    forbidden: frozenset[int],
    positions: list[int],
):
    """Exact total and per-position distributions by full enumeration."""
    structures = enumerate_structures(sequence, forced, forbidden)
    dists = {}
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


def post(payload):
    return client.post(ENSEMBLE_PATH, json=payload)


# -- solver-level fixtures ---------------------------------------------------

# Legal pairs are G0/G1 with C6/C7; the six structures are the empty one, the
# four singleton pairs, and the nested pair {(0, 7), (1, 6)}.
NESTED_SEQ = "GGAAAACC" + "A" * 12


def test_counts_all_structures_not_just_optimal():
    result = count_ensemble(NESTED_SEQ, positions=[0, 1, 6])
    assert result.total == 6
    by_pos = {d.position: d for d in result.distributions}
    assert by_pos[0].unpaired == 3
    assert by_pos[0].partners == ((6, 1), (7, 2))
    assert by_pos[1].unpaired == 3
    assert by_pos[1].partners == ((6, 2), (7, 1))
    assert by_pos[6].unpaired == 3
    assert by_pos[6].partners == ((0, 1), (1, 2))


def test_forced_position_reduces_ensemble():
    result = count_ensemble(NESTED_SEQ, forced_positions=[0], positions=[0, 1])
    # Only structures pairing position 0 survive: {(0,6)}, {(0,7)}, {(0,7),(1,6)}.
    assert result.total == 3
    by_pos = {d.position: d for d in result.distributions}
    assert by_pos[0].unpaired == 0
    assert by_pos[0].partners == ((6, 1), (7, 2))
    assert by_pos[1].unpaired == 2
    assert by_pos[1].partners == ((6, 1), (7, 0))


def test_forbidden_position_stays_unpaired():
    result = count_ensemble(NESTED_SEQ, forbidden_positions=[6], positions=[6, 0])
    # Pairs into position 6 are gone: {}, {(0,7)}, {(1,7)}.
    assert result.total == 3
    by_pos = {d.position: d for d in result.distributions}
    # A forbidden position has no legal partners at all.
    assert by_pos[6].unpaired == 3
    assert by_pos[6].partners == ()
    assert by_pos[0].partners == ((7, 1),)


def test_infeasible_instance_has_zero_total_and_empty_distributions():
    result = count_ensemble("A" * 20, forced_positions=[0], positions=[0, 5])
    assert result.total == 0
    assert [d.position for d in result.distributions] == [0, 5]
    for dist in result.distributions:
        assert dist.unpaired == 0
        assert dist.partners == ()


def test_no_legal_pairs_single_structure():
    result = count_ensemble("A" * 20, positions=[3])
    assert result.total == 1
    (dist,) = result.distributions
    assert dist.unpaired == 1
    assert dist.partners == ()


def test_partners_sorted_and_conserved():
    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAU"
    result = count_ensemble(seq, positions=[2, 9, 17])
    assert result.total > 0
    for dist in result.distributions:
        qs = [q for q, _ in dist.partners]
        assert qs == sorted(qs)
        assert dist.unpaired + sum(c for _, c in dist.partners) == result.total


def test_ensemble_deterministic_repeat_calls():
    args = ("AUGCAUGCAUGCAUGCAUGCAUGC", {1, 5}, {10}, [0, 3, 11])
    assert count_ensemble(*args) == count_ensemble(*args)


def test_ensemble_matches_brute_force():
    rng = random.Random(20260925)
    for _ in range(300):
        n = rng.randint(5, 15)
        seq = "".join(rng.choice("ACGU") for _ in range(n))
        forced = frozenset(i for i in range(n) if rng.random() < 0.15)
        forbidden = frozenset(i for i in range(n) if rng.random() < 0.15)
        positions = sorted(rng.sample(range(n), rng.randint(1, min(4, n))))
        exp_total, exp_dists = brute_distribution(seq, forced, forbidden, positions)
        result = count_ensemble(seq, forced, forbidden, positions)
        assert result.total == exp_total, (seq, forced, forbidden)
        assert [d.position for d in result.distributions] == positions
        for dist in result.distributions:
            exp_unpaired, exp_counts = exp_dists[dist.position]
            assert dist.unpaired == exp_unpaired, (seq, dist.position)
            nonzero = {q: c for q, c in dist.partners if c != 0}
            assert nonzero == exp_counts, (seq, dist.position)
            assert dist.unpaired + sum(c for _, c in dist.partners) == result.total


# -- HTTP-level tests --------------------------------------------------------


def test_ensemble_response_shape_and_conservation():
    resp = post({"sequence": NESTED_SEQ, "positions": [0, 6]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "FEASIBLE"
    assert body["sequence"] == NESTED_SEQ
    assert body["length"] == 20
    assert body["total"] == "6"
    assert [d["position"] for d in body["positions"]] == [0, 6]
    first = body["positions"][0]
    assert first["unpaired"] == "3"
    assert first["partners"] == [
        {"position": 6, "count": "1"},
        {"position": 7, "count": "2"},
    ]
    for dist in body["positions"]:
        subtotal = int(dist["unpaired"]) + sum(int(p["count"]) for p in dist["partners"])
        assert subtotal == int(body["total"])


def test_ensemble_positions_sorted_canonically():
    payload = {"sequence": NESTED_SEQ, "positions": [6, 0]}
    body = post(payload).json()
    assert [d["position"] for d in body["positions"]] == [0, 6]


def test_ensemble_forced_and_forbidden_constraints():
    resp = post({
        "sequence": NESTED_SEQ,
        "forced_positions": [0],
        "forbidden_positions": [7],
        "positions": [0, 7],
    })
    assert resp.status_code == 200
    body = resp.json()
    # Only {(0, 6)} survives: 0 forced, 7 forbidden.
    assert body["total"] == "1"
    by_pos = {d["position"]: d for d in body["positions"]}
    assert by_pos[0]["unpaired"] == "0"
    assert by_pos[0]["partners"] == [{"position": 6, "count": "1"}]
    assert by_pos[7]["unpaired"] == "1"
    assert by_pos[7]["partners"] == []


def test_ensemble_infeasible_response():
    resp = post({"sequence": "A" * 20, "forced_positions": [0], "positions": [0, 5]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "INFEASIBLE"
    assert body["total"] == "0"
    assert body["positions"] == [
        {"position": 0, "unpaired": "0", "partners": []},
        {"position": 5, "unpaired": "0", "partners": []},
    ]


def test_ensemble_determinism_over_http():
    payload = {"sequence": "CUUAAGGGUUAAGUAAGUGU", "positions": [0, 5, 19]}
    assert post(payload).json() == post(payload).json()


def test_ensemble_max_positions_and_length_accepted():
    seq = ("AUGC" * 30)[:120]
    resp = post({"sequence": seq, "positions": list(range(12))})
    assert resp.status_code == 200
    body = resp.json()
    assert body["length"] == 120
    assert len(body["positions"]) == 12
    for dist in body["positions"]:
        subtotal = int(dist["unpaired"]) + sum(int(p["count"]) for p in dist["partners"])
        assert subtotal == int(body["total"])


def test_ensemble_rejections():
    bad_payloads = [
        ("length below minimum", {"sequence": "A" * 19, "positions": [0]}),
        ("length above mode maximum", {"sequence": "A" * 121, "positions": [0]}),
        ("empty sequence", {"sequence": "", "positions": [0]}),
        ("illegal base", {"sequence": "N" + "A" * 19, "positions": [0]}),
        ("missing positions", {"sequence": "A" * 20}),
        ("empty positions", {"sequence": "A" * 20, "positions": []}),
        ("too many positions", {"sequence": "A" * 20, "positions": list(range(13))}),
        ("duplicate positions", {"sequence": "A" * 20, "positions": [3, 3]}),
        ("position out of range", {"sequence": "A" * 20, "positions": [20]}),
        ("negative position", {"sequence": "A" * 20, "positions": [-1]}),
        ("forced out of range", {"sequence": "A" * 20, "positions": [0],
                                 "forced_positions": [20]}),
        ("forbidden negative", {"sequence": "A" * 20, "positions": [0],
                                "forbidden_positions": [-1]}),
        ("position wrong type", {"sequence": "A" * 20, "positions": ["0"]}),
        ("unknown field", {"sequence": "A" * 20, "positions": [0], "extra": 1}),
        ("sequence wrong type", {"sequence": 123, "positions": [0]}),
    ]
    for label, payload in bad_payloads:
        resp = post(payload)
        assert resp.status_code == 422, (label, resp.status_code, resp.text[:200])


def test_ensemble_length_boundaries():
    assert post({"sequence": "A" * 20, "positions": [0]}).status_code == 200
    assert post({"sequence": "A" * 120, "positions": [0]}).status_code == 200


def test_ensemble_case_normalization():
    resp = post({"sequence": "ggaaaacc" + "a" * 12, "positions": [0]})
    assert resp.status_code == 200
    assert resp.json()["sequence"] == NESTED_SEQ


def test_fold_endpoint_unaffected_by_ensemble_addition():
    resp = client.post("/api/v1/fold", json={"sequence": "CUUAAGGGUUAAGUAAGUGU"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OPTIMAL"
    assert body["primary"]["structure"] == "(((((...)))))(....)."
    assert body["witness"]["structure"] == ".((((...))))((....))"
