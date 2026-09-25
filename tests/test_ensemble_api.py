"""HTTP-level tests for the ensemble counting endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENSEMBLE_PATH = "/api/v1/fold/ensemble"


def post(payload):
    return client.post(ENSEMBLE_PATH, json=payload)


def test_basic_count_response_shape():
    seq = "G" + "A" * 18 + "C"
    resp = post({"sequence": seq, "positions": [0, 5, 19]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sequence"] == seq
    assert body["length"] == 20
    assert body["total"] == "2"
    positions = body["positions"]
    assert [p["position"] for p in positions] == [0, 5, 19]
    # Structures: all-unpaired and {(0, 19)}.
    assert positions[0] == {
        "position": 0,
        "unpaired": "1",
        "pairs": [{"position": 19, "count": "1"}],
    }
    assert positions[1] == {"position": 5, "unpaired": "2", "pairs": []}
    assert positions[2] == {
        "position": 19,
        "unpaired": "1",
        "pairs": [{"position": 0, "count": "1"}],
    }


def test_counts_are_decimal_strings():
    resp = post({"sequence": "GGAAAACC" + "A" * 12, "positions": [0, 1, 6, 7]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == "6"
    assert isinstance(body["total"], str)
    for dist in body["positions"]:
        assert isinstance(dist["unpaired"], str)
        assert dist["unpaired"].isdigit()
        for pair in dist["pairs"]:
            assert isinstance(pair["count"], str)
            assert pair["count"].isdigit()


def test_distribution_conservation_and_ordering():
    seq = "CGAUGCAUGCGCUAGCUAGCAUCGAUCGAUGCUAGCUAGCUA" + "GC" * 39
    assert len(seq) == 120
    positions = [0, 7, 15, 23, 31, 40, 55, 63, 77, 88, 99, 110]
    resp = post({"sequence": seq, "positions": positions})
    assert resp.status_code == 200
    body = resp.json()
    total = int(body["total"])
    assert total > 0
    assert [p["position"] for p in body["positions"]] == positions
    for dist in body["positions"]:
        partners = [pair["position"] for pair in dist["pairs"]]
        assert partners == sorted(partners)
        subtotal = int(dist["unpaired"]) + sum(
            int(pair["count"]) for pair in dist["pairs"]
        )
        assert subtotal == total


def test_forced_position_unpaired_count_is_zero():
    seq = "GGAAAACC" + "A" * 12
    resp = post({"sequence": seq, "forced_positions": [0], "positions": [0, 1]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == "3"  # {(0,6)}, {(0,7)}, {(0,7),(1,6)}
    first, second = body["positions"]
    assert first["unpaired"] == "0"
    assert first["pairs"] == [
        {"position": 6, "count": "1"},
        {"position": 7, "count": "2"},
    ]
    assert int(second["unpaired"]) + sum(
        int(p["count"]) for p in second["pairs"]
    ) == 3


def test_forbidden_position_never_pairs():
    seq = "GGAAAACC" + "A" * 12
    resp = post({"sequence": seq, "forbidden_positions": [7], "positions": [7, 0]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == "3"  # {}, {(0,6)}, {(1,6)}
    forbidden, opener = body["positions"]
    assert forbidden == {"position": 7, "unpaired": "3", "pairs": []}
    assert opener["pairs"] == [{"position": 6, "count": "1"}]


def test_infeasible_returns_zero_total_and_empty_distributions():
    resp = post({"sequence": "A" * 20, "forced_positions": [0], "positions": [0, 5]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == "0"
    assert body["positions"] == [
        {"position": 0, "unpaired": "0", "pairs": []},
        {"position": 5, "unpaired": "0", "pairs": []},
    ]


def test_deterministic_repeated_requests():
    payload = {
        "sequence": "AUGCAUGCAUGCAUGCAUGCAUGC",
        "forced_positions": [1, 5],
        "forbidden_positions": [10],
        "positions": [0, 3, 7, 11],
    }
    assert post(payload).json() == post(payload).json()


def test_case_normalization():
    resp = post({"sequence": "g" + "a" * 18 + "c", "positions": [0]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sequence"] == "G" + "A" * 18 + "C"
    assert body["total"] == "2"


def test_length_bounds_for_this_mode():
    # Ensemble mode accepts 20-120 only; 19/121 are rejected even though
    # the plain adjudication endpoint accepts up to 240.
    assert post({"sequence": "A" * 19, "positions": [0]}).status_code == 422
    assert post({"sequence": "A" * 121, "positions": [0]}).status_code == 422
    assert post({"sequence": "A" * 240, "positions": [0]}).status_code == 422
    assert post({"sequence": "A" * 20, "positions": [0]}).status_code == 200
    assert post({"sequence": "A" * 120, "positions": [0]}).status_code == 200


def test_positions_cardinality_bounds():
    assert post({"sequence": "A" * 20, "positions": []}).status_code == 422
    assert (
        post({"sequence": "A" * 20, "positions": list(range(13))}).status_code
        == 422
    )
    assert (
        post({"sequence": "A" * 20, "positions": list(range(12))}).status_code
        == 200
    )
    assert post({"sequence": "A" * 20}).status_code == 422  # positions required


def test_duplicate_positions_rejected():
    resp = post({"sequence": "A" * 20, "positions": [3, 7, 3]})
    assert resp.status_code == 422


def test_illegal_positions_rejected():
    assert (
        post({"sequence": "A" * 20, "positions": [20]}).status_code == 422
    )
    assert (
        post({"sequence": "A" * 20, "positions": [-1]}).status_code == 422
    )
    assert (
        post({"sequence": "A" * 20, "positions": ["3"]}).status_code == 422
    )
    assert (
        post({"sequence": "A" * 20, "positions": [0], "forced_positions": [20]}).status_code
        == 422
    )
    assert (
        post({"sequence": "A" * 20, "positions": [0], "forbidden_positions": [-1]}).status_code
        == 422
    )


def test_other_malformed_input_rejected():
    assert post({"sequence": "N" + "A" * 19, "positions": [0]}).status_code == 422
    assert post({"sequence": 123, "positions": [0]}).status_code == 422
    assert (
        post({"sequence": "A" * 20, "positions": [0], "bogus": 1}).status_code
        == 422
    )


def test_fold_endpoint_unchanged_by_ensemble_mode():
    # Regression: the adjudication endpoint keeps its own 20-240 range.
    resp = client.post("/api/v1/fold", json={"sequence": "A" * 240})
    assert resp.status_code == 200
    resp = client.post("/api/v1/fold", json={"sequence": "G" + "A" * 18 + "C"})
    body = resp.json()
    assert body["status"] == "OPTIMAL"
    assert body["primary"]["structure"] == "(..................)"
