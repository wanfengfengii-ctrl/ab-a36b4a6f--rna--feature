"""HTTP-level tests for the versioned folding adjudication endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post(payload):
    return client.post("/api/v1/fold", json=payload)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unique_optimum_response_shape():
    resp = post({"sequence": "G" + "A" * 18 + "C"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OPTIMAL"
    assert body["length"] == 20
    assert body["unique"] is True
    assert body["witness"] is None
    primary = body["primary"]
    assert primary["structure"] == "(..................)"
    assert primary["pairs"] == [[0, 19]]
    assert primary["score"] == {"pairs": 1, "stacks": 0}


def test_multiple_optima_include_witness():
    resp = post({"sequence": "CUUAAGGGUUAAGUAAGUGU"})
    body = resp.json()
    assert body["status"] == "OPTIMAL"
    assert body["unique"] is False
    assert body["primary"]["structure"] == "(((((...)))))(....)."
    assert body["witness"]["structure"] == ".((((...))))((....))"
    assert body["primary"]["score"] == body["witness"]["score"] == {
        "pairs": 6,
        "stacks": 4,
    }


def test_infeasible_response():
    resp = post({"sequence": "A" * 20, "forced_positions": [0]})
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "INFEASIBLE"
    assert body["primary"] is None
    assert body["witness"] is None


def test_sequence_length_bounds_rejected():
    for bad in ("A" * 19, "A" * 241, ""):
        resp = post({"sequence": bad})
        assert resp.status_code == 422, bad


def test_length_boundaries_accepted():
    assert post({"sequence": "A" * 20}).status_code == 200
    assert post({"sequence": "A" * 240}).status_code == 200


def test_illegal_alphabet_rejected():
    assert post({"sequence": "N" + "A" * 19}).status_code == 422
    assert post({"sequence": "T" + "A" * 19}).status_code == 422


def test_out_of_range_positions_rejected():
    assert post({"sequence": "A" * 20, "forced_positions": [20]}).status_code == 422
    assert post({"sequence": "A" * 20, "forbidden_positions": [-1]}).status_code == 422


def test_extra_field_and_bad_types_rejected():
    assert post({"sequence": "A" * 20, "bogus": 1}).status_code == 422
    assert post({"sequence": 123}).status_code == 422
    assert post({"sequence": "A" * 20, "forced_positions": ["0"]}).status_code == 422
    assert post({"forced_positions": []}).status_code == 422


def test_case_normalization():
    resp = post({"sequence": "acgu" + "a" * 16})
    assert resp.status_code == 200
    assert resp.json()["sequence"] == "ACGU" + "A" * 16


def test_constraints_accepted():
    resp = post({
        "sequence": "G" + "A" * 18 + "C",
        "forced_positions": [0, 19],
        "forbidden_positions": [5],
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "OPTIMAL"
