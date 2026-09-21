"""HTTP 层：登记、列出、按名解码、带类型的错误、内置示范档。"""
import pytest

from helpers import two_state_spec
from hmm_service.app import create_app
from hmm_service.demo import DEMO_MODEL_NAME


def test_register_list_get_roundtrip(client):
    assert client.post("/models", json=two_state_spec("m1")).status_code == 201
    names = [m["name"] for m in client.get("/models").get_json()["models"]]
    assert "m1" in names
    spec = client.get("/models/m1").get_json()
    assert spec["states"] == ["s0", "s1"]
    assert spec["alphabet"] == ["a", "b"]


def test_decode_response_shape(client):
    client.post("/models", json=two_state_spec("m1"))
    res = client.post("/models/m1/decode", json={"observations": "aabba"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["model"] == "m1"
    assert body["length"] == 5
    assert len(body["path"]) == 5
    assert isinstance(body["log_joint_probability"], float)
    assert len(body["posterior"]) == 5
    assert all(len(row) == 2 for row in body["posterior"])
    assert all(sum(row) == pytest.approx(1.0) for row in body["posterior"])
    assert body["tie_break"] == "lowest_state_index"


def test_decode_accepts_symbol_list(client):
    client.post("/models", json=two_state_spec("m1"))
    res = client.post("/models/m1/decode", json={"observations": ["a", "b", "a"]})
    assert res.status_code == 200
    assert res.get_json()["length"] == 3


def test_decode_unregistered_name_rejected(client):
    res = client.post("/models/ghost/decode", json={"observations": "ab"})
    assert res.status_code == 404
    assert res.get_json()["error"]["type"] == "model_not_found"


def test_get_unregistered_name_rejected(client):
    res = client.get("/models/ghost")
    assert res.status_code == 404
    assert res.get_json()["error"]["type"] == "model_not_found"


def test_empty_observations_rejected(client):
    client.post("/models", json=two_state_spec("m1"))
    for empty in ("", []):
        res = client.post("/models/m1/decode", json={"observations": empty})
        assert res.status_code == 400
        assert res.get_json()["error"]["type"] == "empty_observations"


def test_unknown_symbol_rejected(client):
    client.post("/models", json=two_state_spec("m1"))
    res = client.post("/models/m1/decode", json={"observations": "aXb"})
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"]["type"] == "unknown_symbol"
    assert body["error"]["details"]["position"] == 1
    assert body["error"]["details"]["symbol"] == "X"


def test_missing_observations_field_rejected(client):
    client.post("/models", json=two_state_spec("m1"))
    res = client.post("/models/m1/decode", json={})
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_non_json_decode_body_rejected(client):
    client.post("/models", json=two_state_spec("m1"))
    res = client.post("/models/m1/decode", data="ab", content_type="text/plain")
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "invalid_json"


def test_zero_probability_observation_rejected(client):
    client.post("/models", json=two_state_spec(
        "zp",
        initial=[1.0, 0.0],
        transition=[[1.0, 0.0], [0.0, 1.0]],
        emission=[[1.0, 0.0], [0.0, 1.0]],
    ))
    res = client.post("/models/zp/decode", json={"observations": "b"})
    assert res.status_code == 422
    assert res.get_json()["error"]["type"] == "zero_probability"


def test_demo_model_registered_at_startup(client):
    names = [m["name"] for m in client.get("/models").get_json()["models"]]
    assert DEMO_MODEL_NAME in names
    spec = client.get(f"/models/{DEMO_MODEL_NAME}").get_json()
    assert len(spec["states"]) == 3
    assert spec["demo_observation"]


def test_demo_fork_over_http(client):
    # 内置示范档：Viterbi 跟上区段切换，且与逐点后验最大分叉。
    spec = client.get(f"/models/{DEMO_MODEL_NAME}").get_json()
    res = client.post(
        f"/models/{DEMO_MODEL_NAME}/decode",
        json={"observations": spec["demo_observation"]},
    )
    assert res.status_code == 200
    body = res.get_json()
    path = body["path"]
    assert path[5] == "A" and path[25] == "B" and path[55] == "A"
    assert "C" not in path
    states = spec["states"]
    argmax_path = [
        states[max(range(len(states)), key=lambda i: row[i])]
        for row in body["posterior"]
    ]
    assert argmax_path != path
    assert argmax_path[-1] == "C"


def test_models_persist_in_sqlite_file(tmp_path):
    db = str(tmp_path / "models.db")
    app1 = create_app(db_path=db, with_demo=False)
    app1.test_client().post("/models", json=two_state_spec("persisted"))
    app2 = create_app(db_path=db, with_demo=False)
    names = [m["name"] for m in app2.test_client().get("/models").get_json()["models"]]
    assert "persisted" in names


def test_unknown_route_returns_typed_error(client):
    res = client.get("/no-such-route")
    assert res.status_code == 404
    assert res.get_json()["error"]["type"] == "http_error"
