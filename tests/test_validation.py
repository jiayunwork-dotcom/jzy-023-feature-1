"""登记期校验：缺项、维数、归一化、钉死的容差。"""
import pytest

from helpers import two_state_spec
from hmm_service.validation import TOLERANCE


def register(client, spec):
    return client.post("/models", json=spec)


def test_valid_spec_registered(client):
    res = register(client, two_state_spec("ok"))
    assert res.status_code == 201
    assert res.get_json()["name"] == "ok"


def test_missing_field_rejected(client):
    spec = two_state_spec("bad")
    del spec["transition"]
    res = register(client, spec)
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"]["type"] == "validation_error"
    assert any("transition" in d for d in body["error"]["details"])


def test_single_state_rejected(client):
    spec = two_state_spec("bad", states=["only"], initial=[1.0],
                          transition=[[1.0]], emission=[[0.5, 0.5]])
    res = register(client, spec)
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_transition_row_sum_off_rejected(client):
    # 行和明显偏离 1，登记时就拒绝。
    res = register(client, two_state_spec("bad", transition=[[0.7, 0.2], [0.4, 0.6]]))
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"]["type"] == "validation_error"
    assert any("transition" in d for d in body["error"]["details"])


def test_emission_row_sum_off_rejected(client):
    res = register(client, two_state_spec("bad", emission=[[0.9, 0.1], [0.5, 0.6]]))
    assert res.status_code == 400
    assert any("emission" in d for d in res.get_json()["error"]["details"])


def test_initial_sum_off_rejected(client):
    res = register(client, two_state_spec("bad", initial=[0.6, 0.5]))
    assert res.status_code == 400
    assert any("initial" in d for d in res.get_json()["error"]["details"])


def test_transition_dimension_mismatch_rejected(client):
    # 状态数 2，却给了 3x3 转移矩阵。
    res = register(client, two_state_spec(
        "bad",
        transition=[[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]],
    ))
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_emission_dimension_mismatch_rejected(client):
    # 字母表长度 2，发射行却给了 3 列。
    res = register(client, two_state_spec(
        "bad",
        emission=[[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]],
    ))
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_initial_length_mismatch_rejected(client):
    res = register(client, two_state_spec("bad", initial=[0.5, 0.3, 0.2]))
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_probability_out_of_range_rejected(client):
    res = register(client, two_state_spec("bad", initial=[1.2, -0.2]))
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_duplicate_state_names_rejected(client):
    res = register(client, two_state_spec("bad", states=["s0", "s0"]))
    assert res.status_code == 400


def test_non_single_char_alphabet_rejected(client):
    res = register(client, two_state_spec("bad", alphabet=["ab", "c"]))
    assert res.status_code == 400


def test_tolerance_is_pinned(client):
    # 容差钉死为 1e-6：偏离 5e-7 放行，偏离 5e-6 拒绝。
    assert TOLERANCE == 1e-6
    within = two_state_spec("within", initial=[0.5, 0.5 + 5e-7])
    assert register(client, within).status_code == 201
    beyond = two_state_spec("beyond", initial=[0.5, 0.5 + 5e-6])
    assert register(client, beyond).status_code == 400


def test_non_json_body_rejected(client):
    res = client.post("/models", data="not json", content_type="text/plain")
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "invalid_json"


def test_duplicate_name_rejected(client):
    assert register(client, two_state_spec("dup")).status_code == 201
    res = register(client, two_state_spec("dup"))
    assert res.status_code == 409
    assert res.get_json()["error"]["type"] == "model_exists"
