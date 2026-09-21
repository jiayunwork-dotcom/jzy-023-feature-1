"""训练通道的 HTTP 验收：真数据恢复区段、训完即可解码、收敛/触顶、
入口带类型拒绝、并发互不写串。"""
from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

TRUE_TRANSITION = [[0.9, 0.1], [0.1, 0.9]]
TRUE_EMISSION = [[0.8, 0.2], [0.2, 0.8]]


def _generate(seed: int, n_sequences: int, length: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n_sequences):
        state = 0 if rng.random() < 0.5 else 1
        obs = []
        for _ in range(length):
            obs.append("a" if rng.random() < TRUE_EMISSION[state][0] else "b")
            state = 0 if rng.random() < TRUE_TRANSITION[state][0] else 1
        out.append("".join(obs))
    return out


def _training_body(name: str, observations, **overrides) -> dict:
    body = {
        "name": name,
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [0.6, 0.4],
        "transition": [[0.6, 0.4], [0.4, 0.6]],
        "emission": [[0.55, 0.45], [0.45, 0.55]],
        "observations": observations,
        "max_iterations": 40,
        "tolerance": 1e-9,
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def observations():
    return _generate(20260921, n_sequences=16, length=80)


def test_training_endpoint_recovers_regimes_and_returns_full_shape(client, observations):
    res = client.post("/training", json=_training_body("trained", observations))
    assert res.status_code == 201
    body = res.get_json()

    # 结果结构完整。
    assert body["model"] == "trained"
    assert body["n_sequences"] == 16
    assert body["iterations"] >= 1
    assert body["stop_reason"] in ("converged", "max_iterations")
    assert isinstance(body["converged"], bool)
    assert body["max_iterations"] == 40
    assert body["tolerance"] == 1e-9
    assert isinstance(body["final_log_likelihood"], float)
    assert len(body["log_likelihoods"]) >= 2
    assert len(body["states"]) == 2 and body["alphabet"] == ["a", "b"]
    for row in body["transition"]:
        assert sum(row) == pytest.approx(1.0)
    for row in body["emission"]:
        assert sum(row) == pytest.approx(1.0)
    assert sum(body["initial"]) == pytest.approx(1.0)

    # 硬标准：逐轮总对数似然单调不减。
    lls = body["log_likelihoods"]
    assert all(lls[i + 1] >= lls[i] for i in range(len(lls) - 1))

    # 两类区段被拉开（状态顺序可能对调）。
    e0, e1 = body["emission"]
    assert (e0[0] > 0.65 and e1[1] > 0.65) or (e0[1] > 0.65 and e1[0] > 0.65)


def test_trained_model_is_listed_and_fetched(client, observations):
    client.post("/training", json=_training_body("listed", observations))
    names = [m["name"] for m in client.get("/models").get_json()["models"]]
    assert "listed" in names
    spec = client.get("/models/listed").get_json()
    assert spec["states"] == ["s0", "s1"]
    assert len(spec["transition"]) == 2


def test_trained_model_can_be_decoded_immediately(client, observations):
    client.post("/training", json=_training_body("decodable", observations))
    res = client.post(
        "/models/decodable/decode",
        json={"observations": "a" * 12 + "b" * 12},
    )
    assert res.status_code == 200
    body = res.get_json()
    path = body["path"]
    assert len(path) == 24
    # 结构完整：路径、后验表、两类对数概率都在。
    assert len(body["posterior"]) == 24
    assert all(sum(row) == pytest.approx(1.0) for row in body["posterior"])
    assert isinstance(body["log_joint_probability"], float)
    assert isinstance(body["log_likelihood"], float)
    # 学到了区段：前半以某一状态为主，后半切到另一状态。
    assert path[0] != path[-1]
    assert path.count(path[0]) >= 10


def test_convergence_is_reported(client, observations):
    # 接近不动点的真参数 + 大容差：首轮增量就低于容差，标 converged。
    body = _training_body(
        "conv",
        observations,
        initial=[0.5, 0.5],
        transition=TRUE_TRANSITION,
        emission=TRUE_EMISSION,
        max_iterations=50,
        tolerance=10.0,
    )
    res = client.post("/training", json=body)
    assert res.status_code == 201
    result = res.get_json()
    assert result["converged"] is True
    assert result["stop_reason"] == "converged"
    assert result["iterations"] >= 1


def test_max_iterations_cap_is_reported(client, observations):
    body = _training_body("capped", observations, max_iterations=3, tolerance=1e-12)
    res = client.post("/training", json=body)
    assert res.status_code == 201
    result = res.get_json()
    assert result["converged"] is False
    assert result["stop_reason"] == "max_iterations"
    assert result["iterations"] == 3


def test_training_accepts_symbol_arrays_and_sparse_alphabet(client):
    # 观测可以是符号数组；没在数据里出现的字母表符号仍是合法发射列。
    body = _training_body(
        "arr",
        [["a", "b"], ["a", "a", "b"]],
        alphabet=["a", "b", "c"],
        emission=[[0.55, 0.4, 0.05], [0.4, 0.55, 0.05]],
        max_iterations=5,
    )
    res = client.post("/training", json=body)
    assert res.status_code == 201
    assert len(res.get_json()["emission"][0]) == 3


def test_duplicate_trained_name_conflicts_like_registration(client, observations):
    client.post("/training", json=_training_body("dup", observations))
    res = client.post("/training", json=_training_body("dup", observations))
    assert res.status_code == 409
    assert res.get_json()["error"]["type"] == "model_exists"
    # 原有档没被覆盖：第二次训练用不同数据，取出来还应是第一次的结果。
    first = client.get("/models/dup").get_json()
    other = _generate(999, 4, 30)
    client.post("/training", json=_training_body("dup", other))
    again = client.get("/models/dup").get_json()
    assert again["emission"] == first["emission"]


def test_name_clash_with_manually_registered_model(client, observations):
    from helpers import two_state_spec

    client.post("/models", json=two_state_spec("manual"))
    res = client.post("/training", json=_training_body("manual", observations))
    assert res.status_code == 409
    assert res.get_json()["error"]["type"] == "model_exists"


# ---------- 开训前带类型拒绝 ----------

@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("observations", [], "empty_training_set"),
        ("observations", ["a", ""], "empty_training_set"),
        ("observations", ["aX"], "unknown_symbol"),
        ("observations", ["a", ["b", "z"]], "unknown_symbol"),
        ("transition", [[0.7, 0.2], [0.4, 0.6]], "validation_error"),
        ("initial", [0.6, 0.5], "validation_error"),
        ("emission", [[0.9, 0.1], [0.5, 0.6]], "validation_error"),
        ("transition", [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]], "validation_error"),
        ("emission", [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]], "validation_error"),
        ("initial", [0.5, 0.3, 0.2], "validation_error"),
        ("max_iterations", 0, "validation_error"),
        ("max_iterations", -1, "validation_error"),
        ("max_iterations", 1.5, "validation_error"),
        ("tolerance", 0.0, "validation_error"),
        ("tolerance", -1e-6, "validation_error"),
        ("tolerance", "x", "validation_error"),
    ],
)
def test_invalid_requests_rejected_before_iteration(client, field, value, error_type):
    body = _training_body("rejected", ["ab"], max_iterations=5)
    body[field] = value
    res = client.post("/training", json=body)
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == error_type


def test_single_state_seed_rejected(client):
    body = _training_body(
        "rejected",
        ["ab"],
        states=["only"],
        initial=[1.0],
        transition=[[1.0]],
        emission=[[0.5, 0.5]],
        max_iterations=5,
    )
    res = client.post("/training", json=body)
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_missing_field_rejected(client):
    res = client.post("/training", json={"name": "x"})
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_missing_name_rejected(client):
    body = _training_body("x", ["ab"])
    del body["name"]
    res = client.post("/training", json=body)
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_non_json_body_rejected(client):
    res = client.post("/training", data="nope", content_type="text/plain")
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "invalid_json"


def test_observations_not_an_array_rejected(client):
    res = client.post("/training", json=_training_body("x", "abab"))
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_dead_state_training_succeeds_without_nan(client):
    # 整批里几乎不出现的第三状态：不崩、对数似然不回头、矩阵无 NaN。
    body = {
        "name": "dead",
        "states": ["s0", "s1", "s2"],
        "alphabet": ["a", "b"],
        "initial": [0.5, 0.5, 0.0],
        "transition": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.3, 0.3, 0.4]],
        "emission": [[0.7, 0.3], [0.3, 0.7], [0.5, 0.5]],
        "observations": ["ab" * 20, "aaaabbbb" * 5],
        "max_iterations": 20,
        "tolerance": 1e-9,
    }
    res = client.post("/training", json=body)
    assert res.status_code == 201
    result = res.get_json()
    lls = result["log_likelihoods"]
    assert all(lls[i + 1] >= lls[i] for i in range(len(lls) - 1))
    for matrix in (result["transition"], result["emission"]):
        assert all(math.isfinite(v) for row in matrix for v in row)
    # 死状态行原样兜底。
    assert result["transition"][2] == [0.3, 0.3, 0.4]


def test_zero_initial_likelihood_reported_not_crashing(client):
    body = {
        "name": "zeroll",
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [1.0, 0.0],
        "transition": [[1.0, 0.0], [0.0, 1.0]],
        "emission": [[1.0, 0.0], [0.0, 1.0]],
        "observations": ["ab"],
        "max_iterations": 10,
    }
    res = client.post("/training", json=body)
    assert res.status_code == 201
    result = res.get_json()
    assert result["iterations"] == 0
    assert result["stop_reason"] == "zero_initial_likelihood"
    assert result["final_log_likelihood"] is None
    # 档仍正式落库，可列可取（拿去解码含 b 的串会走 zero_probability，
    # 但档本身和登记档没有任何区别）。
    assert client.get("/models/zeroll").status_code == 200


# ---------- 并发 ----------

def test_concurrent_training_no_cross_talk(app):
    data_a = _generate(11, 12, 60)
    data_b = _generate(22, 12, 60)

    def run(item):
        idx, (tag, data) = item
        c = app.test_client()
        # 故意用不同的初值与上限，确保中间量不可能互相串。
        body = _training_body(
            f"par-{tag}-{idx}",
            data,
            initial=[0.6, 0.4] if tag == "a" else [0.4, 0.6],
            transition=[[0.6, 0.4], [0.4, 0.6]]
            if tag == "a"
            else [[0.7, 0.3], [0.3, 0.7]],
            emission=[[0.55, 0.45], [0.45, 0.55]],
            max_iterations=10 if tag == "a" else 12,
            tolerance=1e-9,
        )
        res = c.post("/training", json=body)
        assert res.status_code == 201
        return tag, res.get_json()

    pairs = [("a", data_a), ("b", data_b)] * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, enumerate(pairs)))

    # 顺序基准：同一请求重放一次必须得到完全相同的产物。
    ca, cb = app.test_client(), app.test_client()
    base_a = ca.post(
        "/training",
        json=_training_body(
            "base-a", data_a, max_iterations=10, tolerance=1e-9,
        ),
    ).get_json()
    base_b = cb.post(
        "/training",
        json=_training_body(
            "base-b", data_b,
            initial=[0.4, 0.6],
            transition=[[0.7, 0.3], [0.3, 0.7]],
            max_iterations=12, tolerance=1e-9,
        ),
    ).get_json()

    for tag, body in results:
        base = base_a if tag == "a" else base_b
        assert body["log_likelihoods"] == base["log_likelihoods"]
        assert body["emission"] == base["emission"]
        assert body["transition"] == base["transition"]
        assert body["iterations"] == base["iterations"]

    # 两类名字各自落了 8 份档，内容都与该类自己的基准一致。
    names = [m["name"] for m in app.test_client().get("/models").get_json()["models"]]
    par_names = [n for n in names if n.startswith("par-")]
    assert len(par_names) == 16
    assert sum(n.startswith("par-a-") for n in par_names) == 8
    assert sum(n.startswith("par-b-") for n in par_names) == 8


def test_concurrent_same_name_only_one_wins(app, observations):
    # 并发用同一名字训练：恰好一个成功，其余 model_exists，库里只有一份。
    def run(_):
        c = app.test_client()
        return c.post("/training", json=_training_body("race", observations)).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(run, range(8)))
    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    names = [m["name"] for m in app.test_client().get("/models").get_json()["models"]]
    assert names.count("race") == 1
