"""训练通道：EM 单调性、区段分离、收敛/触顶、零计数兜底、入口拒绝、并发。"""
import math
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from helpers import two_state_spec

# 「真」模型：两个粘性区段状态，分别偏好 a / b，区段切换明显。
TRUE_MODEL = {
    "states": ["s0", "s1"],
    "alphabet": ["a", "b"],
    "initial": [0.5, 0.5],
    "transition": [[0.95, 0.05], [0.05, 0.95]],
    "emission": [[0.9, 0.1], [0.1, 0.9]],
}

# 另一份差异明显的「真」模型，供并发测试区分两个任务的产物。
TRUE_MODEL_B = {
    "states": ["s0", "s1"],
    "alphabet": ["a", "b"],
    "initial": [0.5, 0.5],
    "transition": [[0.9, 0.1], [0.1, 0.9]],
    "emission": [[0.7, 0.3], [0.3, 0.7]],
}


def simulate(spec, lengths, seed):
    """按给定模型祖先采样观测串（固定种子，确定性）。"""
    rng = random.Random(seed)

    def draw(dist):
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(dist):
            acc += p
            if r < acc:
                return i
        return len(dist) - 1

    sequences = []
    for length in lengths:
        state = draw(spec["initial"])
        obs = []
        for _ in range(length):
            obs.append(spec["alphabet"][draw(spec["emission"][state])])
            state = draw(spec["transition"][state])
        sequences.append("".join(obs))
    return sequences


def train_payload(name, sequences, **overrides):
    """一份合法的训练请求：初值刻意模糊（接近均匀但不在对称鞍点上）。"""
    payload = {
        "name": name,
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [0.5, 0.5],
        "transition": [[0.8, 0.2], [0.2, 0.8]],
        "emission": [[0.6, 0.4], [0.45, 0.55]],
        "observations": sequences,
    }
    payload.update(overrides)
    return payload


def assert_monotonic(log_likelihoods):
    """总对数似然逐轮单调不减（留浮点噪声余量）。"""
    for prev, curr in zip(log_likelihoods, log_likelihoods[1:]):
        assert curr >= prev - 1e-9


@pytest.fixture(scope="module")
def training_sequences():
    # 由「真」模型生成、带明显区段切换的一批观测。
    return simulate(TRUE_MODEL, [60] * 24, seed=20240901)


def test_training_monotonic_and_separates_regimes(client, training_sequences):
    res = client.post(
        "/models/train",
        json=train_payload(
            "learned", training_sequences, max_iterations=200, tolerance=1e-7
        ),
    )
    assert res.status_code == 201
    body = res.get_json()
    lls = body["log_likelihoods"]
    assert len(lls) == body["iterations"] + 1
    assert body["iterations"] > 1
    assert_monotonic(lls)
    assert lls[-1] > lls[0]  # 训练确实把似然推上去了

    # 估出的发射分布把两类区段区分开：一个状态倒向 a，另一个倒向 b
    # （与状态标签次序无关，按 P(a) 排序后比较）。
    model = body["model"]
    p_of_a = sorted(row[0] for row in model["emission"])
    assert p_of_a[0] < 0.3
    assert p_of_a[1] > 0.7

    # 每一行严格满足归一约束（与登记同一套容差）。
    assert sum(model["initial"]) == pytest.approx(1.0)
    for row in model["transition"] + model["emission"]:
        assert sum(row) == pytest.approx(1.0)


def test_long_sequences_no_underflow(client):
    # 长串 + 多序列：期望量在对数域累加，不塌成全零、不产出退化矩阵。
    sequences = [("a" * 50 + "b" * 50) * 20, "ab" * 1000]
    res = client.post(
        "/models/train",
        json=train_payload("long", sequences, max_iterations=30),
    )
    assert res.status_code == 201
    body = res.get_json()
    assert all(math.isfinite(v) for v in body["log_likelihoods"])
    assert_monotonic(body["log_likelihoods"])
    model = body["model"]
    for row in model["transition"] + model["emission"]:
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in row)
        assert sum(row) == pytest.approx(1.0)
    # 没有退化成全零或全均匀：两类区段被区分开。
    p_of_a = sorted(row[0] for row in model["emission"])
    assert p_of_a[1] - p_of_a[0] > 0.3


def test_trained_model_lists_gets_and_decodes(client, training_sequences):
    res = client.post("/models/train", json=train_payload("usable", training_sequences))
    assert res.status_code == 201

    # 训练出的档与手工登记的档在后续使用上没有任何区别。
    names = [m["name"] for m in client.get("/models").get_json()["models"]]
    assert "usable" in names
    spec = client.get("/models/usable").get_json()
    assert spec["states"] == ["s0", "s1"]
    assert spec["alphabet"] == ["a", "b"]

    dec = client.post(
        "/models/usable/decode", json={"observations": training_sequences[0]}
    )
    assert dec.status_code == 200
    body = dec.get_json()
    assert body["model"] == "usable"
    assert len(body["path"]) == len(training_sequences[0])
    assert all(s in ("s0", "s1") for s in body["path"])
    assert len(body["posterior"]) == len(training_sequences[0])
    assert all(sum(row) == pytest.approx(1.0) for row in body["posterior"])
    assert isinstance(body["log_joint_probability"], float)
    assert isinstance(body["log_likelihood"], float)
    assert body["tie_break"] == "lowest_state_index"


def test_converges_before_max_iterations(client, training_sequences):
    res = client.post(
        "/models/train",
        json=train_payload(
            "conv", training_sequences, max_iterations=500, tolerance=1e-4
        ),
    )
    assert res.status_code == 201
    body = res.get_json()
    assert body["converged"] is True
    assert body["stop_reason"] == "converged"
    assert body["iterations"] < 500
    lls = body["log_likelihoods"]
    # 停下的那一轮，相邻两轮增量确实小于容差。
    assert lls[-1] - lls[-2] < 1e-4
    assert body["log_likelihood"] == lls[-1]


def test_hits_max_iterations(client, training_sequences):
    res = client.post(
        "/models/train",
        json=train_payload(
            "capped", training_sequences, max_iterations=3, tolerance=1e-12
        ),
    )
    assert res.status_code == 201
    body = res.get_json()
    assert body["converged"] is False
    assert body["stop_reason"] == "max_iterations"
    assert body["iterations"] == 3
    assert len(body["log_likelihoods"]) == 4  # 第 0 轮（初值）+ 3 轮重估
    assert_monotonic(body["log_likelihoods"])


def test_unreachable_state_falls_back_without_regress(client, training_sequences):
    # 第三个状态恒不可达（初值 0、任何转移都到不了它），期望次数恒为零，
    # 重估走兜底：退回上一轮的行，不崩、不产生 NaN、似然不回头。
    res = client.post(
        "/models/train",
        json=train_payload(
            "ghost-state",
            training_sequences,
            states=["s0", "s1", "ghost"],
            initial=[0.5, 0.5, 0.0],
            transition=[
                [0.8, 0.2, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            ],
            emission=[
                [0.6, 0.4],
                [0.45, 0.55],
                [0.5, 0.5],
            ],
            max_iterations=50,
        ),
    )
    assert res.status_code == 201
    body = res.get_json()
    assert_monotonic(body["log_likelihoods"])

    model = body["model"]
    for row in model["transition"] + model["emission"] + [model["initial"]]:
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in row)
    assert sum(model["initial"]) == pytest.approx(1.0)
    for row in model["transition"] + model["emission"]:
        assert sum(row) == pytest.approx(1.0)
    # 兜底退回上一轮的行：ghost 行保持初值不变。
    assert model["transition"][2] == pytest.approx([0.0, 0.0, 1.0])
    assert model["emission"][2] == pytest.approx([0.5, 0.5])
    assert model["initial"][2] == 0.0


def test_extreme_initial_guess_settles_without_nan(client):
    # 初值离谱：s0 几乎只发 a、s1 几乎只发 b 且转移近乎锁死，
    # 迭代要么平稳爬升要么停在局部解，不抛异常、不出 NaN。
    sequences = ["ab" * 20, "ba" * 20, "a" * 15 + "b" * 15]
    res = client.post(
        "/models/train",
        json=train_payload(
            "wild",
            sequences,
            initial=[0.999, 0.001],
            transition=[[0.999, 0.001], [0.5, 0.5]],
            emission=[[0.999, 0.001], [0.001, 0.999]],
            max_iterations=200,
        ),
    )
    assert res.status_code == 201
    body = res.get_json()
    assert all(math.isfinite(v) for v in body["log_likelihoods"])
    assert_monotonic(body["log_likelihoods"])
    model = body["model"]
    for row in model["transition"] + model["emission"] + [model["initial"]]:
        assert all(math.isfinite(v) for v in row)


def test_impossible_under_initial_guess_rejected(client):
    # 初值下观测串概率为 0：带类型拒绝，不是未预期的崩溃。
    res = client.post(
        "/models/train",
        json=train_payload(
            "impossible",
            ["b"],
            initial=[1.0, 0.0],
            transition=[[1.0, 0.0], [0.0, 1.0]],
            emission=[[1.0, 0.0], [0.0, 1.0]],
        ),
    )
    assert res.status_code == 422
    assert res.get_json()["error"]["type"] == "zero_probability"


def test_unknown_symbol_in_training_set_rejected(client):
    res = client.post(
        "/models/train", json=train_payload("bad", ["aab", "aXb", "bba"])
    )
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"]["type"] == "unknown_symbol"
    assert body["error"]["details"]["sequence"] == 1
    assert body["error"]["details"]["position"] == 1
    assert body["error"]["details"]["symbol"] == "X"


def test_unnormalized_initial_guess_rejected(client):
    res = client.post(
        "/models/train",
        json=train_payload("bad", ["aab"], transition=[[0.7, 0.2], [0.4, 0.6]]),
    )
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"]["type"] == "validation_error"
    assert any("transition" in d for d in body["error"]["details"])


def test_dimension_mismatch_initial_guess_rejected(client):
    res = client.post(
        "/models/train",
        json=train_payload(
            "bad", ["aab"], transition=[[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]
        ),
    )
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_empty_training_set_rejected(client):
    res = client.post("/models/train", json=train_payload("bad", []))
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "empty_observations"


def test_empty_sequence_in_training_set_rejected(client):
    res = client.post(
        "/models/train", json=train_payload("bad", ["aab", "", "bba"])
    )
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"]["type"] == "empty_observations"
    assert body["error"]["details"]["sequence"] == 1


def test_single_state_initial_guess_rejected(client):
    res = client.post(
        "/models/train",
        json=train_payload(
            "bad",
            ["aab"],
            states=["only"],
            initial=[1.0],
            transition=[[1.0]],
            emission=[[0.5, 0.5]],
        ),
    )
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_missing_observations_field_rejected(client):
    payload = train_payload("bad", ["aab"])
    del payload["observations"]
    res = client.post("/models/train", json=payload)
    assert res.status_code == 400
    assert res.get_json()["error"]["type"] == "validation_error"


def test_bad_training_controls_rejected(client):
    for override in ({"max_iterations": 0}, {"max_iterations": 1.5},
                     {"tolerance": 0}, {"tolerance": -1.0}):
        res = client.post(
            "/models/train", json=train_payload("bad", ["aab"], **override)
        )
        assert res.status_code == 400
        assert res.get_json()["error"]["type"] == "validation_error"


def test_training_name_conflict_rejected(client):
    client.post("/models", json=two_state_spec("taken"))
    res = client.post("/models/train", json=train_payload("taken", ["aab", "bba"]))
    assert res.status_code == 409
    assert res.get_json()["error"]["type"] == "model_exists"
    # 原档不被悄悄覆盖。
    spec = client.get("/models/taken").get_json()
    assert spec["emission"] == two_state_spec("taken")["emission"]


def test_concurrent_training_no_cross_talk(app, training_sequences):
    sequences_b = simulate(TRUE_MODEL_B, [50] * 16, seed=1234)

    # 先各自顺序训练一次，作为基准。
    client = app.test_client()
    expected = {}
    for name, seqs in (("seq-a", training_sequences), ("seq-b", sequences_b)):
        res = client.post("/models/train", json=train_payload(name, seqs))
        assert res.status_code == 201
        expected[name] = res.get_json()
    # 两批数据训出的档确实不同，后面才能谈「互不串」。
    assert expected["seq-a"]["model"]["emission"] != expected["seq-b"]["model"]["emission"]

    def work(n):
        # 每个线程用独立客户端打满整个 HTTP 栈。
        c = app.test_client()
        seqs, base = (training_sequences, "seq-a") if n % 2 == 0 else (sequences_b, "seq-b")
        res = c.post("/models/train", json=train_payload(f"par-{n}", seqs))
        assert res.status_code == 201
        return base, f"par-{n}", res.get_json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, range(8)))

    for base, name, body in results:
        exp = expected[base]
        # 中间量与产物必须与该任务自己的基准完全一致，不得串到另一任务。
        assert body["model"] == {**exp["model"], "name": name}
        assert body["log_likelihoods"] == exp["log_likelihoods"]
        assert body["iterations"] == exp["iterations"]


def test_concurrent_training_same_name_one_wins(app, training_sequences):
    def work(_):
        c = app.test_client()
        return c.post(
            "/models/train", json=train_payload("race", training_sequences)
        ).status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(work, range(4)))
    # 同名撞名按登记语义处理：恰好一个落库，其余 409，不覆盖。
    assert codes.count(201) == 1
    assert codes.count(409) == 3
