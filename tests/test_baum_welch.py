"""Baum-Welch 训练引擎：E 步期望量、M 步重估、单调不减、收敛/触顶、边界。"""
from __future__ import annotations

import math
import random

import pytest

from hmm_service.posterior import forward_backward, logsumexp, posterior
from hmm_service.training.accumulator import accumulate
from hmm_service.training.engine import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    STOP_CONVERGED,
    STOP_MAX_ITERATIONS,
    STOP_ZERO_INITIAL,
    train,
)
from hmm_service.training.expectations import (
    NEG_INF,
    ModelParams,
    sequence_expectation,
)

# 一个「真」两状态模型：状态各自偏好 a / b，高自转 -> 明显区段切换。
TRUE_INITIAL = [0.5, 0.5]
TRUE_TRANSITION = [[0.9, 0.1], [0.1, 0.9]]
TRUE_EMISSION = [[0.8, 0.2], [0.2, 0.8]]


def _generate_sequences(seed: int, n_sequences: int, length: int) -> list[list[int]]:
    rng = random.Random(seed)
    sequences = []
    for _ in range(n_sequences):
        state = 0 if rng.random() < 0.5 else 1
        obs = []
        for _ in range(length):
            obs.append(0 if rng.random() < TRUE_EMISSION[state][0] else 1)
            state = 0 if rng.random() < TRUE_TRANSITION[state][0] else 1
        sequences.append(obs)
    return sequences


@pytest.fixture(scope="module")
def sequences():
    return _generate_sequences(20260921, n_sequences=16, length=80)


def _seed_model() -> dict:
    return {
        "name": "seed",
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [0.6, 0.4],
        "transition": [[0.6, 0.4], [0.4, 0.6]],
        "emission": [[0.55, 0.45], [0.45, 0.55]],
    }


# ---------- E 步期望量 ----------

def test_gamma_matches_decode_side_posterior_and_is_normalized():
    # 训练侧的 gamma 必须与解码侧 posterior 的逐时刻后验是同一个量，
    # 每行和为 1；这也卡住「忘了除 P(O)」的实现。
    params = ModelParams.from_spec(_seed_model())
    obs = [0, 1, 0, 0, 1]
    exp = sequence_expectation(params, obs)

    log_initial = [math.log(v) for v in params.initial]
    log_transition = [[math.log(v) for v in row] for row in params.transition]
    log_emission = [[math.log(v) for v in row] for row in params.emission]
    log_emit = [[log_emission[j][obs[t]] for j in range(2)] for t in range(5)]
    log_alpha, log_beta = forward_backward(log_initial, log_transition, log_emit)
    expected_gamma = posterior(log_alpha, log_beta)

    for t in range(5):
        for i in range(2):
            assert math.exp(exp.log_gamma[t][i]) == pytest.approx(expected_gamma[t][i])
        assert math.fsum(math.exp(v) for v in exp.log_gamma[t]) == pytest.approx(1.0)


def test_xi_counts_sum_to_transitions_per_timestep():
    # 对所有状态对、所有相邻时刻的 xi 之和必须恰为 T-1（每个时刻一次转移）。
    params = ModelParams.from_spec(_seed_model())
    obs = [0, 1, 1, 0]
    exp = sequence_expectation(params, obs)
    total = math.fsum(
        math.exp(v) for row in exp.log_xi_sum for v in row if v != NEG_INF
    )
    assert total == pytest.approx(len(obs) - 1)
    # 逐行 xi 与 gamma 的标准关系：对 j 求和 xi_t(i,j) 再对 t 求和 =
    # sum_{t<T-1} gamma_t(i)。
    for i in range(2):
        xi_row = math.fsum(math.exp(v) for v in exp.log_xi_sum[i])
        gamma_row = math.fsum(
            math.exp(exp.log_gamma[t][i]) for t in range(len(obs) - 1)
        )
        assert xi_row == pytest.approx(gamma_row, abs=1e-12)


def test_log_likelihood_matches_forward_mass():
    params = ModelParams.from_spec(_seed_model())
    obs = [1, 0, 1]
    exp = sequence_expectation(params, obs)
    log_initial = [math.log(v) for v in params.initial]
    log_transition = [[math.log(v) for v in row] for row in params.transition]
    log_emission = [[math.log(v) for v in row] for row in params.emission]
    log_emit = [[log_emission[j][obs[t]] for j in range(2)] for t in range(3)]
    log_alpha, _ = forward_backward(log_initial, log_transition, log_emit)
    assert exp.log_likelihood == pytest.approx(logsumexp(log_alpha[-1]))


# ---------- 多序列聚合 ----------

def test_batch_aggregate_sums_log_likelihoods_and_counts(sequences):
    params = ModelParams.from_spec(_seed_model())
    expectations = [sequence_expectation(params, seq) for seq in sequences]
    batch = accumulate(sequences, expectations, n_symbols=2)

    assert batch.total_log_likelihood == pytest.approx(
        math.fsum(e.log_likelihood for e in expectations)
    )
    # 发射期望总数 = 全部时刻数；转移期望总数 = 全部 (T_k - 1)；
    # 首时刻期望总数 = 序列条数。
    total_emission = math.fsum(
        math.exp(v)
        for i in range(2)
        for v in batch.log_emission_num[i]
    )
    assert total_emission == pytest.approx(sum(len(s) for s in sequences))
    total_transition = math.fsum(
        math.exp(v)
        for i in range(2)
        for v in batch.log_transition_num[i]
    )
    assert total_transition == pytest.approx(
        sum(len(s) - 1 for s in sequences)
    )
    assert math.fsum(math.exp(v) for v in batch.log_init) == pytest.approx(
        len(sequences)
    )


# ---------- 主循环：单调不减、收敛/触顶 ----------

def test_log_likelihood_monotone_non_decreasing(sequences):
    result = train(_seed_model(), sequences, max_iterations=40, tolerance=1e-9)
    lls = result.log_likelihoods
    assert len(lls) >= 2
    for prev, nxt in zip(lls, lls[1:]):
        assert nxt >= prev


def test_training_learns_two_regime_emissions(sequences):
    # 迭代若干轮后，两个状态的发射向量必须朝不同符号拉开（允许状态顺序对调）。
    result = train(_seed_model(), sequences, max_iterations=40, tolerance=1e-9)
    emissions = result.params.emission
    e0, e1 = emissions
    separated = (
        e0[0] > 0.65 and e1[1] > 0.65
    ) or (
        e0[1] > 0.65 and e1[0] > 0.65
    )
    assert separated, emissions
    # 自转也应明显高于初始猜测（真模型 0.9）。
    diagonals = [result.params.transition[i][i] for i in range(2)]
    assert min(diagonals) > 0.8, result.params.transition


def test_convergence_stop_marks_converged(sequences):
    # 给一个接近不动点的初值 + 大容差，第一轮增量就低于容差，必须判收敛。
    near_fixed = {
        "name": "seed",
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [0.5, 0.5],
        "transition": TRUE_TRANSITION,
        "emission": TRUE_EMISSION,
    }
    result = train(near_fixed, sequences, max_iterations=50, tolerance=10.0)
    assert result.converged is True
    assert result.stop_reason == STOP_CONVERGED
    assert result.iterations >= 1


def test_max_iterations_stop_marks_cap(sequences):
    result = train(_seed_model(), sequences, max_iterations=5, tolerance=1e-12)
    assert result.converged is False
    assert result.stop_reason == STOP_MAX_ITERATIONS
    assert result.iterations == 5
    assert len(result.log_likelihoods) == 6  # 第 0 轮 + 5 轮重估


def test_result_fields_report_metadata(sequences):
    result = train(_seed_model(), sequences, max_iterations=7, tolerance=1e-9)
    assert result.max_iterations == 7
    assert result.tolerance == 1e-9
    assert result.n_sequences == len(sequences)
    assert result.final_log_likelihood == result.log_likelihoods[-1]
    assert math.isfinite(result.final_log_likelihood)


def test_estimated_rows_are_normalized(sequences):
    result = train(_seed_model(), sequences, max_iterations=20, tolerance=1e-9)
    for row in result.params.transition:
        assert math.fsum(row) == pytest.approx(1.0, abs=1e-12)
    for row in result.params.emission:
        assert math.fsum(row) == pytest.approx(1.0, abs=1e-12)
    assert math.fsum(result.params.initial) == pytest.approx(1.0, abs=1e-12)


def test_to_spec_roundtrips_registration_validation():
    # 估出的档必须原样通过登记通道的同一套校验。
    from hmm_service.validation import validate_model_spec

    sequences = _generate_sequences(1, 6, 40)
    result = train(_seed_model(), sequences, max_iterations=10, tolerance=1e-9)
    spec = validate_model_spec(
        result.to_spec("learned", ["s0", "s1"], ["a", "b"])
    )
    assert spec["name"] == "learned"
    assert len(spec["transition"]) == 2 and len(spec["emission"][0]) == 2


# ---------- 边界 ----------

def test_unreachable_state_does_not_crash_and_keeps_monotone():
    # s2 初值为 0 且无人转入：它那一行期望恒为零，重估保留上一轮取值，
    # 不除零、不出 NaN、对数似然不回头。
    model = {
        "name": "dead",
        "states": ["s0", "s1", "s2"],
        "alphabet": ["a", "b"],
        "initial": [0.5, 0.5, 0.0],
        "transition": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.3, 0.3, 0.4]],
        "emission": [[0.7, 0.3], [0.3, 0.7], [0.5, 0.5]],
    }
    sequences = _generate_sequences(3, 8, 50)
    result = train(model, sequences, max_iterations=15, tolerance=1e-9)
    assert result.stop_reason in (STOP_CONVERGED, STOP_MAX_ITERATIONS)
    for prev, nxt in zip(result.log_likelihoods, result.log_likelihoods[1:]):
        assert nxt >= prev
    # 死状态的行被原样保留（兜底，而不是除零）。
    assert result.params.transition[2] == [0.3, 0.3, 0.4]
    assert result.params.emission[2] == [0.5, 0.5]
    flat = (
        [v for row in result.params.transition for v in row]
        + [v for row in result.params.emission for v in row]
        + result.params.initial
    )
    assert all(math.isfinite(v) for v in flat)


def test_zero_initial_likelihood_stops_cleanly_at_iteration_zero():
    # 唯一活着的状态只会发 a，观测却含 b：P(O)=0，不抛异常、不出 NaN，
    # 第 0 轮明确停下，参数原样保留。
    model = {
        "name": "z",
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [1.0, 0.0],
        "transition": [[1.0, 0.0], [0.0, 1.0]],
        "emission": [[1.0, 0.0], [0.0, 1.0]],
    }
    result = train(model, [[0, 1, 0]], max_iterations=10, tolerance=1e-6)
    assert result.iterations == 0
    assert result.stop_reason == STOP_ZERO_INITIAL
    assert result.final_log_likelihood is None
    assert result.log_likelihoods == []
    assert result.params.initial == [1.0, 0.0]


def test_single_step_sequences_supported():
    # 长度 1 的序列没有转移可累加：转移期望全 -inf，各行走上一轮兜底。
    result = train(_seed_model(), [[0], [1], [0]], max_iterations=5, tolerance=1e-12)
    assert result.stop_reason in (STOP_CONVERGED, STOP_MAX_ITERATIONS)
    flat = [v for row in result.params.transition for v in row]
    assert all(math.isfinite(v) for v in flat)
    for row in result.params.transition:
        assert math.fsum(row) == pytest.approx(1.0)


def test_training_is_independent_between_calls(sequences):
    # 同一初值连训两次，产物必须完全一致（无共享中间量泄漏）。
    r1 = train(_seed_model(), sequences, max_iterations=10, tolerance=1e-9)
    r2 = train(_seed_model(), sequences, max_iterations=10, tolerance=1e-9)
    assert r1.log_likelihoods == r2.log_likelihoods
    assert r1.params.transition == r2.params.transition
    assert r1.params.emission == r2.params.emission


def test_defaults():
    assert DEFAULT_MAX_ITERATIONS == 100
    assert DEFAULT_TOLERANCE == 1e-6


def test_each_iteration_explains_batch_exactly_once(sequences):
    # 注入替身计数：k 轮迭代应当解释 k+1 次（第 0 轮 + 每轮重估后一次），
    # 不多跑一次 E 步。
    calls = []
    from hmm_service.training.expectations import sequence_expectation as real_e
    from hmm_service.training.accumulator import accumulate as real_acc

    def counting_explain(params, seqs, n_symbols):
        calls.append(1)
        expectations = [real_e(params, seq) for seq in seqs]
        batch = real_acc(seqs, expectations, n_symbols)
        return batch.total_log_likelihood, batch

    result = train(
        _seed_model(), sequences, max_iterations=5, tolerance=1e-12,
        explain_batch=counting_explain,
    )
    assert len(calls) == result.iterations + 1
    assert result.iterations == 5
