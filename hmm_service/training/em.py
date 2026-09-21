"""EM 主循环：多序列聚合 + 单轮重估 + 收敛控制。

每一轮先用当前参数解释整批观测（各序列充分统计量聚合成整批期望量），
再回过头重估参数。总对数似然逐轮单调不减；相邻两轮增量小于容差即
收敛，否则到达迭代次数上限停下——两种停法都如实记录在结果里。
全部中间量为本函数局部状态，并发训练任务互不写串。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..posterior import NEG_INF
from .aggregation import BatchStatistics
from .expectation import sequence_statistics
from .maximization import Parameters, reestimate

CONVERGED = "converged"
MAX_ITERATIONS = "max_iterations"


@dataclass
class TrainResult:
    """一次训练的全部产物。"""

    parameters: Parameters
    log_likelihoods: list[float]  # 含初值档自己的第 0 轮
    converged: bool
    stop_reason: str

    @property
    def iterations(self) -> int:
        """实际执行的重估轮数。"""
        return len(self.log_likelihoods) - 1

    @property
    def log_likelihood(self) -> float:
        """最终参数下整批观测的总对数似然。"""
        return self.log_likelihoods[-1]


def _log_vector(values: list[float]) -> list[float]:
    return [math.log(v) if v > 0.0 else NEG_INF for v in values]


def _log_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [_log_vector(row) for row in matrix]


def _expectation_step(
    params: Parameters, obs_idx_batch: list[list[int]]
) -> BatchStatistics:
    """用当前参数解释整批观测：逐序列充分统计量聚合成整批期望量。"""
    log_initial = _log_vector(params.initial)
    log_transition = _log_matrix(params.transition)
    log_emission = _log_matrix(params.emission)
    batch = BatchStatistics(len(params.initial), len(log_emission[0]))
    for obs_idx in obs_idx_batch:
        batch.add(sequence_statistics(log_initial, log_transition, log_emission, obs_idx))
    return batch


def train_em(
    spec: dict,
    sequences: list[list[str]],
    max_iterations: int,
    tolerance: float,
) -> TrainResult:
    """在初值档 ``spec`` 上对整批观测串跑 EM，直到收敛或触顶。

    ``spec`` 必须已过 ``validate_model_spec``；``sequences`` 必须已过
    ``validate_observations``（本函数不做入口校验）。
    """
    index_of = {sym: i for i, sym in enumerate(spec["alphabet"])}
    obs_idx_batch = [[index_of[s] for s in seq] for seq in sequences]

    params = Parameters(
        initial=list(spec["initial"]),
        transition=[list(row) for row in spec["transition"]],
        emission=[list(row) for row in spec["emission"]],
    )

    # 第 0 轮：初值档自己解释这批观测，记下起始总对数似然。
    batch = _expectation_step(params, obs_idx_batch)
    ll_prev = batch.total_log_likelihood
    log_likelihoods = [ll_prev]

    converged = False
    for _ in range(max_iterations):
        params = reestimate(batch, params)
        batch = _expectation_step(params, obs_idx_batch)
        ll = batch.total_log_likelihood
        log_likelihoods.append(ll)
        if ll - ll_prev < tolerance:
            converged = True
            break
        ll_prev = ll

    return TrainResult(
        parameters=params,
        log_likelihoods=log_likelihoods,
        converged=converged,
        stop_reason=CONVERGED if converged else MAX_ITERATIONS,
    )
