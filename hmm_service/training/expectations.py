"""单条观测串的期望量（Baum-Welch 的 E 步），全程对数域。

用当前参数解释一条观测串，算出：

- ``log_likelihood``：该串在当前参数下的对数似然 ``log P(O)``
- ``log_gamma``：逐时刻停在各状态的期望（后验）的对数，形状 [时刻][状态]
- ``log_xi_sum``：相邻时刻在状态对之间转移的期望次数之和（对数域），
  形状 [状态][状态]，对 t = 0..T-2 累加
- ``log_init_count``：首时刻停在各状态的期望，即 ``log_gamma[0]``

所有累加都在对数域做（logaddexp），长串相乘不会下溢成零。
前向-后向直接复用解码侧 :mod:`hmm_service.posterior`（只读调用，
不改其既有对外行为）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..posterior import forward_backward, logsumexp

NEG_INF = float("-inf")


def logaddexp(a: float, b: float) -> float:
    """两个对数域值的稳定相加；-inf 是吸收元。"""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        a, b = b, a
        # 现在 b >= a，且二者均有限。
    return b + math.log1p(math.exp(a - b))


@dataclass(frozen=True)
class ModelParams:
    """训练过程中手上的一份参数（线性域，每行归一）。"""

    initial: list[float]
    transition: list[list[float]]
    emission: list[list[float]]

    @classmethod
    def from_spec(cls, model: dict) -> "ModelParams":
        return cls(
            initial=[float(v) for v in model["initial"]],
            transition=[[float(v) for v in row] for row in model["transition"]],
            emission=[[float(v) for v in row] for row in model["emission"]],
        )


@dataclass(frozen=True)
class SequenceExpectation:
    """一条观测串在当前参数下的全部期望量。"""

    log_likelihood: float
    log_gamma: list[list[float]]
    log_xi_sum: list[list[float]]
    log_init_count: list[float]


def _log_vector(values: list[float]) -> list[float]:
    return [math.log(v) if v > 0.0 else NEG_INF for v in values]


def _log_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [_log_vector(row) for row in matrix]


def sequence_expectation(
    params: ModelParams, obs_idx: list[int]
) -> SequenceExpectation:
    """对单条观测串跑一遍前向-后向，算出 E 步所需的全部期望量。

    ``obs_idx`` 是观测符号在字母表中的下标；至少 1 个时刻
    （空串在训练入口就被挡下）。
    """
    n_states = len(params.initial)
    n_steps = len(obs_idx)

    log_initial = _log_vector(params.initial)
    log_transition = _log_matrix(params.transition)
    log_emission = _log_matrix(params.emission)
    # log_emit[t][j]：状态 j 发射第 t 个观测符号的对数概率。
    log_emit = [
        [log_emission[j][obs_idx[t]] for j in range(n_states)]
        for t in range(n_steps)
    ]

    log_alpha, log_beta = forward_backward(log_initial, log_transition, log_emit)
    log_likelihood = logsumexp(log_alpha[-1])

    # log_gamma[t][i] = log P(X_t=i | O, 参数)。
    # alpha+beta 是与观测的联合概率（对数域），逐行除以同一个 P(O)
    # （对数域减去 log P(O)）才归一；与下面 xi 的逐 t 归一用同一个常数。
    # P(O)=0（-inf）时后验无定义，全部留 -inf，调用方在引擎层直接判停。
    if log_likelihood == NEG_INF:
        log_gamma = [[NEG_INF] * n_states for _ in range(n_steps)]
    else:
        log_gamma = [
            [
                log_alpha[t][i] + log_beta[t][i] - log_likelihood
                for i in range(n_states)
            ]
            for t in range(n_steps)
        ]

    # log_xi_sum[i][j]：对所有相邻时刻 t=0..T-2，log P(X_t=i, X_{t+1}=j | O)
    # 在对数域累加。xi 与 gamma 差一个对所有状态对的归一常数 log P(O)，
    # 每个 t 单独归一一次，再逐 t 对数域累加。
    log_xi_sum = [[NEG_INF] * n_states for _ in range(n_states)]
    for t in range(n_steps - 1):
        for i in range(n_states):
            for j in range(n_states):
                log_ij = (
                    log_alpha[t][i]
                    + log_transition[i][j]
                    + log_emit[t + 1][j]
                    + log_beta[t + 1][j]
                )
                if log_ij != NEG_INF:
                    log_ij -= log_likelihood
                log_xi_sum[i][j] = logaddexp(log_xi_sum[i][j], log_ij)

    return SequenceExpectation(
        log_likelihood=log_likelihood,
        log_gamma=log_gamma,
        log_xi_sum=log_xi_sum,
        log_init_count=list(log_gamma[0]),
    )
