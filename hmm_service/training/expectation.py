"""单条观测串的期望量：对数域充分统计量（每轮「解释观测」的一步）。

复用 posterior.forward_backward 算前向-后向，再把「每时刻停在各状态」
（gamma）与「相邻两时刻在状态对之间转移」（xi）的期望在对数域里累加
成本条序列的充分统计量。这些是带权重的期望量而不是硬计数；全程不
离开对数域，长串不会下溢成全零。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import ZeroProbabilityError
from ..posterior import NEG_INF, forward_backward, logsumexp


def logaddexp(a: float, b: float) -> float:
    """对数域加法：返回 log(exp(a) + exp(b))，两端皆可 -inf。"""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


@dataclass
class SequenceStatistics:
    """一条序列在当前参数下的充分统计量（全部对数域）与对数似然。"""

    log_likelihood: float
    log_gamma0: list[float]  # 首时刻停在各状态的期望
    log_xi: list[list[float]]  # 各状态对之间转移的期望次数
    log_emit_counts: list[list[float]]  # 各状态发射各符号的期望次数


def sequence_statistics(
    log_initial: list[float],
    log_transition: list[list[float]],
    log_emission: list[list[float]],
    obs_idx: list[int],
) -> SequenceStatistics:
    """对一条观测串（符号下标序列）计算对数域充分统计量。

    该串在当前参数下概率为 0 时抛 ``ZeroProbabilityError``：后验无定义，
    期望量无从谈起，无法继续训练。
    """
    n_states = len(log_initial)
    n_symbols = len(log_emission[0])
    n_steps = len(obs_idx)

    # log_emit[t][j]：状态 j 发射第 t 个观测符号的对数概率。
    log_emit = [
        [log_emission[j][obs_idx[t]] for j in range(n_states)]
        for t in range(n_steps)
    ]
    log_alpha, log_beta = forward_backward(log_initial, log_transition, log_emit)
    ll = logsumexp(log_alpha[-1])
    if ll == NEG_INF:
        raise ZeroProbabilityError("训练串在当前参数下概率为 0，无法继续训练")

    # 首时刻后验：初值重估的分子。
    log_gamma0 = [log_alpha[0][i] + log_beta[0][i] - ll for i in range(n_states)]

    # 逐时刻把 gamma 累进发射计数、把 xi 累进转移计数（对数域）。
    log_emit_counts = [[NEG_INF] * n_symbols for _ in range(n_states)]
    for t in range(n_steps):
        sym = obs_idx[t]
        for i in range(n_states):
            log_g = log_alpha[t][i] + log_beta[t][i] - ll
            log_emit_counts[i][sym] = logaddexp(log_emit_counts[i][sym], log_g)

    log_xi = [[NEG_INF] * n_states for _ in range(n_states)]
    for t in range(n_steps - 1):
        emit_next = log_emit[t + 1]
        beta_next = log_beta[t + 1]
        for i in range(n_states):
            base = log_alpha[t][i] - ll
            trans_row = log_transition[i]
            xi_row = log_xi[i]
            for j in range(n_states):
                v = base + trans_row[j] + emit_next[j] + beta_next[j]
                xi_row[j] = logaddexp(xi_row[j], v)

    return SequenceStatistics(ll, log_gamma0, log_xi, log_emit_counts)
