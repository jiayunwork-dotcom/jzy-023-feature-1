"""对数域前向-后向递推与逐时刻后验。

前向在时刻 0 用初值乘该时刻发射（对数域为相加）；后向从序列末尾的
对数域单位值 0.0 往回走。同一时刻前向与后向相加（对数域）再归一，
得到该时刻各状态的后验，每行之和为 1。全程 logsumexp，不会下溢。
"""
from __future__ import annotations

import math

NEG_INF = float("-inf")


def logsumexp(values: list[float]) -> float:
    """对数域求和；全为 -inf 时返回 -inf。"""
    top = max(values)
    if top == NEG_INF:
        return NEG_INF
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def forward_backward(
    log_initial: list[float],
    log_transition: list[list[float]],
    log_emit: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """返回 (log_alpha, log_beta)，形状均为 [时刻][状态]。"""
    n_steps = len(log_emit)
    n_states = len(log_initial)

    log_alpha = [[NEG_INF] * n_states for _ in range(n_steps)]
    # 时刻 0：初值乘该时刻发射（对数域相加）。
    for i in range(n_states):
        log_alpha[0][i] = log_initial[i] + log_emit[0][i]
    for t in range(1, n_steps):
        for j in range(n_states):
            log_alpha[t][j] = log_emit[t][j] + logsumexp(
                [log_alpha[t - 1][i] + log_transition[i][j] for i in range(n_states)]
            )

    # 末尾取对数域单位值 0.0（即线性域的 1），逐步往回走。
    log_beta = [[0.0] * n_states for _ in range(n_steps)]
    for t in range(n_steps - 2, -1, -1):
        for i in range(n_states):
            log_beta[t][i] = logsumexp(
                [
                    log_transition[i][j] + log_emit[t + 1][j] + log_beta[t + 1][j]
                    for j in range(n_states)
                ]
            )
    return log_alpha, log_beta


def posterior(log_alpha: list[list[float]], log_beta: list[list[float]]) -> list[list[float]]:
    """同一时刻前向+后向（对数域）再归一；返回线性域概率，每行和为 1。"""
    n_steps = len(log_alpha)
    n_states = len(log_alpha[0])
    gamma: list[list[float]] = []
    for t in range(n_steps):
        log_joint = [log_alpha[t][i] + log_beta[t][i] for i in range(n_states)]
        norm = logsumexp(log_joint)
        gamma.append([math.exp(v - norm) for v in log_joint])
    return gamma
