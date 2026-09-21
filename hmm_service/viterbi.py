"""对数域 Viterbi：最大化整段联合概率 P(观测, 路径)，并回溯出唯一路径。

并列打破规则钉死为 ``TIE_BREAK_RULE``：无论是选最优前驱还是选终点，
都按状态下标升序扫描、严格大于才更新，因此并列时恒取下标最小的状态。
该规则随每次解码结果一并返回。
"""
from __future__ import annotations

NEG_INF = float("-inf")

TIE_BREAK_RULE = "lowest_state_index"


def viterbi(
    log_initial: list[float],
    log_transition: list[list[float]],
    log_emit: list[list[float]],
) -> tuple[list[int], float]:
    """返回 (最优路径的状态下标列表, 该路径与观测的对数联合概率)。

    ``log_emit[t][j]`` 是状态 j 发射第 t 个观测符号的对数概率。
    全程对数域累加，零概率以 -inf 表示，长串不会下溢。
    """
    n_steps = len(log_emit)
    n_states = len(log_initial)

    delta = [[NEG_INF] * n_states for _ in range(n_steps)]
    backptr = [[0] * n_states for _ in range(n_steps)]

    for i in range(n_states):
        delta[0][i] = log_initial[i] + log_emit[0][i]

    for t in range(1, n_steps):
        prev, curr, bp = delta[t - 1], delta[t], backptr[t]
        emit_t = log_emit[t]
        for j in range(n_states):
            best = NEG_INF
            best_i = 0
            for i in range(n_states):
                score = prev[i] + log_transition[i][j]
                if score > best:  # 严格大于：并列时保留先扫到的（下标最小）前驱
                    best = score
                    best_i = i
            curr[j] = emit_t[j] + best
            bp[j] = best_i

    # 终点同样按下标升序、严格大于才更新：并列取下标最小的状态。
    best_final = NEG_INF
    last = 0
    for i in range(n_states):
        if delta[n_steps - 1][i] > best_final:
            best_final = delta[n_steps - 1][i]
            last = i

    path = [0] * n_steps
    path[n_steps - 1] = last
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backptr[t][path[t]]
    return path, best_final
