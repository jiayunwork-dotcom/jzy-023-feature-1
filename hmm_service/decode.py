"""把 Viterbi 与前向-后向组装成一次完整解码。

纯函数、全部局部状态：多档并行解码时各自持有自己的路径与后验表，
不存在共享可变状态，互不写串。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ZeroProbabilityError
from .posterior import forward_backward, logsumexp, posterior
from .viterbi import TIE_BREAK_RULE, viterbi

NEG_INF = float("-inf")


def _log_vector(values: list[float]) -> list[float]:
    return [math.log(v) if v > 0.0 else NEG_INF for v in values]


def _log_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [_log_vector(row) for row in matrix]


@dataclass
class DecodeResult:
    """一次解码的全部产物。"""

    path: list[str]
    log_joint_probability: float
    log_likelihood: float
    posterior: list[list[float]]
    tie_break: str


def decode(model: dict, symbols: list[str]) -> DecodeResult:
    """对一份模型档与一串观测符号执行解码。

    返回 Viterbi 路径（状态名）、该路径与观测的对数联合概率、
    逐时刻后验表，以及所用的并列打破规则。
    """
    states: list[str] = model["states"]
    index_of = {sym: i for i, sym in enumerate(model["alphabet"])}
    obs_idx = [index_of[s] for s in symbols]

    log_initial = _log_vector(model["initial"])
    log_transition = _log_matrix(model["transition"])
    log_emission = _log_matrix(model["emission"])
    # log_emit[t][j]：状态 j 发射第 t 个观测符号的对数概率。
    log_emit = [
        [log_emission[j][obs_idx[t]] for j in range(len(states))]
        for t in range(len(obs_idx))
    ]

    path_idx, log_joint = viterbi(log_initial, log_transition, log_emit)
    if log_joint == NEG_INF:
        raise ZeroProbabilityError("该观测串在此模型下概率为 0，无法解码")

    log_alpha, log_beta = forward_backward(log_initial, log_transition, log_emit)
    gamma = posterior(log_alpha, log_beta)

    return DecodeResult(
        path=[states[i] for i in path_idx],
        log_joint_probability=log_joint,
        log_likelihood=logsumexp(log_alpha[-1]),
        posterior=gamma,
        tie_break=TIE_BREAK_RULE,
    )
