"""单轮重估（M 步）：由整批期望量定出下一轮的转移、发射、初值。

转移的新值来自「从某状态转出到各状态的期望次数」按行归一，发射的
新值来自「某状态下观测到各符号的期望次数」按行归一，初值来自各
序列首时刻停在各状态的期望的归一。归一在对数域做（减行的
logsumexp 再取 exp），期望次数再小也不会下溢成 NaN。

某状态在整批里几乎没被用上、整行期望次数为零（对数域 -inf）时，
退回上一轮的该行：该状态本轮权重为零，取哪一行都不影响似然下界，
退回旧行既保住归一约束，也不会把单调上升的对数似然打回头。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..posterior import NEG_INF, logsumexp
from .aggregation import BatchStatistics


@dataclass
class Parameters:
    """一轮 HMM 参数（线性域，各行严格归一）。"""

    initial: list[float]
    transition: list[list[float]]
    emission: list[list[float]]


def _exp_or_zero(log_v: float) -> float:
    return math.exp(log_v) if log_v != NEG_INF else 0.0


def _normalize_row(log_row: list[float], fallback: list[float]) -> list[float]:
    """对数域行归一；整行零计数（全 -inf）时退回上一轮的行。"""
    total = logsumexp(log_row)
    if total == NEG_INF:
        return list(fallback)
    return [_exp_or_zero(v - total) for v in log_row]


def reestimate(batch: BatchStatistics, previous: Parameters) -> Parameters:
    """由整批期望量重估一轮参数；零计数行退回 ``previous`` 的对应行。"""
    n_states = len(batch.log_initial)
    # 初值：各序列首时刻期望之和归一（整批和为序列条数，不会整行零计数）。
    initial = _normalize_row(batch.log_initial, previous.initial)
    transition = [
        _normalize_row(batch.log_transition[i], previous.transition[i])
        for i in range(n_states)
    ]
    emission = [
        _normalize_row(batch.log_emission[i], previous.emission[i])
        for i in range(n_states)
    ]
    return Parameters(initial, transition, emission)
