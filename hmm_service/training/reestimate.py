"""单轮重估（Baum-Welch 的 M 步）：用整批累加的期望量定出下一轮参数。

- 转移新值：``从 i 转到 j 的期望次数 / 从 i 转出的期望总次数``，按行归一
- 发射新值：``状态 i 观测到符号 s 的期望次数 / i 被占用的期望总时刻``，
  按行归一
- 初值新值：``各序列首时刻停在各状态的期望`` 归一

除法全部在对数域做（log 分子减 log 分母，再归一化），分母趋近于零
（该状态整批几乎没被用上）时不除零：**退回该行上一轮的取值**。
这是有依据的兜底——EM 的重估只是在已有可行解上抬对数似然，缺数据时
保留上一轮的可行行不会把已经在单调上升的对数似然打回头；也避免直接
落到均匀分布把该状态好不容易学出来的倾向抹掉。

每行结果再过一道线性归一，严格满足登记通道的容差判据（``1e-6``）。
"""
from __future__ import annotations

import math

from ..validation import TOLERANCE
from .accumulator import BatchExpectations
from .expectations import NEG_INF, ModelParams


def _normalize_log_row(
    log_row: list[float], fallback: list[float]
) -> list[float]:
    """把一行对数域相对权重归一化成概率行。

    行总期望为零（全 ``-inf``）时退回 ``fallback``（上一轮该行）。
    结果做一次线性归一，兜住浮点尾巴，保证行和严格贴近 1。
    """
    top = max(log_row)
    if top == NEG_INF:
        return list(fallback)
    row = [math.exp(v - top) for v in log_row]
    total = math.fsum(row)
    if total <= 0.0:
        return list(fallback)
    normalized = [v / total for v in row]
    # 线性归一后行和就是 1；再过一遍和检查，挡住任何意外退化。
    assert abs(math.fsum(normalized) - 1.0) <= TOLERANCE
    return normalized


def _ratio_row(
    log_num: list[float], log_den: float, fallback: list[float]
) -> list[float]:
    """``exp(log_num - log_den)`` 后按行归一；分母为零直接退回上一轮。

    必须先判分母：``-inf - (-inf)`` 在浮点里是 NaN，不能让它进到指数。
    """
    if log_den == NEG_INF:
        return list(fallback)
    return _normalize_log_row([v - log_den for v in log_num], fallback)


def reestimate(
    batch: BatchExpectations, previous: ModelParams
) -> ModelParams:
    """用整批期望量重估一份新参数；缺数据的行保留上一轮取值。"""
    n_states = len(previous.initial)
    n_symbols = len(previous.emission[0])

    transition = [
        _ratio_row(
            [batch.log_transition_num[i][j] for j in range(n_states)],
            batch.log_transition_den[i],
            previous.transition[i],
        )
        for i in range(n_states)
    ]
    emission = [
        _ratio_row(
            [batch.log_emission_num[i][s] for s in range(n_symbols)],
            batch.log_emission_den[i],
            previous.emission[i],
        )
        for i in range(n_states)
    ]
    # 初值：所有序列首时刻对各状态都没有期望时（等价于一批序列的首观测
    # 在当前模型下统统零概率），log_init 全 -inf，保留上一轮初值。
    initial = _normalize_log_row(batch.log_init, previous.initial)

    return ModelParams(initial=initial, transition=transition, emission=emission)
