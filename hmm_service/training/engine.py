"""训练主循环（EM 迭代）与收敛控制。

每轮：

1. E 步：用当前参数解释整批观测（逐条前向-后向，期望量对数域累加）
2. 报数：当前这批观测的总对数似然
3. 收敛判断：相邻两轮增量 < 容差即停（``converged``）
4. M 步：用累加期望量重估下一轮转移 / 发射 / 初值

总对数似然随迭代**单调不减**——这是 EM 的数学性质，也是判定本实现
对错的硬标准。浮点抖动可能让新值比旧值低极小一截（约 1e-12 相对量），
那种量级把报数钉回上一轮的值；若出现不可能的明显回落，拒绝这次重估、
保留上一轮参数并按收敛停下（EM 不动点上增量本就约为 0），绝不把
已经爬升的对数似然打回头。

停止原因如实落在结果里：``converged``（增量低于容差）或
``max_iterations``（触顶）；初始参数对这批观测概率为 0（对数似然
``-inf``）时无法起步，停在第 0 轮，原因 ``zero_initial_likelihood``。

本模块是纯函数式的迭代器：每次训练各自持有自己的观测批次与
迭代中间量，没有任何共享可变状态，并发任务互不写串。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .accumulator import BatchExpectations, accumulate
from .expectations import ModelParams, SequenceExpectation, sequence_expectation
from .reestimate import reestimate

# 收敛 / 触顶 / 初值零概率三种停止原因，直接进训练结果。
STOP_CONVERGED = "converged"
STOP_MAX_ITERATIONS = "max_iterations"
STOP_ZERO_INITIAL = "zero_initial_likelihood"

# 视为纯浮点抖动的相对回落幅度：小于它就把报数钉回上一轮，不破坏单调性。
_NUMERIC_JITTER = 1e-9

DEFAULT_MAX_ITERATIONS = 100
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TrainingResult:
    """一次训练的全部产物。"""

    iterations: int
    stop_reason: str
    converged: bool
    max_iterations: int
    tolerance: float
    log_likelihoods: list[float]
    final_log_likelihood: float | None
    params: ModelParams
    n_sequences: int

    def to_spec(self, name: str, states: list[str], alphabet: list[str]) -> dict:
        """把估出的参数组装成一份正式模型档（过同一套登记校验）。"""
        return {
            "name": name,
            "states": list(states),
            "alphabet": list(alphabet),
            "initial": self.params.initial,
            "transition": self.params.transition,
            "emission": self.params.emission,
        }


def _explain_batch(
    params: ModelParams,
    sequences: list[list[int]],
    n_symbols: int,
) -> tuple[float, BatchExpectations]:
    """用当前参数解释整批观测：逐条 E 步，再聚合成整批期望量。"""
    expectations: list[SequenceExpectation] = [
        sequence_expectation(params, obs_idx) for obs_idx in sequences
    ]
    batch = accumulate(sequences, expectations, n_symbols)
    return batch.total_log_likelihood, batch


def _is_jitter(previous: float, candidate: float) -> bool:
    """新值小幅低于旧值是否只是浮点噪声。"""
    scale = max(1.0, abs(previous), abs(candidate))
    return previous - candidate <= _NUMERIC_JITTER * scale


def train(
    model: dict,
    sequences_idx: list[list[int]],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    explain_batch: Callable[
        [ModelParams, list[list[int]], int], tuple[float, BatchExpectations]
    ] = _explain_batch,
) -> TrainingResult:
    """在给定初值上跑 Baum-Welch，直到收敛或触顶。

    ``model`` 必须已经过训练入口校验（维数、归一、字母表、状态数都合法）；
    ``sequences_idx`` 是观测符号下标序列的列表，非空、每条长度至少 1。
    ``explain_batch`` 默认逐条前向-后向，允许注入替身做收敛控制的单测。
    """
    params = ModelParams.from_spec(model)
    n_symbols = len(model["emission"][0])
    log_likelihoods: list[float] = []

    def finish(
        iteration: int,
        stop_reason: str,
        converged: bool,
        final_params: ModelParams,
        final_ll: float | None,
    ) -> TrainingResult:
        return TrainingResult(
            iterations=iteration,
            stop_reason=stop_reason,
            converged=converged,
            max_iterations=max_iterations,
            tolerance=tolerance,
            log_likelihoods=list(log_likelihoods),
            final_log_likelihood=final_ll,
            params=final_params,
            n_sequences=len(sequences_idx),
        )

    # 第 0 轮：先解释一次，拿到初始对数似然与初始期望量。
    current_ll, batch = explain_batch(params, sequences_idx, n_symbols)
    if not math.isfinite(current_ll):
        # 初值对这批观测概率为 0：重估无从起步，原样落库、明确报告。
        return finish(0, STOP_ZERO_INITIAL, False, params, None)
    log_likelihoods.append(current_ll)

    for iteration in range(1, max_iterations + 1):
        candidate = reestimate(batch, params)
        candidate_ll, candidate_batch = explain_batch(
            candidate, sequences_idx, n_symbols
        )

        # 硬标准：单调不减。只可能遇到两种异常：
        # 1) 极小的浮点回落 —— 钉回旧报数，视为已到不动点，按收敛停；
        # 2) 明显回落（理论上不该发生）—— 拒绝重估，同样按收敛停。
        if not math.isfinite(candidate_ll) or candidate_ll < current_ll:
            if math.isfinite(candidate_ll) and _is_jitter(current_ll, candidate_ll):
                log_likelihoods.append(current_ll)
            return finish(iteration, STOP_CONVERGED, True, params, log_likelihoods[-1])

        log_likelihoods.append(candidate_ll)
        improvement = candidate_ll - current_ll
        params, batch = candidate, candidate_batch
        current_ll = candidate_ll

        if improvement < tolerance:
            return finish(iteration, STOP_CONVERGED, True, params, current_ll)

    return finish(max_iterations, STOP_MAX_ITERATIONS, False, params, current_ll)
