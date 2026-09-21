"""训练请求的开训前校验：不合法的请求一律在迭代开始前挡下。

复用登记通道对模型初值的全部校验（状态数 ≥ 2、维数相符、取值在
[0, 1]、行和与初值和在同一套 ``1e-6`` 容差内归一），再加上训练特有
的入口约束：

- 观测批次是非空数组，每条观测串非空，且只含字母表内符号；
- ``max_iterations`` 是正整数；``tolerance`` 是有限正数。

任何一项不过都返回带类型的错误（``validation_error`` /
``unknown_symbol`` / ``empty_training_set``），不会等迭代跑到一半才炸。
"""
from __future__ import annotations

import math
from typing import Any

from ..errors import EmptyTrainingSetError, UnknownSymbolError, ValidationError
from ..validation import validate_model_spec

MAX_ITERATIONS_CEILING = 10_000


def validate_training_request(
    body: Any,
) -> tuple[dict, list[list[str]], int, float]:
    """校验训练请求，返回 (已校验模型档, 观测符号批次, 上限, 容差)。"""
    if not isinstance(body, dict):
        raise ValidationError("训练请求体必须是 JSON 对象")

    # 初值按登记通道同一套标准校验：状态数、维数、取值、归一，一次挡全。
    # 名字在训练请求里是产出档的名字，同样必须非空。
    if not isinstance(body.get("name"), str) or not body.get("name", "").strip():
        raise ValidationError("name 必须是非空字符串")
    model = validate_model_spec(body)

    max_iterations = _validate_max_iterations(body.get("max_iterations"))
    tolerance = _validate_tolerance(body.get("tolerance"))

    sequences = _validate_observation_batch(body.get("observations"), model["alphabet"])
    return model, sequences, max_iterations, tolerance


def _validate_max_iterations(raw: Any) -> int:
    if raw is None:
        from .engine import DEFAULT_MAX_ITERATIONS

        return DEFAULT_MAX_ITERATIONS
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValidationError("max_iterations 必须是不小于 1 的整数")
    if raw > MAX_ITERATIONS_CEILING:
        raise ValidationError(
            f"max_iterations 超出上限 {MAX_ITERATIONS_CEILING}"
        )
    return raw


def _validate_tolerance(raw: Any) -> float:
    if raw is None:
        from .engine import DEFAULT_TOLERANCE

        return DEFAULT_TOLERANCE
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValidationError("tolerance 必须是有限正数")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError("tolerance 必须是有限正数")
    return value


def _validate_observation_batch(raw: Any, alphabet: list[str]) -> list[list[str]]:
    """观测批次：非空数组，每条串非空且只含字母表内符号。"""
    if raw is None:
        raise ValidationError(
            "缺少 observations 字段", details=["缺少字段: observations"]
        )
    if not isinstance(raw, list):
        raise ValidationError("observations 必须是观测串数组")
    if len(raw) == 0:
        raise EmptyTrainingSetError("训练观测批次为空，至少需要 1 条观测串")

    sequences: list[list[str]] = []
    for k, item in enumerate(raw):
        if isinstance(item, str):
            symbols = list(item)
        elif isinstance(item, list) and all(isinstance(s, str) for s in item):
            symbols = list(item)
        else:
            raise ValidationError(
                f"第 {k} 条观测串必须是字符串或符号数组"
            )
        if len(symbols) == 0:
            raise EmptyTrainingSetError(
                f"第 {k} 条观测串为空，每条训练串长度至少为 1",
                details={"sequence_index": k},
            )
        known = set(alphabet)
        for pos, sym in enumerate(symbols):
            if sym not in known:
                raise UnknownSymbolError(
                    f"第 {k} 条观测串位置 {pos} 的符号 {sym!r} 不在字母表内",
                    details={"sequence_index": k, "position": pos, "symbol": sym},
                )
        sequences.append(symbols)
    return sequences
