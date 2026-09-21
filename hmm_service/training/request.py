"""训练请求校验：初值档 + 训练集 + 迭代控制，开训前全部挡下。

初值档直接过与登记通道同一套 ``validate_model_spec``（状态数、维数、
行和、钉死的容差完全一致）；训练集逐串过 ``validate_observations``
（空串、字母表之外的符号分别抛对应类型，并补上第几条串的上下文）。
任何问题都在迭代开始之前以带类型的错误返回，不会跑到一半才炸。
"""
from __future__ import annotations

import math
from typing import Any

from ..errors import EmptyObservationsError, UnknownSymbolError, ValidationError
from ..validation import validate_model_spec, validate_observations

DEFAULT_MAX_ITERATIONS = 100
DEFAULT_TOLERANCE = 1e-4
# 迭代上限的天花板，防止误把上限写成天文数字把请求挂死。
MAX_MAX_ITERATIONS = 10_000


def _validate_max_iterations(raw: Any) -> int:
    if raw is None:
        return DEFAULT_MAX_ITERATIONS
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int)
        or not 1 <= raw <= MAX_MAX_ITERATIONS
    ):
        raise ValidationError(
            f"max_iterations 必须是 1..{MAX_MAX_ITERATIONS} 的整数"
        )
    return raw


def _validate_tolerance(raw: Any) -> float:
    if raw is None:
        return DEFAULT_TOLERANCE
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(raw)
        or raw <= 0.0
    ):
        raise ValidationError("tolerance 必须是正的有限数值")
    return float(raw)


def validate_training_request(
    body: Any,
) -> tuple[dict, list[list[str]], int, float]:
    """校验训练请求，返回 (初值档, 观测串列表, 迭代上限, 收敛容差)。"""
    if not isinstance(body, dict):
        raise ValidationError("训练请求必须是 JSON 对象")
    # 初值档：名字、状态数、字母表、维数、归一化，与登记同一套判据。
    spec = validate_model_spec(body)

    raw = body.get("observations")
    if raw is None:
        raise ValidationError(
            "缺少 observations 字段", details=["缺少字段: observations"]
        )
    if not isinstance(raw, list):
        raise ValidationError("observations 必须是观测串数组")
    if len(raw) == 0:
        raise EmptyObservationsError("训练集为空，至少需要一条观测串")

    sequences: list[list[str]] = []
    for idx, raw_seq in enumerate(raw):
        try:
            sequences.append(validate_observations(raw_seq, spec["alphabet"]))
        except UnknownSymbolError as err:
            details = {"sequence": idx}
            if isinstance(err.details, dict):
                details.update(err.details)
            raise UnknownSymbolError(
                f"第 {idx} 条训练串：{err.message}", details=details
            ) from None
        except EmptyObservationsError:
            raise EmptyObservationsError(
                f"第 {idx} 条训练串为空，长度至少为 1",
                details={"sequence": idx},
            ) from None

    return (
        spec,
        sequences,
        _validate_max_iterations(body.get("max_iterations")),
        _validate_tolerance(body.get("tolerance")),
    )
