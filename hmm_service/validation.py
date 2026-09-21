"""模型档与观测串的校验：归一化、维数、字母表、观测符号。

归一化容差钉死为 ``TOLERANCE``，登记时校验，解码前不会再有机会
遇到维数不符或未归一的模型档。
"""
from __future__ import annotations

import math
from typing import Any

from .errors import (
    EmptyObservationsError,
    UnknownSymbolError,
    ValidationError,
)

# 行和 / 初值和与 1 的容差，钉死，调用方不得调整。
TOLERANCE = 1e-6

REQUIRED_FIELDS = ("name", "states", "alphabet", "initial", "transition", "emission")

_MAX_NAME_LENGTH = 128


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_probability(value: Any, where: str, problems: list[str]) -> None:
    if not _is_finite_number(value):
        problems.append(f"{where} 不是有限数值: {value!r}")
    elif not 0.0 <= value <= 1.0:
        problems.append(f"{where} 超出 [0, 1]: {value!r}")


def _validate_name(raw: Any, problems: list[str]) -> str:
    if not isinstance(raw, str) or not raw.strip():
        problems.append("name 必须是非空字符串")
        return ""
    name = raw.strip()
    if len(name) > _MAX_NAME_LENGTH:
        problems.append(f"name 过长（>{_MAX_NAME_LENGTH}）")
    return name


def _validate_states(raw: Any, problems: list[str]) -> list[str]:
    if not isinstance(raw, list) or len(raw) < 2:
        problems.append("states 必须至少包含 2 个状态")
        return []
    if not all(isinstance(s, str) and s for s in raw):
        problems.append("states 必须全部是非空字符串")
        return []
    if len(set(raw)) != len(raw):
        problems.append("states 存在重名")
        return []
    return list(raw)


def _validate_alphabet(raw: Any, problems: list[str]) -> list[str]:
    if not isinstance(raw, list) or len(raw) < 1:
        problems.append("alphabet 必须至少包含 1 个符号")
        return []
    if not all(isinstance(s, str) and len(s) == 1 for s in raw):
        problems.append("alphabet 符号必须是单字符字符串（观测串按字符切分）")
        return []
    if len(set(raw)) != len(raw):
        problems.append("alphabet 存在重复符号")
        return []
    return list(raw)


def _validate_vector(raw: Any, length: int, label: str, problems: list[str]) -> list[float]:
    if not isinstance(raw, list) or len(raw) != length:
        problems.append(f"{label} 必须是长度 {length} 的数组")
        return []
    for i, v in enumerate(raw):
        _check_probability(v, f"{label}[{i}]", problems)
    return [float(v) for v in raw if _is_finite_number(v)]


def _validate_matrix(raw: Any, rows: int, cols: int, label: str, problems: list[str]) -> list[list[float]]:
    if not isinstance(raw, list) or len(raw) != rows:
        problems.append(f"{label} 必须有 {rows} 行")
        return []
    matrix: list[list[float]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != cols:
            problems.append(f"{label} 第 {i} 行必须是长度 {cols} 的数组")
            continue
        for j, v in enumerate(row):
            _check_probability(v, f"{label}[{i}][{j}]", problems)
        matrix.append([float(v) for v in row if _is_finite_number(v)])
    return matrix


def _check_row_sums(matrix: list[list[float]], label: str, problems: list[str]) -> None:
    for i, row in enumerate(matrix):
        total = math.fsum(row)
        if abs(total - 1.0) > TOLERANCE:
            problems.append(
                f"{label} 第 {i} 行和为 {total!r}，偏离 1 超过容差 {TOLERANCE}"
            )


def validate_model_spec(spec: Any) -> dict:
    """校验并归一化一份模型档；任何一项不合法都抛 ``ValidationError``。

    返回只含已校验字段的新字典（浮点化），调用方不得假设传入对象被复用。
    """
    if not isinstance(spec, dict):
        raise ValidationError("模型档必须是 JSON 对象")
    missing = [f for f in REQUIRED_FIELDS if f not in spec]
    if missing:
        raise ValidationError(
            "模型档缺项", details=[f"缺少字段: {f}" for f in missing]
        )

    # 第一阶段：名字、状态集、字母表。
    problems: list[str] = []
    name = _validate_name(spec["name"], problems)
    states = _validate_states(spec["states"], problems)
    alphabet = _validate_alphabet(spec["alphabet"], problems)
    if problems:
        raise ValidationError("模型档校验失败", details=problems)

    n_states, n_symbols = len(states), len(alphabet)

    # 第二阶段：维数与取值范围。
    problems = []
    initial = _validate_vector(spec["initial"], n_states, "initial", problems)
    transition = _validate_matrix(spec["transition"], n_states, n_states, "transition", problems)
    emission = _validate_matrix(spec["emission"], n_states, n_symbols, "emission", problems)
    if problems:
        raise ValidationError("模型档校验失败", details=problems)

    # 第三阶段：归一化（此时维数与取值都已合法，行和才有意义）。
    problems = []
    total = math.fsum(initial)
    if abs(total - 1.0) > TOLERANCE:
        problems.append(f"initial 之和为 {total!r}，偏离 1 超过容差 {TOLERANCE}")
    _check_row_sums(transition, "transition", problems)
    _check_row_sums(emission, "emission", problems)
    if problems:
        raise ValidationError("模型档归一化校验失败", details=problems)

    normalized: dict[str, Any] = {
        "name": name,
        "states": states,
        "alphabet": alphabet,
        "initial": initial,
        "transition": transition,
        "emission": emission,
    }

    # 可选的示范观测串（内置示范档使用），同样必须落在字母表内。
    if "demo_observation" in spec:
        symbols = validate_observations(spec["demo_observation"], alphabet)
        normalized["demo_observation"] = "".join(symbols)
    return normalized


def validate_observations(raw: Any, alphabet: list[str]) -> list[str]:
    """把请求里的观测串规范成符号列表，并逐个核对字母表。

    接受字符串（按字符切分）或符号数组；空串与未知符号分别抛
    ``EmptyObservationsError`` 与 ``UnknownSymbolError``。
    """
    if raw is None:
        raise ValidationError("缺少 observations 字段", details=["缺少字段: observations"])
    if isinstance(raw, str):
        symbols = list(raw)
    elif isinstance(raw, list):
        if not all(isinstance(s, str) for s in raw):
            raise ValidationError("观测串数组元素必须是字符串符号")
        symbols = list(raw)
    else:
        raise ValidationError("observations 必须是字符串或符号数组")

    if len(symbols) == 0:
        raise EmptyObservationsError("观测串为空，长度至少为 1")

    known = set(alphabet)
    for pos, sym in enumerate(symbols):
        if sym not in known:
            raise UnknownSymbolError(
                f"位置 {pos} 的符号 {sym!r} 不在该档字母表内",
                details={"position": pos, "symbol": sym},
            )
    return symbols
