"""测试共享的模型档构造助手。"""
from __future__ import annotations


def two_state_spec(name: str = "m", **overrides) -> dict:
    """一份合法的两状态基准档，可按字段覆盖。"""
    spec = {
        "name": name,
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [0.6, 0.4],
        "transition": [[0.7, 0.3], [0.4, 0.6]],
        "emission": [[0.9, 0.1], [0.2, 0.8]],
    }
    spec.update(overrides)
    return spec
