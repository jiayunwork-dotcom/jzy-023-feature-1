"""内置的三状态分叉示范档。

A / B 是两个典型区段状态（分别偏好 x / y），自转概率高，Viterbi 能跟上
人为嵌入的区段切换。C 是吸收态陷阱：单条路径进入 C 每步收益
（1.0 * 0.5 = 0.5）不如留在 A/B（0.7 * 0.8 = 0.56），所以 Viterbi 全程
不踩 C；但概率质量每一步都在漏入 C 且只进不出，长序列末尾的逐时刻
后验因此偏向 C。于是同一份示范观测上，"后验逐时刻取最大"得到的符号串
与 Viterbi 路径必然不同 —— 用来卡住用后验冒充 Viterbi 的实现。
"""
from __future__ import annotations

DEMO_MODEL_NAME = "fork-demo"

# 人为嵌入两段区段切换：x 区段 -> y 区段 -> x 区段。
DEMO_OBSERVATION = "x" * 20 + "y" * 20 + "x" * 20


def demo_model_spec() -> dict:
    return {
        "name": DEMO_MODEL_NAME,
        "states": ["A", "B", "C"],
        "alphabet": ["x", "y"],
        "initial": [0.5, 0.5, 0.0],
        "transition": [
            [0.70, 0.05, 0.25],
            [0.05, 0.70, 0.25],
            [0.00, 0.00, 1.00],
        ],
        "emission": [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.5, 0.5],
        ],
        "demo_observation": DEMO_OBSERVATION,
    }
