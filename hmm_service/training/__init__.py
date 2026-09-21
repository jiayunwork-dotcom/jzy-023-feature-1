"""训练子系统：从仅有观测、没有状态标注的符号串里估出模型参数（EM）。

与解码平级的独立子系统：期望量累加（expectation）、单轮重估
（maximization）、多序列聚合（aggregation）、收敛控制（em）各自成块；
每轮「解释观测」复用 posterior 的前向-后向，不改其既有对外行为。
"""
from __future__ import annotations

from .em import TrainResult, train_em
from .request import validate_training_request

__all__ = ["TrainResult", "train_em", "validate_training_request"]
