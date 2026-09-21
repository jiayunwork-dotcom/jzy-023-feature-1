"""带类型的 API 错误。

所有对外错误都通过 ``type`` 字段区分，HTTP 层统一序列化成
``{"error": {"type": ..., "message": ..., "details": ...}}``。
"""
from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """所有可预期 API 错误的基类。"""

    status_code = 400
    error_type = "api_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        body: dict[str, Any] = {"type": self.error_type, "message": self.message}
        if self.details is not None:
            body["details"] = self.details
        return {"error": body}


class ValidationError(ApiError):
    """模型档或请求体校验失败（缺项、维数不符、未归一、取值非法等）。"""

    status_code = 400
    error_type = "validation_error"


class InvalidJsonError(ApiError):
    """请求体不是合法的 JSON 对象。"""

    status_code = 400
    error_type = "invalid_json"


class EmptyObservationsError(ApiError):
    """观测串为空。"""

    status_code = 400
    error_type = "empty_observations"


class EmptyTrainingSetError(ApiError):
    """训练观测批次为空（一条观测串都没有，或批次里夹了空串）。"""

    status_code = 400
    error_type = "empty_training_set"


class UnknownSymbolError(ApiError):
    """观测符号不在该档字母表内。"""

    status_code = 400
    error_type = "unknown_symbol"


class ModelNotFoundError(ApiError):
    """点名了未登记的模型档。"""

    status_code = 404
    error_type = "model_not_found"


class ModelExistsError(ApiError):
    """同名模型档已登记。"""

    status_code = 409
    error_type = "model_exists"


class ZeroProbabilityError(ApiError):
    """观测串在该模型下概率为 0，无法给出有意义的路径与后验。"""

    status_code = 422
    error_type = "zero_probability"
