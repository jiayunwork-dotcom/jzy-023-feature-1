"""Flask HTTP 层：模型档登记、列出、按名解码。仅经 HTTP 对外。"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .decode import decode
from .demo import demo_model_spec
from .errors import ApiError, InvalidJsonError
from .store import ModelStore
from .training.engine import train
from .training.validation import validate_training_request
from .validation import validate_model_spec, validate_observations

DEFAULT_DB_PATH = "hmm_models.db"


def create_app(db_path: str | None = None, with_demo: bool = True) -> Flask:
    app = Flask(__name__)
    store = ModelStore(db_path or os.environ.get("HMM_DB_PATH", DEFAULT_DB_PATH))
    app.extensions["model_store"] = store
    if with_demo:
        store.create_if_absent(validate_model_spec(demo_model_spec()))

    @app.errorhandler(ApiError)
    def handle_api_error(err: ApiError):
        return jsonify(err.to_dict()), err.status_code

    @app.errorhandler(Exception)
    def handle_unexpected(err: Exception):
        if isinstance(err, HTTPException):
            return (
                jsonify({"error": {"type": "http_error", "message": err.description}}),
                err.code,
            )
        app.logger.exception("未预期的错误")
        return (
            jsonify({"error": {"type": "internal_error", "message": "服务器内部错误"}}),
            500,
        )

    def _json_body() -> dict:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise InvalidJsonError("请求体必须是 JSON 对象")
        return body

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "hmm-decode-service",
                "endpoints": {
                    "register": "POST /models",
                    "list": "GET /models",
                    "get": "GET /models/<name>",
                    "decode": "POST /models/<name>/decode",
                    "train": "POST /training",
                },
            }
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/models")
    def register_model():
        spec = validate_model_spec(_json_body())
        store.create(spec)
        return jsonify(spec), 201

    @app.get("/models")
    def list_models():
        return jsonify({"models": store.list()})

    @app.get("/models/<name>")
    def get_model(name: str):
        return jsonify(store.get(name))

    @app.post("/models/<name>/decode")
    def decode_model(name: str):
        model = store.get(name)  # 未登记的名字在这里抛 model_not_found
        body = _json_body()
        symbols = validate_observations(body.get("observations"), model["alphabet"])
        result = decode(model, symbols)
        return jsonify(
            {
                "model": model["name"],
                "observations": "".join(symbols),
                "length": len(symbols),
                "path": result.path,
                "log_joint_probability": result.log_joint_probability,
                "log_likelihood": result.log_likelihood,
                "posterior": result.posterior,
                "tie_break": result.tie_break,
            }
        )

    @app.post("/training")
    def train_model():
        body = _json_body()
        # 开训前挡掉一切不合法：状态数、维数、归一、未知符号、空批次、
        # 迭代上限、容差。不合法不会进入任何一次迭代。
        seed, sequences, max_iterations, tolerance = validate_training_request(body)

        index_of = {sym: i for i, sym in enumerate(seed["alphabet"])}
        sequences_idx = [[index_of[s] for s in seq] for seq in sequences]

        result = train(
            seed,
            sequences_idx,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        # 估出的参数按正式模型档过同一套登记校验，再用调用方给的名字落库；
        # 撞名沿用登记语义抛 model_exists，绝不悄悄覆盖。
        spec = validate_model_spec(
            result.to_spec(seed["name"], seed["states"], seed["alphabet"])
        )
        store.create(spec)

        return jsonify(
            {
                "model": spec["name"],
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
                "converged": result.converged,
                "max_iterations": result.max_iterations,
                "tolerance": result.tolerance,
                "n_sequences": result.n_sequences,
                "log_likelihoods": result.log_likelihoods,
                "final_log_likelihood": result.final_log_likelihood,
                "states": spec["states"],
                "alphabet": spec["alphabet"],
                "initial": spec["initial"],
                "transition": spec["transition"],
                "emission": spec["emission"],
            }
        ), 201

    return app
