"""Flask HTTP 层：模型档登记、列出、按名解码。仅经 HTTP 对外。"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .decode import decode
from .demo import demo_model_spec
from .errors import ApiError, InvalidJsonError
from .store import ModelStore
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

    return app
