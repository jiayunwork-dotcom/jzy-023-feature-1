"""模型档的进程内 SQLite 存取。

单个共享连接加锁串行化写入；读取返回反序列化后的新字典，
调用方拿到的互不是同一份对象，并行解码互不写串。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from .errors import ModelExistsError, ModelNotFoundError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    spec TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class ModelStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def create(self, spec: dict) -> None:
        """登记新档；同名已存在时抛 ``ModelExistsError``。"""
        payload = json.dumps(spec, ensure_ascii=False)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO models (name, spec, created_at) VALUES (?, ?, ?)",
                    (spec["name"], payload, datetime.now(timezone.utc).isoformat()),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                raise ModelExistsError(f"模型档 {spec['name']!r} 已存在") from None

    def create_if_absent(self, spec: dict) -> None:
        """不存在才登记（内置示范档启动注册用）。"""
        payload = json.dumps(spec, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO models (name, spec, created_at) VALUES (?, ?, ?)",
                (spec["name"], payload, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def exists(self, name: str) -> bool:
        """该名字是否已登记（训练落库前的快速撞名检查）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM models WHERE name = ?", (name,)
            ).fetchone()
        return row is not None

    def get(self, name: str) -> dict:
        """按名取档；未登记抛 ``ModelNotFoundError``。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT spec FROM models WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            raise ModelNotFoundError(f"模型档 {name!r} 未登记")
        return json.loads(row["spec"])

    def list(self) -> list[dict]:
        """列出全部已登记档的概要。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, spec, created_at FROM models ORDER BY name"
            ).fetchall()
        summaries = []
        for row in rows:
            spec = json.loads(row["spec"])
            summaries.append(
                {
                    "name": row["name"],
                    "states": spec["states"],
                    "alphabet": spec["alphabet"],
                    "created_at": row["created_at"],
                }
            )
        return summaries

    def close(self) -> None:
        with self._lock:
            self._conn.close()
