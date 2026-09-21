import pytest

from hmm_service.app import create_app


@pytest.fixture()
def app():
    # 每个测试一份独立的进程内 SQLite，互不影响。
    return create_app(db_path=":memory:")


@pytest.fixture()
def client(app):
    return app.test_client()
