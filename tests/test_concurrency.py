"""多档并行解码：各自持有自己的路径与后验表，互不写串。"""
from concurrent.futures import ThreadPoolExecutor

from helpers import two_state_spec

OBS_A = "a" * 30 + "b" * 30
OBS_B = "ba" * 40


def _register_two_models(client):
    client.post("/models", json=two_state_spec("ma"))
    client.post("/models", json=two_state_spec(
        "mb",
        initial=[0.5, 0.5],
        transition=[[0.8, 0.2], [0.2, 0.8]],
        emission=[[0.7, 0.3], [0.3, 0.7]],
    ))


def test_parallel_decode_of_two_models_no_cross_talk(app):
    client = app.test_client()
    _register_two_models(client)

    # 先各自顺序解码一次，作为基准。
    expected = {}
    for name, obs in (("ma", OBS_A), ("mb", OBS_B)):
        body = client.post(f"/models/{name}/decode", json={"observations": obs}).get_json()
        expected[name] = body
    assert expected["ma"]["path"] != expected["mb"]["path"]

    def work(n):
        # 每个线程用独立客户端打满整个 HTTP 栈。
        c = app.test_client()
        name, obs = ("ma", OBS_A) if n % 2 == 0 else ("mb", OBS_B)
        res = c.post(f"/models/{name}/decode", json={"observations": obs})
        assert res.status_code == 200
        return name, res.get_json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(64)))

    for name, body in results:
        # 路径与后验表必须与该档自己的基准完全一致，不得串到另一档。
        assert body == expected[name]
        assert body["model"] == name
