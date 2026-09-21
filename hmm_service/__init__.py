"""HMM 解码服务：模型档登记与按名解码（Viterbi + 逐时刻后验）。"""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """惰性导入，避免纯算法模块被强制依赖 Flask。"""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
