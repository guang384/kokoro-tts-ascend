"""pytest 根配置：把项目根目录放进 sys.path（各测试直接 import 顶层模块），
并在任何项目模块导入前固定健康目录到临时位置（心跳文件不落 /tmp 全局）。"""
import os
import sys
import tempfile

import fakeredis
import pytest

import proc_common

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault(
    "KOKORO_HEALTH_DIR",
    os.path.join(tempfile.mkdtemp(prefix="kokoro-test-health-"), "health"))


@pytest.fixture(autouse=True)
def fakeredis_client(monkeypatch):
    """redis 客户端全局兜底：每测试独立 FakeServer，_make_redis_client 指向
    它——任何测试路径的 redis 操作（含 gateway 终态清理等隐性调用）都在
    假实例上执行（Windows 无 AF_UNIX；真 redis-server 语义由镜像内测试
    与 test_ipc_crashsafe 覆盖）。需要真实 redis 的测试在用例内重新覆盖
    _make_redis_client。"""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
    monkeypatch.setattr(proc_common, "_make_redis_client", lambda: client)
    monkeypatch.setattr(proc_common, "_SHARED_REDIS", None)  # 单例重置：上个测试的 FakeServer 连接已死
    return client
