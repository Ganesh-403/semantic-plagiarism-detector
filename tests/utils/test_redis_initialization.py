"""Cold cache initialization must finish under concurrent requests."""

import subprocess
import sys
from pathlib import Path


def test_cold_singleton_initialization_does_not_deadlock():
    code = """
from concurrent.futures import ThreadPoolExecutor
from src.utils.redis_cache import RedisCache

connections = []
RedisCache._instance = None
RedisCache._connect = lambda self: connections.append(self)
with ThreadPoolExecutor(max_workers=16) as pool:
    instances = list(pool.map(lambda _: RedisCache.get_instance(), range(64)))
assert len({id(instance) for instance in instances}) == 1
assert len(connections) == 1
assert instances[0].set('probe', b'cached')
assert RedisCache.get_instance().get('probe') == b'cached'
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
