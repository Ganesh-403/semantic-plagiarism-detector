"""Production can run without Redis; configured Redis must use credentials."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("redis_env, succeeds", [
    ({}, True),
    ({"REDIS_HOST": "redis.example"}, False),
    ({"REDIS_HOST": "redis.example", "REDIS_PASSWORD": "changeme"}, False),  # pragma: allowlist secret
    ({"REDIS_HOST": "redis.example", "REDIS_PASSWORD": "SafeRedisPassword123!"}, True),  # pragma: allowlist secret
    ({"REDIS_URL": "rediss://user:SafeRedisPassword123%21@redis.example:6379"}, True),  # pragma: allowlist secret
    ({"REDIS_URL": "redis://redis.example:6379"}, False),
])
def test_production_redis_configuration(tmp_path, redis_env, succeeds):
    env = os.environ.copy()
    for key in ["REDIS_URL", "REDIS_HOST", "REDIS_PASSWORD"]:
        env.pop(key, None)
    env.update(APP_ENV="production", JWT_SECRET_KEY="ProductionTestSecretWithEnoughCharacters123!",  # pragma: allowlist secret
               SPD_STATE_DIR=str(tmp_path))
    env.update(redis_env)
    result = subprocess.run([sys.executable, "-c", "import src.core.app_config"], env=env,
                            capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is succeeds, result.stderr
    assert "SafeRedisPassword123" not in result.stderr
