# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
test_docker_compose_override.py
-------------------------------
Tests to verify docker-compose.override.yml configuration for local dev hot-reloading (Issue #2944).
"""

from pathlib import Path

import yaml


def test_docker_compose_override_exists_and_mounts_directories():
    """Verify docker-compose.override.yml mounts ./src and ./app into /app container."""
    repo_root = Path(__file__).resolve().parents[2]
    override_path = repo_root / "docker-compose.override.yml"

    assert override_path.exists(), "docker-compose.override.yml must exist at repo root"

    with open(override_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(
        config, dict
    ), "docker-compose.override.yml must be valid YAML mapping"
    assert "services" in config, "'services' section must be defined"
    assert "app" in config["services"], "'app' service must be defined under services"
    assert (
        "volumes" in config["services"]["app"]
    ), "'volumes' must be defined for app service"

    volumes = config["services"]["app"]["volumes"]
    assert isinstance(volumes, list), "volumes must be a list"

    # Normalize volume paths
    volume_strings = [str(v) for v in volumes]

    has_src = any("./src:/app/src" in v for v in volume_strings)
    has_app = any("./app:/app/app" in v for v in volume_strings)

    assert has_src, "Must mount ./src to /app/src for hot-reloading"
    assert has_app, "Must mount ./app to /app/app for hot-reloading"
