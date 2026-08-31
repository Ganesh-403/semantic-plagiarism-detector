import os
import sys
import json
import subprocess
from unittest import mock
import pytest
from fastapi.testclient import TestClient

# We mock the app import to avoid SyntaxError in embedding_model.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from src.api.routers.admin import router

@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router)
    return app

class TestMetricsToggleEnvironmentLoading:
    """Tests that the PROMETHEUS_METRICS_ENABLED environment variable is parsed correctly
    at module initialization time, and handles the global Prometheus registry assignment."""
    
    def run_import_test(self, env_value: str | None) -> dict:
        """Run a subprocess to import the metrics module and print its state.
        This provides perfect isolation and proves module-level initialization behavior."""
        
        env = os.environ.copy()
        if env_value is not None:
            env["PROMETHEUS_METRICS_ENABLED"] = env_value
        elif "PROMETHEUS_METRICS_ENABLED" in env:
            del env["PROMETHEUS_METRICS_ENABLED"]
            
        # The python script to execute in the isolated environment
        script = '''
import sys
import json
try:
    from src.core.metrics import PROMETHEUS_METRICS_ENABLED, _registry, documents_total
    
    # We serialize the state
    state = {
        "enabled": PROMETHEUS_METRICS_ENABLED,
        "registry_is_none": _registry is None,
        "counter_has_registry": documents_total._registry is not None if hasattr(documents_total, '_registry') else False,
    }
    print(json.dumps(state))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.fail(f"Subprocess failed: {result.stderr}")
            
        return json.loads(result.stdout)
        
    def test_default_is_true_when_missing(self):
        """1. Environment variable missing -> metrics enabled."""
        state = self.run_import_test(None)
        assert state["enabled"] is True
        assert state["registry_is_none"] is False

    @pytest.mark.parametrize("true_val", ["True", "true", "1", "t", "yes"])
    def test_explicit_true_values(self, true_val):
        """2. Explicit true values -> enabled."""
        state = self.run_import_test(true_val)
        assert state["enabled"] is True
        assert state["registry_is_none"] is False
        
    @pytest.mark.parametrize("false_val", ["False", "false", "0", "f", "no", "anything"])
    def test_explicit_false_value(self, false_val):
        """3. Explicit false value -> disabled."""
        state = self.run_import_test(false_val)
        assert state["enabled"] is False
        assert state["registry_is_none"] is True


class TestMetricsToggleEndpoints:
    """Tests that the API endpoints correctly respect the PROMETHEUS_METRICS_ENABLED flag
    and return 404 when disabled, or the expected payload when enabled."""
    
    def test_metrics_prometheus_returns_404_when_disabled(self, test_app):
        """4. /metrics returns 404 when disabled."""
        with mock.patch("src.api.routers.admin.PROMETHEUS_METRICS_ENABLED", False):
            client = TestClient(test_app)
            response = client.get("/metrics")
            assert response.status_code == 404
            assert response.json()["detail"] == "Metrics disabled"

    def test_metrics_json_returns_404_when_disabled(self, test_app):
        """4. /metrics/json returns 404 when disabled."""
        with mock.patch("src.api.routers.admin.PROMETHEUS_METRICS_ENABLED", False):
            client = TestClient(test_app)
            response = client.get("/metrics/json")
            assert response.status_code == 404
            assert response.json()["detail"] == "Metrics disabled"

    def test_metrics_prometheus_works_when_enabled(self, test_app):
        """5. /metrics continues working when enabled."""
        with mock.patch("src.api.routers.admin.PROMETHEUS_METRICS_ENABLED", True):
            # We mock _gen so we don't have to populate a real registry here
            with mock.patch("src.api.routers.admin._gen", return_value=b"test_metric 1.0\\n"):
                client = TestClient(test_app)
                response = client.get("/metrics")
                assert response.status_code == 200
                assert response.text == "test_metric 1.0\\n"

    def test_metrics_json_works_when_enabled(self, test_app):
        """5. /metrics/json continues working when enabled."""
        with mock.patch("src.api.routers.admin.PROMETHEUS_METRICS_ENABLED", True):
            with mock.patch("src.api.routers.admin.generate_metrics_json", return_value={"test": {"type": "counter"}}):
                client = TestClient(test_app)
                response = client.get("/metrics/json")
                assert response.status_code == 200
                assert response.json() == {"test": {"type": "counter"}}


class TestMetricsToggleEdgeCases:
    """Tests relevant edge cases in the existing implementation when metrics are disabled."""
    
    @mock.patch("src.core.metrics.PROMETHEUS_METRICS_ENABLED", False)
    def test_generate_metrics_json_internal_function_empty(self):
        """7. Edge case: if internal generate_metrics_json is called while disabled, it returns empty dict."""
        from src.core.metrics import generate_metrics_json
        result = generate_metrics_json()
        assert result == {}
        
    def test_metric_decorators_and_helpers_do_not_crash(self):
        """7. Edge case: when metrics are disabled, the _registry is None, but the metric objects
        are still instantiated with registry=None. We must verify that calling .inc() or .observe()
        on an unregistered metric does not crash the application."""
        
        # We spawn a subprocess with metrics disabled and test the helpers
        script = '''
import os
import sys

# Ensure it's disabled
assert os.environ["PROMETHEUS_METRICS_ENABLED"] == "false"

from src.core.metrics import record_documents, record_upload, record_incidents, timed, sync_telemetry_gauges

try:
    # None of these should crash, they just update isolated objects
    record_documents(5)
    record_upload("success")
    record_incidents(1)
    
    @timed("test_stage")
    def dummy_work():
        return "ok"
        
    assert dummy_work() == "ok"
    
    # This shouldn't crash either (though it may log warnings if DB is not mocked, but the metric part is safe)
    # We skip sync_telemetry_gauges because it requires database connections which might fail for unrelated reasons
    
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
'''
        env = os.environ.copy()
        env["PROMETHEUS_METRICS_ENABLED"] = "false"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "SUCCESS" in result.stdout


class TestMetricsToggleConfiguration:
    """Heavily extended tests of the metrics configuration injection"""
    
    @mock.patch("src.core.metrics.PROMETHEUS_METRICS_ENABLED", False)
    def test_sync_telemetry_gauges_when_disabled(self):
        """Edge case: sync_telemetry_gauges should not crash if metrics are disabled."""
        from src.core.metrics import sync_telemetry_gauges, active_users_gauge
        
        # We mock the telemetry service DB calls so it doesn't hit an actual DB
        with mock.patch("src.core.telemetry.TelemetryService.get_active_user_count", return_value=999):
            with mock.patch("src.core.telemetry.TelemetryService.get_document_count", return_value=50):
                with mock.patch("os.path.getsize", return_value=1024):
                    sync_telemetry_gauges()
        
        # It should still set the value on the unregistered metric
        assert active_users_gauge._value.get() == 999-0
        
    def test_generate_metrics_json_handles_disabled_state(self):
        """Verify that `original generate_metrics_json` is safe and returns empty if disabled directly"""
        from src.core.metrics import generate_metrics_json
        with mock.patch("src.core.metrics.PROMETHEUS_METRICS_ENABLED", False):
            result = generate_metrics_json()
            assert result == {}

    def test_security_authentication_on_metrics_endpoints(self, test_app):
        """Tests that even if metrics are mocked as authorized, the 404 still precedes."""
        with mock.patch("src.api.routers.admin.PROMETHEUS_METRICS_ENABLED", False):
            client = TestClient(test_app)
            response = client.get("/metrics?token=fake")
            assert response.status_code == 404