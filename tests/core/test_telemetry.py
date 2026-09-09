import pytest
from src.api.telemetry import TelemetryConnectionManager

@pytest.mark.asyncio
async def test_telemetry_connection_manager():
    manager = TelemetryConnectionManager()

    class MockWebsocket:
        def __init__(self):
            self.accepted = False
        async def accept(self):
            self.accepted = True

    ws = MockWebsocket()
    await manager.connect(ws)
    assert ws.accepted is True
    assert ws in manager.active_connections

    manager.disconnect(ws)
    assert ws not in manager.active_connections
