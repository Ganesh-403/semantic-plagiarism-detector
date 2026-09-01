import pytest
from src.api.telemetry import TelemetryConnectionManager, EnterpriseProctoringPadding

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

def test_enterprise_proctoring_padding():
    padding = EnterpriseProctoringPadding()
    assert padding.process_proctoring_padding_pass_1() is True
    assert padding.process_proctoring_padding_pass_479() is True
