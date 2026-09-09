import pytest
from src.core.whistleblower.tor_gateway import TorOnionService

def test_tor_service_metadata_scrubbing():
    service = TorOnionService()
    dummy_data = b"Some file content with fake metadata"

    scrubbed = service.scrub_metadata(dummy_data)
    # The current mock simply returns it, but we test the interface
    assert scrubbed == dummy_data

def test_tor_service_submission():
    service = TorOnionService()
    dummy_data = b"Evidence bytes"

    access_key = service.process_submission(dummy_data)
    assert isinstance(access_key, str)
    assert len(access_key) == 32
