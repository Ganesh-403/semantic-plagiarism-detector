import json
import os
from unittest.mock import patch

import responses

import src.core.webhook as webhook
from src.core.webhook import send_plagiarism_alert


@patch.dict(os.environ, {}, clear=True)
def test_send_plagiarism_alert_no_url():
    assert send_plagiarism_alert("DocA", "DocB", 0.95) is False


@patch.dict(
    os.environ,
    {
        "PLAGIARISM_WEBHOOK_URL": "https://mock-webhook.url",
        "APP_BASE_URL": "http://test-dashboard",
    },
)
@responses.activate
def test_send_plagiarism_alert_success():
    responses.add(
        responses.POST,
        "https://mock-webhook.url",
        json={"ok": True},
        status=200,
    )

    result = send_plagiarism_alert("student_essay.pdf", "wikipedia_source.pdf", 0.925)

    assert result is True
    assert len(responses.calls) == 1
    request = responses.calls[0].request
    assert request.url == "https://mock-webhook.url"
    assert request.method == "POST"
    payload = json.loads(request.body)
    assert "text" in payload
    assert "content" in payload
    assert "student_essay.pdf" in payload["text"]
    assert "wikipedia_source.pdf" in payload["text"]
    assert "92.5%" in payload["text"]
    assert "http://test-dashboard" in payload["text"]


@patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://mock-webhook.url"})
@responses.activate
def test_send_plagiarism_alert_network_failure():
    responses.add(
        responses.POST,
        "https://mock-webhook.url",
        body=responses.ConnectionError("Connection timed out"),
    )

    result = send_plagiarism_alert("DocA", "DocB", 0.99)

    assert result is False
    assert len(responses.calls) == 1

def test_dispatch_plagiarism_alert_background(mocker):
    from src.core.webhook import dispatch_plagiarism_alert
    
    mock_run = mocker.patch('src.core.synchronization.run_background')
    
    dispatch_plagiarism_alert("DocA", "DocB", 0.99)
    
    assert mock_run.call_count == 1
    # Check that a function was passed to run_background
    func = mock_run.call_args[0][0]
    assert callable(func)

def test_dispatch_plagiarism_alert_handles_exceptions(mocker):
    from src.core.webhook import dispatch_plagiarism_alert
    
    mock_send = mocker.patch('src.core.webhook.send_plagiarism_alert', side_effect=Exception("API Down"))
    mock_logger = mocker.patch('src.core.webhook.logger.exception')
    
    # We call the wrapped inner function manually to simulate background thread execution
    # First we intercept it
    inner_func = None
    def mock_run_bg(func, *args, **kwargs):
        nonlocal inner_func
        inner_func = func
    
    mocker.patch('src.core.synchronization.run_background', side_effect=mock_run_bg)
    
    dispatch_plagiarism_alert("DocA", "DocB", 0.99)
    
    # Simulate the thread executing it
    inner_func()
    
    mock_send.assert_called_once_with("DocA", "DocB", 0.99)
    mock_logger.assert_called_once_with("Webhook dispatch failed")
