import pytest
from fastapi.testclient import TestClient
import prometheus_client
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary
from unittest.mock import patch

# We mock the app import to avoid SyntaxError in embedding_model.py
# Or better, we can just create a dummy FastAPI app and include the router for testing.
from fastapi import FastAPI
from src.api.routers.admin import router

@pytest.fixture
def clean_app(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.setattr('prometheus_client.REGISTRY', registry)
    monkeypatch.setattr('prometheus_client.exposition.generate_latest', lambda *args, **kwargs: prometheus_client.exposition.generate_latest(registry))
    
    app = FastAPI()
    app.include_router(router)
    return app, registry

def test_metrics_json_endpoint_empty(clean_app):
    app, registry = clean_app
    client = TestClient(app)
    response = client.get('/metrics/json')
    assert response.status_code == 200
    assert response.json() == {}

def test_metrics_json_endpoint_counter(clean_app):
    app, registry = clean_app
    c = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'], registry=registry)
    c.labels(method='GET', endpoint='/api/v1/scan').inc(10)
    c.labels(method='POST', endpoint='/api/v1/upload').inc(5)
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    assert response.status_code == 200
    
    data = response.json()
    assert 'api_requests' in data
    family = data['api_requests']
    assert family['type'] == 'counter'
    assert family['help'] == 'Total API requests'
    
    metrics = family['metrics']
    assert len(metrics) == 4  # 2 total + 2 created
    
    get_sample = next(m for m in metrics if m['name'] == 'api_requests_total' and m['labels'].get('method') == 'GET')
    assert get_sample['value'] == 10.0
    assert get_sample['labels']['endpoint'] == '/api/v1/scan'
    
    post_sample = next(m for m in metrics if m['name'] == 'api_requests_total' and m['labels'].get('method') == 'POST')
    assert post_sample['value'] == 5.0
    assert post_sample['labels']['endpoint'] == '/api/v1/upload'

def test_metrics_json_endpoint_gauge(clean_app):
    app, registry = clean_app
    g = Gauge('memory_usage_bytes', 'Current memory usage', ['process'], registry=registry)
    g.labels(process='worker_1').set(1024.5)
    g.labels(process='worker_2').set(2048.0)
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    assert response.status_code == 200
    
    data = response.json()
    assert 'memory_usage_bytes' in data
    family = data['memory_usage_bytes']
    assert family['type'] == 'gauge'
    
    metrics = family['metrics']
    assert len(metrics) == 2
    
    w1 = next(m for m in metrics if m['labels'].get('process') == 'worker_1')
    assert w1['name'] == 'memory_usage_bytes'
    assert w1['value'] == 1024.5
    
    w2 = next(m for m in metrics if m['labels'].get('process') == 'worker_2')
    assert w2['value'] == 2048.0

def test_metrics_json_endpoint_histogram(clean_app):
    app, registry = clean_app
    h = Histogram('response_time_seconds', 'Response time', ['endpoint'], registry=registry, buckets=(0.1, 0.5, 1.0))
    h.labels(endpoint='/api/fast').observe(0.05)
    h.labels(endpoint='/api/fast').observe(0.05)
    h.labels(endpoint='/api/slow').observe(0.8)
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    assert response.status_code == 200
    
    data = response.json()
    assert 'response_time_seconds' in data
    family = data['response_time_seconds']
    assert family['type'] == 'histogram'
    
    metrics = family['metrics']
    
    # fast endpoint checks
    fast_buckets = [m for m in metrics if m['name'] == 'response_time_seconds_bucket' and m['labels'].get('endpoint') == '/api/fast']
    assert len(fast_buckets) == 4 # 0.1, 0.5, 1.0, +Inf
    b_0_1 = next(b for b in fast_buckets if b['labels']['le'] == '0.1')
    assert b_0_1['value'] == 2.0
    
    fast_count = next(m for m in metrics if m['name'] == 'response_time_seconds_count' and m['labels'].get('endpoint') == '/api/fast')
    assert fast_count['value'] == 2.0
    
    fast_sum = next(m for m in metrics if m['name'] == 'response_time_seconds_sum' and m['labels'].get('endpoint') == '/api/fast')
    assert fast_sum['value'] == 0.1
    
    # slow endpoint checks
    slow_buckets = [m for m in metrics if m['name'] == 'response_time_seconds_bucket' and m['labels'].get('endpoint') == '/api/slow']
    b_0_5 = next(b for b in slow_buckets if b['labels']['le'] == '0.5')
    assert b_0_5['value'] == 0.0
    b_1_0 = next(b for b in slow_buckets if b['labels']['le'] == '1.0')
    assert b_1_0['value'] == 1.0
    
    slow_count = next(m for m in metrics if m['name'] == 'response_time_seconds_count' and m['labels'].get('endpoint') == '/api/slow')
    assert slow_count['value'] == 1.0

def test_metrics_json_endpoint_multiple_families(clean_app):
    app, registry = clean_app
    c = Counter('requests_total', 'Total requests', registry=registry)
    c.inc()
    g = Gauge('active_users', 'Active users', registry=registry)
    g.set(5)
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    data = response.json()
    
    assert 'requests' in data
    assert 'active_users' in data
    
    assert data['requests']['type'] == 'counter'
    assert data['active_users']['type'] == 'gauge'

def test_metrics_json_endpoint_content_type(clean_app):
    app, registry = clean_app
    client = TestClient(app)
    response = client.get('/metrics/json')
    assert response.headers['content-type'] == 'application/json'
    
def test_metrics_json_endpoint_malformed_no_metrics(clean_app):
    app, registry = clean_app
    # If no metrics are registered, should still return empty JSON object
    client = TestClient(app)
    response = client.get('/metrics/json')
    assert response.status_code == 200
    assert response.json() == {}

def test_metrics_json_schema_validation(clean_app):
    app, registry = clean_app
    c = Counter('schema_counter', 'Schema counter', registry=registry)
    c.inc(1.5)
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    assert response.status_code == 200
    
    data = response.json()
    # Import the schema
    from src.api.schemas import MetricFamily
    
    # Validate each family in the response against the Pydantic schema
    for family_name, family_data in data.items():
        # This will raise a ValidationError if the data doesn't match the schema
        validated_family = MetricFamily(**family_data)
        assert validated_family.name == family_name
        assert validated_family.type in ('counter', 'gauge', 'histogram', 'summary', 'untyped')

def test_metrics_json_endpoint_summary(clean_app):
    app, registry = clean_app
    s = Summary('api_payload_size_bytes', 'Size of API payloads', ['endpoint'], registry=registry)
    s.labels(endpoint='/api/upload').observe(1024)
    s.labels(endpoint='/api/upload').observe(2048)
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    data = response.json()
    
    assert 'api_payload_size_bytes' in data
    family = data['api_payload_size_bytes']
    assert family['type'] == 'summary'
    
    metrics = family['metrics']
    count_sample = next(m for m in metrics if m['name'] == 'api_payload_size_bytes_count')
    assert count_sample['value'] == 2.0
    
    sum_sample = next(m for m in metrics if m['name'] == 'api_payload_size_bytes_sum')
    assert sum_sample['value'] == 3072.0

def test_metrics_json_endpoint_unicode_labels(clean_app):
    app, registry = clean_app
    c = Counter('unicode_test', 'Unicode', ['path'], registry=registry)
    c.labels(path='/api/v1/測試').inc(1)
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    data = response.json()
    
    metrics = data['unicode_test']['metrics']
    sample = next(m for m in metrics if m['name'] == 'unicode_test_total')
    assert sample['labels']['path'] == '/api/v1/測試'

def test_metrics_json_endpoint_large_number_of_metrics(clean_app):
    app, registry = clean_app
    g = Gauge('dynamic_gauge', 'Dynamic', ['id'], registry=registry)
    for i in range(100):
        g.labels(id=str(i)).set(i * 1.5)
        
    client = TestClient(app)
    response = client.get('/metrics/json')
    data = response.json()
    
    metrics = data['dynamic_gauge']['metrics']
    assert len(metrics) == 100
    for i in range(100):
        sample = next(m for m in metrics if m['labels']['id'] == str(i))
        assert sample['value'] == i * 1.5

def test_metrics_json_endpoint_sanitization(clean_app):
    app, registry = clean_app
    # Prometheus metric names are automatically sanitized, but we want to ensure
    # our JSON endpoint correctly forwards the sanitized names
    c = Counter('dirty_metric_name_1', 'Dirty name', registry=registry)
    c.inc()
    
    client = TestClient(app)
    response = client.get('/metrics/json')
    data = response.json()
    
    assert 'dirty_metric_name_1' in data
    assert data['dirty_metric_name_1']['name'] == 'dirty_metric_name_1'
