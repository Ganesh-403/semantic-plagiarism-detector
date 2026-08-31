import pytest
from src.core.metrics import generate_metrics_json
from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram, Summary, Enum, Info

@pytest.fixture
def clean_registry(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.setattr('prometheus_client.REGISTRY', registry)
    # Also need to monkey patch generate_latest since generate_metrics_json calls it without args, using global REGISTRY
    import prometheus_client
    monkeypatch.setattr(prometheus_client, 'generate_latest', lambda: prometheus_client.exposition.generate_latest(registry))
    return registry

def test_generate_metrics_json_empty(clean_registry):
    payload = generate_metrics_json()
    assert payload == {}

def test_generate_metrics_json_counter(clean_registry):
    c = Counter('test_counter', 'A test counter', ['label1'])
    c.labels(label1='value1').inc(5.5)
    
    payload = generate_metrics_json()
    assert 'test_counter' in payload
    family = payload['test_counter']
    assert family['name'] == 'test_counter'
    assert family['type'] == 'counter'
    assert family['help'] == 'A test counter'
    
    samples = family['metrics']
    # There should be test_counter_total and test_counter_created
    total_samples = [s for s in samples if s['name'] == 'test_counter_total']
    assert len(total_samples) == 1
    sample = total_samples[0]
    assert sample['labels'] == {'label1': 'value1'}
    assert sample['value'] == 5.5

def test_generate_metrics_json_gauge(clean_registry):
    g = Gauge('test_gauge', 'A test gauge')
    g.set(42.0)
    
    payload = generate_metrics_json()
    assert 'test_gauge' in payload
    family = payload['test_gauge']
    assert family['name'] == 'test_gauge'
    assert family['type'] == 'gauge'
    
    samples = family['metrics']
    assert len(samples) == 1
    sample = samples[0]
    assert sample['name'] == 'test_gauge'
    assert sample['labels'] == {}
    assert sample['value'] == 42.0

def test_generate_metrics_json_histogram(clean_registry):
    h = Histogram('test_hist', 'A test histogram', ['method'])
    h.labels(method='get').observe(0.5)
    
    payload = generate_metrics_json()
    assert 'test_hist' in payload
    family = payload['test_hist']
    assert family['name'] == 'test_hist'
    assert family['type'] == 'histogram'
    
    samples = family['metrics']
    
    # Check buckets
    buckets = [s for s in samples if s['name'] == 'test_hist_bucket']
    assert len(buckets) > 0
    
    # Bucket for 0.5 should have value 1.0 (since we observed 0.5)
    bucket_0_5 = [s for s in buckets if s['labels'].get('le') == '0.5'][0]
    assert bucket_0_5['value'] == 1.0
    
    # Check count
    counts = [s for s in samples if s['name'] == 'test_hist_count']
    assert len(counts) == 1
    assert counts[0]['labels']['method'] == 'get'
    assert counts[0]['value'] == 1.0
    
    # Check sum
    sums = [s for s in samples if s['name'] == 'test_hist_sum']
    assert len(sums) == 1
    assert sums[0]['labels']['method'] == 'get'
    assert sums[0]['value'] == 0.5

def test_generate_metrics_json_multiple_labels(clean_registry):
    c = Counter('multi_label', 'Multi label', ['method', 'path', 'status'])
    c.labels(method='GET', path='/api', status='200').inc()
    c.labels(method='POST', path='/api', status='500').inc(2)
    
    payload = generate_metrics_json()
    samples = payload['multi_label']['metrics']
    
    totals = [s for s in samples if s['name'] == 'multi_label_total']
    assert len(totals) == 2
    
    get_sample = next(s for s in totals if s['labels']['method'] == 'GET')
    assert get_sample['labels'] == {'method': 'GET', 'path': '/api', 'status': '200'}
    assert get_sample['value'] == 1.0
    
    post_sample = next(s for s in totals if s['labels']['method'] == 'POST')
    assert post_sample['labels'] == {'method': 'POST', 'path': '/api', 'status': '500'}
    assert post_sample['value'] == 2.0

def test_generate_metrics_json_summary(clean_registry):
    s = Summary('test_summary', 'A test summary')
    s.observe(1.0)
    s.observe(2.0)
    
    payload = generate_metrics_json()
    assert 'test_summary' in payload
    family = payload['test_summary']
    assert family['type'] == 'summary'
    
    samples = family['metrics']
    count_sample = next(samp for samp in samples if samp['name'] == 'test_summary_count')
    assert count_sample['value'] == 2.0
    
    sum_sample = next(samp for samp in samples if samp['name'] == 'test_summary_sum')
    assert sum_sample['value'] == 3.0

def test_generate_metrics_json_info_and_enum(clean_registry):
    i = Info('test_info', 'A test info')
    i.info({'version': '1.0.0'})
    
    e = Enum('test_enum', 'A test enum', states=['starting', 'running', 'stopped'])
    e.state('running')
    
    payload = generate_metrics_json()
    
    assert 'test_info' in payload
    info_family = payload['test_info']
    assert info_family['type'] == 'info'
    info_sample = next(s for s in info_family['metrics'] if s['name'] == 'test_info_info')
    assert info_sample['labels']['version'] == '1.0.0'
    assert info_sample['value'] == 1.0
    
    assert 'test_enum' in payload
    enum_family = payload['test_enum']
    # Enum might be exposed as multiple gauge-like samples or untyped
    # Just check that it's present and properly parsed
    assert len(enum_family['metrics']) > 0
    running_sample = next(s for s in enum_family['metrics'] if s['labels'].get('test_enum') == 'running')
    assert running_sample['value'] == 1.0

