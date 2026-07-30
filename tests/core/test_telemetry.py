from unittest.mock import patch


from src.core.telemetry import TelemetryService

# ---------------------------------------------------------
# Test Active User Count
# ---------------------------------------------------------

def test_telemetry_cache_hit():
    """
    Test that TelemetryService.get_active_user_count correctly returns a cached value 
    without querying the DB.
    """
    with patch('src.core.telemetry.get_cache') as mock_get_cache, \
         patch('src.core.telemetry.get_user_count') as mock_get_user_count:
        
        mock_get_cache.return_value = "42"
        
        count = TelemetryService.get_active_user_count()
        
        assert count == 42
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_USER_COUNT)
        mock_get_user_count.assert_not_called()

def test_telemetry_cache_miss():
    """
    Test that on cache miss, the service queries the DB and populates the cache.
    """
    with patch('src.core.telemetry.get_cache') as mock_get_cache, \
         patch('src.core.telemetry.get_user_count') as mock_get_user_count, \
         patch('src.core.telemetry.set_cache') as mock_set_cache:
        
        mock_get_cache.return_value = None
        mock_get_user_count.return_value = 17
        
        count = TelemetryService.get_active_user_count()
        
        assert count == 17
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_USER_COUNT)
        mock_get_user_count.assert_called_once()
        mock_set_cache.assert_called_once_with(
            TelemetryService.CACHE_KEY_USER_COUNT, 
            "17", 
            expire=TelemetryService.CACHE_TTL_SECONDS
        )

def test_telemetry_db_failure():
    """
    Test that if the database lookup fails entirely, the service handles it gracefully.
    """
    with patch('src.core.telemetry.get_cache') as mock_get_cache, \
         patch('src.core.telemetry.get_user_count') as mock_get_user_count:
        
        mock_get_cache.return_value = None
        mock_get_user_count.side_effect = Exception("DB Connection Lost")
        
        count = TelemetryService.get_active_user_count()
        
        assert count == 0

# ---------------------------------------------------------
# Test Document Count
# ---------------------------------------------------------

def test_telemetry_doc_count_cache_hit():
    """
    Test that TelemetryService.get_document_count hits the cache.
    """
    with patch('src.core.telemetry.get_cache') as mock_get_cache, \
         patch('src.core.telemetry.get_all_documents') as mock_get_all_documents:
        
        mock_get_cache.return_value = "99"
        
        count = TelemetryService.get_document_count()
        
        assert count == 99
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_DOC_COUNT)
        mock_get_all_documents.assert_not_called()

def test_telemetry_doc_count_cache_miss():
    """
    Test that on cache miss, get_document_count queries DB and populates cache.
    """
    with patch('src.core.telemetry.get_cache') as mock_get_cache, \
         patch('src.core.telemetry.get_all_documents') as mock_get_all_documents, \
         patch('src.core.telemetry.set_cache') as mock_set_cache:
        
        mock_get_cache.return_value = None
        mock_get_all_documents.return_value = [{"doc": 1}, {"doc": 2}, {"doc": 3}]
        
        count = TelemetryService.get_document_count()
        
        assert count == 3
        mock_get_cache.assert_called_once_with(TelemetryService.CACHE_KEY_DOC_COUNT)
        mock_get_all_documents.assert_called_once()
        mock_set_cache.assert_called_once_with(
            TelemetryService.CACHE_KEY_DOC_COUNT, 
            "3", 
            expire=TelemetryService.CACHE_TTL_SECONDS
        )

def test_telemetry_doc_db_failure():
    """
    Test that if document DB query fails, service falls back safely.
    """
    with patch('src.core.telemetry.get_cache') as mock_get_cache, \
         patch('src.core.telemetry.get_all_documents') as mock_get_all_documents:
        
        mock_get_cache.return_value = None
        mock_get_all_documents.side_effect = Exception("DB Fault")
        
        count = TelemetryService.get_document_count()
        
        assert count == 0

# ---------------------------------------------------------
# Test Force Refresh
# ---------------------------------------------------------

def test_telemetry_force_refresh():
    """
    Test that force_refresh bypasses the get_cache check and immediately updates cache.
    """
    with patch('src.core.telemetry.get_user_count') as mock_get_user_count, \
         patch('src.core.telemetry.get_all_documents') as mock_get_all_documents, \
         patch('src.core.telemetry.set_cache') as mock_set_cache:
        
        mock_get_user_count.return_value = 100
        mock_get_all_documents.return_value = [1] * 550
        
        TelemetryService.force_refresh_metrics()
        
        mock_get_user_count.assert_called_once()
        mock_get_all_documents.assert_called_once()
        assert mock_set_cache.call_count == 2
