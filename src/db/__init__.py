"""Public compatibility exports, loaded on demand without startup side effects."""

from importlib import import_module

_EXPORTS = {
    "AuthRepository": (".auth", "AuthRepository"),
    "add_user": (".auth", "add_user"),
    "auth_repo": (".auth", "auth_repo"),
    "clear_revocation_cache": (".auth", "clear_revocation_cache"),
    "delete_user": (".auth", "delete_user"),
    "disable_2fa": (".auth", "disable_2fa"),
    "enable_2fa": (".auth", "enable_2fa"),
    "generate_sso_state": (".auth", "generate_sso_state"),
    "get_2fa_status": (".auth", "get_2fa_status"),
    "get_all_users": (".auth", "get_all_users"),
    "get_recent_audit_events": (".auth", "get_recent_audit_events"),
    "get_user_active_status": (".auth", "get_user_active_status"),
    "get_user_last_login": (".auth", "get_user_last_login"),
    "get_user_role": (".auth", "get_user_role"),
    "init_db": (".auth", "init_db"),
    "is_token_revoked": (".auth", "is_token_revoked"),
    "is_user_active": (".auth", "is_user_active"),
    "revoke_token": (".auth", "revoke_token"),
    "revoke_all_user_refresh_tokens": (".auth", "revoke_all_user_refresh_tokens"),
    "set_password_change_required": (".auth", "set_password_change_required"),
    "set_user_active_status": (".auth", "set_user_active_status"),
    "store_sso_state": (".auth", "store_sso_state"),
    "update_password": (".auth", "update_password"),
    "update_user_profile": (".auth", "update_user_profile"),
    "validate_sso_state": (".auth", "validate_sso_state"),
    "verify_sso_state": (".auth", "verify_sso_state"),
    "verify_user": (".auth", "verify_user"),
    "BaseRepository": (".base", "BaseRepository"),
    "get_read_connection": (".common", "get_read_connection"),
    "create_connection": (".connection", "create_connection"),
    "get_connection": (".connection", "get_connection"),
    "CorpusRepository": (".corpus_db", "CorpusRepository"),
    "add_chunks": (".corpus_db", "add_chunks"),
    "add_document": (".corpus_db", "add_document"),
    "clear_all_data": (".corpus_db", "clear_all_data"),
    "corpus_repo": (".corpus_db", "corpus_repo"),
    "delete_document": (".corpus_db", "delete_document"),
    "get_all_documents": (".corpus_db", "get_all_documents"),
    "get_all_embeddings": (".corpus_db", "get_all_embeddings"),
    "get_chunk_registry": (".corpus_db", "get_chunk_registry"),
    "get_deleted_documents_count": (".corpus_db", "get_deleted_documents_count"),
    "get_document_by_hash": (".corpus_db", "get_document_by_hash"),
    "get_document_chunks_count": (".corpus_db", "get_document_chunks_count"),
    "get_documents_by_class": (".corpus_db", "get_documents_by_class"),
    "get_total_document_count": (".corpus_db", "get_total_document_count"),
    "get_unique_class_sections": (".corpus_db", "get_unique_class_sections"),
    "init_corpus_db": (".corpus_db", "init_corpus_db"),
    "restore_document": (".corpus_db", "restore_document"),
    "soft_delete_document": (".corpus_db", "soft_delete_document"),
    "IncidentsRepository": (".incidents", "IncidentsRepository"),
    "get_incidents_by_assignment": (".incidents", "get_incidents_by_assignment"),
    "get_incidents_repo": (".incidents", "get_incidents_repo"),
    "get_recent_incidents": (".incidents", "get_recent_incidents"),
    "log_incident": (".incidents", "log_incident"),
    "bulk_update_incident_status": (".incidents", "bulk_update_incident_status"),
    "add_false_positive": (".incidents", "add_false_positive"),
    "dismiss_incident": (".incidents", "dismiss_incident"),
    "AUTH_SCHEMA_VERSION": (".migrations", "AUTH_SCHEMA_VERSION"),
    "CORPUS_SCHEMA_VERSION": (".migrations", "CORPUS_SCHEMA_VERSION"),
    "column_exists": (".migrations", "column_exists"),
    "get_user_version": (".migrations", "get_user_version"),
    "index_exists": (".migrations", "index_exists"),
    "migrate_auth_database": (".migrations", "migrate_auth_database"),
    "migrate_corpus_database": (".migrations", "migrate_corpus_database"),
    "table_exists": (".migrations", "table_exists"),
    "incidents_repo": (".incidents", "get_incidents_repo"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    if name == "incidents_repo":
        return value()
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
