from .auth import (
    AuthRepository,
    add_user,
    auth_repo,
    clear_revocation_cache,
    delete_user,
    disable_2fa,
    enable_2fa,
    generate_sso_state,
    get_2fa_status,
    get_all_users,
    get_recent_audit_events,
    get_user_active_status,
    get_user_last_login,
    get_user_role,
    init_db,
    is_token_revoked,
    is_user_active,
    revoke_token,
    revoke_all_user_refresh_tokens,
    set_password_change_required,
    set_user_active_status,
    store_sso_state,
    update_password,
    update_user_profile,
    validate_sso_state,
    verify_sso_state,
    verify_user,
)

from .base import BaseRepository
from .common import get_read_connection
from .connection import create_connection, get_connection
from .corpus_db import (
    CorpusRepository,
    add_chunks,
    add_document,
    clear_all_data,
    corpus_repo,
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_deleted_documents_count,
    get_document_by_hash,
    get_document_chunks_count,
    get_documents_by_class,
    get_total_document_count,
    get_unique_class_sections,
    init_corpus_db,
    restore_document,
    soft_delete_document,
)
from .incidents import (
    IncidentsRepository,
    get_incidents_by_assignment,
    get_incidents_repo,
    get_recent_incidents,
    log_incident,
    bulk_update_incident_status,
    add_false_positive,
    dismiss_incident,
)

__all__ = [
    "BaseRepository",
    "AuthRepository",
    "CorpusRepository",
    "IncidentsRepository",
    "create_connection",
    "get_connection",
    "auth_repo",
    "corpus_repo",
    "get_incidents_repo",
    "incidents_repo",
    "get_read_connection",
    "init_db",
    "verify_user",
    "get_user_role",
    "get_all_users",
    "add_user",
    "delete_user",
    "update_password",
    "revoke_all_user_refresh_tokens",
    "update_user_profile",
    "get_2fa_status",
    "enable_2fa",
    "disable_2fa",
    "get_user_active_status",
    "set_user_active_status",
    "is_user_active",
    "set_password_change_required",
    "get_recent_audit_events",
    "get_user_last_login",
    "init_corpus_db",
    "add_document",
    "get_document_by_hash",
    "get_all_documents",
    "soft_delete_document",
    "restore_document",
    "get_total_document_count",
    "get_deleted_documents_count",
    "get_documents_by_class",
    "add_chunks",
    "get_chunk_registry",
    "get_all_embeddings",
    "delete_document",
    "clear_all_data",
    "clear_revocation_cache",
    "is_token_revoked",
    "revoke_token",
    "get_document_chunks_count",
    "get_unique_class_sections",
    "get_incidents_by_assignment",
    "get_recent_incidents",
    "log_incident",
    "bulk_update_incident_status",
    "add_false_positive",
    "dismiss_incident",
]


from .migrations import AUTH_SCHEMA_VERSION as AUTH_SCHEMA_VERSION  # noqa: F401
from .migrations import CORPUS_SCHEMA_VERSION as CORPUS_SCHEMA_VERSION
from .migrations import column_exists as column_exists
from .migrations import get_user_version as get_user_version
from .migrations import index_exists as index_exists
from .migrations import migrate_auth_database as migrate_auth_database
from .migrations import migrate_corpus_database as migrate_corpus_database
from .migrations import table_exists as table_exists

__all__.extend(
    [
        "AUTH_SCHEMA_VERSION",
        "CORPUS_SCHEMA_VERSION",
        "column_exists",
        "get_user_version",
        "index_exists",
        "migrate_auth_database",
        "migrate_corpus_database",
        "table_exists",
    ]
)


def __getattr__(name: str):
    """PEP 562 module-level lazy attribute access.

    Preserves ``from src.db import incidents_repo`` for existing callers
    without eagerly constructing ``IncidentsRepository`` at package import
    time — forwards to :func:`src.db.incidents.get_incidents_repo`, which
    creates the singleton lazily on first actual access.
    """
    if name == "incidents_repo":
        return get_incidents_repo()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
