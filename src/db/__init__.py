from .auth import (
    add_user,
    delete_user,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_all_users,
    get_recent_audit_events,
    get_user_active_status,
    get_user_role,
    init_db,
    is_user_active,
    set_user_active_status,
    update_password,
    update_user_profile,
    verify_user,
)
from .common import get_read_connection
from .corpus_db import (
    add_chunks,
    add_document,
    clear_all_data,
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
)
from .incidents import get_incidents_by_assignment, get_recent_incidents, log_incident

__all__ = [
    "get_read_connection",
    "init_db",
    "verify_user",
    "get_user_role",
    "get_all_users",
    "add_user",
    "delete_user",
    "update_password",
    "update_user_profile",
    "get_2fa_status",
    "enable_2fa",
    "disable_2fa",
    "get_user_active_status",
    "set_user_active_status",
    "is_user_active",
    "update_user_profile",
    "init_corpus_db",
    "add_document",
    "get_document_by_hash",
    "get_all_documents",
    "add_chunks",
    "get_chunk_registry",
    "get_all_embeddings",
    "delete_document",
    "clear_all_data",
    "get_document_chunks_count",
    "get_unique_class_sections",
    "get_documents_by_class",
    "get_total_document_count",
    "get_deleted_documents_count",
    "get_recent_incidents",
    "get_recent_audit_events",
    "log_incident",
    "get_incidents_by_assignment",
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
