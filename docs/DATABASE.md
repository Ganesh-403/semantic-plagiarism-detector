# Database Schema

This project uses two SQLite databases: `users.db` (authentication) and `corpus.db` (documents, embeddings, and plagiarism incidents).

## users.db

```mermaid
erDiagram
    USERS ||--o{ SECURITY_AUDIT_LOG : "logged by username"

    USERS {
        INTEGER id PK
        TEXT username UK "NOT NULL"
        TEXT password "NOT NULL"
        TEXT role "NOT NULL, default 'teacher'"
        INTEGER tour_completed "NOT NULL, default 0"
        TEXT otp_secret "nullable"
        INTEGER two_factor_enabled "NOT NULL, default 0"
        TEXT preferences "default '{}'"
        INTEGER is_active "NOT NULL, default 1"
        TEXT theme "NOT NULL, default 'light'"
        TEXT last_login_at "nullable"
    }

    SECURITY_AUDIT_LOG {
        INTEGER id PK
        TEXT event_type "NOT NULL"
        TEXT username "NOT NULL"
        TEXT timestamp "NOT NULL"
        TEXT details "nullable"
    }
```

**Indices:** `idx_users_role` (users.role), `idx_audit_log_username` (security_audit_log.username), `idx_audit_log_event_type` (security_audit_log.event_type)

> Note: `security_audit_log.username` is not a declared SQL foreign key (no `REFERENCES` clause), it's only linked by convention.

## corpus.db

```mermaid
erDiagram
    DOCUMENTS ||--o{ CHUNKS : "referenced by filename"
    DOCUMENTS ||--o{ PLAGIARISM_INCIDENTS : "referenced by filename"
    DOCUMENTS ||--o{ FALSE_POSITIVES : "referenced by filename"

    DOCUMENTS {
        INTEGER id PK
        TEXT filename UK "NOT NULL"
        TEXT file_hash UK "NOT NULL"
        TEXT upload_date "NOT NULL"
        TEXT class_section "nullable"
        TEXT student_name "nullable"
        TEXT assignment_title "nullable"
        TEXT detected_language "nullable"
        INTEGER is_deleted "default 0"
        TEXT deleted_at "nullable"
    }

    CHUNKS {
        INTEGER vector_id PK
        TEXT filename FK "NOT NULL, ON DELETE CASCADE"
        INTEGER chunk_index "NOT NULL"
        TEXT chunk_text "NOT NULL"
        BLOB embedding "NOT NULL"
    }

    PLAGIARISM_INCIDENTS {
        TEXT incident_id PK
        TEXT document_a "NOT NULL"
        TEXT document_b "NOT NULL"
        REAL similarity_score "NOT NULL"
        TEXT severity_rank "NOT NULL"
        TEXT review_status "NOT NULL, default 'Pending', CHECK IN ('Pending','Resolved')"
        TEXT date_flagged "NOT NULL"
        TEXT last_seen "NOT NULL"
        REAL threshold_at_time_of_flag "NOT NULL, default 0.0"
    }

    FALSE_POSITIVES {
        TEXT document_a PK
        TEXT document_b PK
        TIMESTAMP date_dismissed "default CURRENT_TIMESTAMP"
    }

    TRANSLATION_CACHE {
        TEXT text_hash PK
        TEXT foreign_text "NOT NULL"
        TEXT translated_text "NOT NULL"
        TEXT source_lang "nullable"
        TEXT target_lang "default 'en'"
        TIMESTAMP created_at "default CURRENT_TIMESTAMP"
    }
```

**Indices:** `idx_documents_upload_date`, `idx_documents_class_section`, `idx_chunks_filename`, `idx_incidents_status`, `idx_translation_cache_created_at`

> Note: `chunks.filename`, `plagiarism_incidents.document_a/document_b`, and `false_positives.document_a/document_b` all reference `documents.filename` logically; only `chunks.filename` is enforced with a real `FOREIGN KEY ... ON DELETE CASCADE`.

## Incident pagination

`get_all_incidents()` returns a maximum of 50 visible incidents by default.
Use `limit` and `offset` to request additional pages:

```python
from src.db.incidents import (
    get_all_incidents,
    get_total_incidents_count,
)

total = get_total_incidents_count()
first_page = get_all_incidents(limit=50, offset=0)
second_page = get_all_incidents(limit=50, offset=50)
```

Pagination uses stable ordering by descending flag date and ascending incident
ID. Both the page query and total count exclude incidents linked to
soft-deleted documents.
