CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    upload_date TEXT NOT NULL,
    class_section TEXT,
    student_name TEXT,
    assignment_title TEXT,
    pdf_author TEXT,
    pdf_creation_date TEXT,
    pdf_title TEXT,
    tags TEXT,
    detected_language TEXT,
    owner TEXT,
    is_deleted INTEGER DEFAULT 0,
    deleted_at TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    vector_id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (filename)
    REFERENCES documents(filename)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS deleted_chunks (
    vector_id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    deleted_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS plagiarism_incidents (
    incident_id TEXT PRIMARY KEY,
    document_a TEXT NOT NULL,
    document_b TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    severity_rank TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'Pending'
        CHECK (review_status IN ('Pending', 'Resolved')),
    date_flagged TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    threshold_at_time_of_flag REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS false_positives (
    document_a TEXT,
    document_b TEXT,
    date_dismissed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dismissed_by TEXT DEFAULT 'admin',
    dismissal_reason TEXT,
    PRIMARY KEY (document_a, document_b)
);

CREATE TABLE IF NOT EXISTS scan_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    document_count INTEGER NOT NULL,
    avg_similarity REAL NOT NULL,
    max_similarity REAL NOT NULL,
    flagged_count INTEGER NOT NULL,
    threshold_used REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_incidents_date ON plagiarism_incidents(date_flagged);
