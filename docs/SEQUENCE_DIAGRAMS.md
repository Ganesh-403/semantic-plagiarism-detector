# Sequence Diagrams

System interactions for core workflows in the Semantic Plagiarism Detector.

## 1. Authentication (Login)

JWT token generation flow via `/api/v1/auth/login`.

```mermaid
sequenceDiagram
    autonumber
    participant Client as User / Client
    participant API as API Server (/api/v1/auth/login)
    participant Auth as Security / JWT Module
    participant DB as Database

    Client->>API: POST /api/v1/auth/login (Credentials)
    API->>Auth: Validate Credentials
    
    opt If Database Validation is configured
        Auth->>DB: Query User Record
        DB-->>Auth: User Data / Password Hash
    end
    
    Auth-->>API: Credentials Validated
    API->>Auth: Request JWT Token generation (create_jwt_token)
    Auth-->>API: Signed JWT Session Token
    API-->>Client: 200 OK + { "token": "..." }
```

## 2. Document Scan & Incident Logging

Processing flow for `/api/v1/scan` including text extraction, transformer embeddings, FAISS vector search, and SQLite incident logging.

```mermaid
sequenceDiagram
    autonumber
    participant Client as User / Client
    participant API as API Server (/api/v1/scan)
    participant Extractor as Text Extractor & Chunker
    participant Embedder as Transformer Embedding Model
    participant FAISS as FAISS Vector Index
    participant DB as SQLite Database (Incidents)

    Client->>API: Upload Document (PDF/DOCX/TXT)
    
    API->>Extractor: Extract text from document
    Extractor-->>API: Raw Text
    
    API->>Extractor: Split text into chunks
    Extractor-->>API: List of Text Chunks
    
    API->>Embedder: Generate embeddings for chunks
    Embedder-->>API: Vector Embeddings (e.g., 384-dimensional)
    
    API->>FAISS: Vector Search (Nearest Neighbors for Chunks)
    FAISS-->>API: Top-K similar chunks from corpus
    
    API->>API: Compute Chunk & Document-level Similarity Scores
    
    alt Plagiarism Flagged (Similarity >= Threshold)
        API->>DB: Log Plagiarism Incident (severity, scores, matched chunks)
        DB-->>API: Incident successfully logged
    end
    
    API-->>Client: SimilarityCheckResponse (scores, matches, flags)
```
