# Semantic Plagiarism Detector - REST API Specification

Welcome to the **Semantic Plagiarism Detector REST API** documentation. This API provides programmatically accessible endpoints for Learning Management Systems (LMS), automated grading pipelines, and institutional submission portals to perform semantic plagiarism detection, retrieve stored plagiarism incidents, monitor service health, and execute administrative corpus management tasks.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Base URL & Protocol](#base-url--protocol)
- [Authentication & Authorization](#authentication--authorization)
- [Global Rate Limiting & Headers](#global-rate-limiting--headers)
- [Standard Error Response Schema](#standard-error-response-schema)
- [API Endpoints Overview](#api-endpoints-overview)
  - [1. Authentication Endpoint](#1-authentication-endpoint)
    - [`POST /api/v1/auth/login`](#post-apiv1authlogin)
  - [2. Plagiarism Scanning & Detection](#2-plagiarism-scanning--detection)
    - [`POST /api/v1/scan`](#post-apiv1scan)
    - [`GET /api/v1/incidents`](#get-apiv1incidents)
  - [3. Health & Telemetry Probes](#3-health--telemetry-probes)
    - [`GET /api/v1/healthz` / `GET /healthz`](#get-apiv1healthz--get-healthz)
    - [`GET /health`](#get-health)
    - [`GET /metrics`](#get-metrics)
    - [`GET /metrics/json`](#get-metricsjson)
  - [4. System Administration & Rate Limiting](#4-system-administration--rate-limiting)
    - [`GET /api/v1/rate_limit`](#get-apiv1rate_limit)
    - [`GET /api/v1/version`](#get-apiv1version)
    - [`POST /api/v1/clear`](#post-apiv1clear)
- [Client Integration Examples](#client-integration-examples)
  - [Python Integration (`requests` / `httpx`)](#python-integration-requests--httpx)
  - [JavaScript / Node.js Integration (`fetch`)](#javascript--nodejs-integration-fetch)
  - [cURL Command Reference Quicksheet](#curl-command-reference-quicksheet)

---

## Overview & Architecture

The REST API is built on **FastAPI** and uses high-performance asynchronous request handling. Under the hood, document processing leverages:

- **SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`)** for 384-dimensional dense semantic vector embeddings.
- **FAISS (Facebook AI Similarity Search)** for high-speed vector index retrieval.
- **SQLite Database (`corpus.db`)** for persistent document metadata, text chunk storage, and incident history.
- **Redis Cache & SlowAPI** for token bucket rate-limiting and response caching.

---

## Base URL & Protocol

All API requests must use `HTTP/1.1` or `HTTP/2` over TLS (`HTTPS`) in production environments.

- **Local Development Base URL:** `http://localhost:8000`
- **Production Base URL:** `https://api.plagiarism-detector.institution.edu`

---

## Authentication & Authorization

Secured endpoints require a Bearer token supplied in the standard HTTP `Authorization` request header.

```http
Authorization: Bearer <YOUR_API_BEARER_TOKEN>
```

### Public vs. Authenticated Endpoints

| Endpoint Path | Access Level | Description |
| :--- | :--- | :--- |
| `POST /api/v1/auth/login` | **Public** | Generate an authentication token |
| `GET /api/v1/healthz` | **Public** | Orchestration readiness & liveness health probe |
| `GET /healthz` | **Public** | Alias for health probe |
| `GET /health` | **Public** | Simple service health check |
| `GET /metrics` | **Public** | Prometheus format metrics export |
| `GET /metrics/json` | **Public** | JSON format operational metrics |
| `GET /api/v1/version` | **Public** | API version indicator |
| `GET /api/v1/rate_limit` | **Public** | Telemetry rate limit check |
| `POST /api/v1/scan` | **Bearer Auth** | Document upload & semantic scanning |
| `GET /api/v1/incidents` | **Bearer Auth** | Query flagged plagiarism incidents |
| `POST /api/v1/clear` | **Bearer Auth (Admin)** | Reset database and FAISS index |

---

## Global Rate Limiting & Headers

The API enforces rate limits using the SlowAPI token bucket algorithm per IP address.

### Rate Limit Response Headers

| Header Name | Type | Description |
| :--- | :--- | :--- |
| `X-RateLimit-Limit` | Integer | Total requests permitted per time window |
| `X-RateLimit-Remaining` | Integer | Number of remaining allowed requests |
| `X-RateLimit-Reset` | Integer | Seconds remaining until limit window resets |

When rate limits are exceeded, the API responds with HTTP status code `429 Too Many Requests`.

---

## Standard Error Response Schema

All error responses return a standardized JSON payload:

```json
{
  "error": true,
  "message": "Human-readable error explanation message.",
  "details": [
    {
      "field": "body.file",
      "message": "Field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## API Endpoints Overview

### 1. Authentication Endpoint

#### `POST /api/v1/auth/login`

**Summary:** Authenticate user and issue API session token.
**HTTP Method:** `POST`
**Content-Type:** `application/json`
**Authentication:** None (Public)

##### Request Body Parameters

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `email` | String | Yes | Registered user email address |
| `password` | String | Yes | Account password |

##### Example Request (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
        "email": "instructor@university.edu",
        "password": "SecurePassword123!"
      }'
```

##### Successful Response (`200 OK`)

```json
{
  "token": "dev-bearer-token"
}
```

##### Error Responses

- **`400 Bad Request`** (Invalid JSON or payload body):

```json
{
  "detail": "Invalid JSON request payload body."
}
```

- **`401 Unauthorized`** (Invalid credentials):

```json
{
  "detail": "Invalid credentials provided."
}
```

---

### 2. Plagiarism Scanning & Detection

#### `POST /api/v1/scan`

**Summary:** Scan an uploaded document against the indexed corpus for semantic plagiarism.
**HTTP Method:** `POST`
**Content-Type:** `multipart/form-data`
**Authentication:** `Bearer <TOKEN>`

##### Request Header Requirements

```http
Authorization: Bearer dev-bearer-token
Content-Type: multipart/form-data
```

##### Form Parameters

| Parameter Name | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `file` | Binary File | Yes | — | Document file (`.pdf`, `.docx`, `.txt`) |
| `threshold` | Float | No | `0.59` | Plagiarism similarity threshold (`0.0` to `1.0`) |
| `top_k` | Integer | No | `3` | Number of top paragraph-level chunk matches per doc (`1` to `10`) |

##### Example Request (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Authorization: Bearer dev-bearer-token" \
  -F "file=@/path/to/student_essay.pdf" \
  -F "threshold=0.65" \
  -F "top_k=3"
```

##### Successful Response (`200 OK` - Plagiarism Flagged)

```json
{
  "filename": "student_essay.pdf",
  "word_count": 1420,
  "chunk_count": 8,
  "plagiarism_flagged": true,
  "threshold_used": 0.65,
  "overall_document_similarity": 0.8842,
  "max_chunk_similarity": 0.9415,
  "matched_documents_count": 2,
  "matched_documents": [
    {
      "filename": "previous_submission_2025.pdf",
      "document_similarity_score": 0.8842,
      "max_chunk_similarity_score": 0.9415,
      "severity": "🔴 High",
      "flagged_chunks": [
        {
          "uploaded_chunk": "Artificial intelligence algorithms process large-scale vector datasets using dense matrix multiplication.",
          "matched_chunk": "Artificial intelligence models analyze large vector datasets via dense matrix multiplication operations.",
          "similarity_score": 0.9415
        },
        {
          "uploaded_chunk": "Transformers utilize multi-head self-attention mechanisms to construct contextual token embeddings.",
          "matched_chunk": "Transformer networks employ multi-head self-attention to generate contextual word embeddings.",
          "similarity_score": 0.8920
        }
      ]
    },
    {
      "filename": "reference_paper_alpha.docx",
      "document_similarity_score": 0.6720,
      "max_chunk_similarity_score": 0.7110,
      "severity": "🟡 Medium",
      "flagged_chunks": [
        {
          "uploaded_chunk": "The gradient descent optimizer adjusts weight parameters iteratively to minimize loss functions.",
          "matched_chunk": "Gradient descent optimization updates model weights iteratively to minimize target objective functions.",
          "similarity_score": 0.7110
        }
      ]
    }
  ]
}
```

##### Successful Response (`200 OK` - No Plagiarism Flagged)

```json
{
  "filename": "original_research_paper.txt",
  "word_count": 850,
  "chunk_count": 4,
  "plagiarism_flagged": false,
  "threshold_used": 0.65,
  "overall_document_similarity": 0.3120,
  "max_chunk_similarity": 0.4210,
  "matched_documents_count": 0,
  "matched_documents": []
}
```

##### Error Responses

- **`400 Bad Request`** (Empty file uploaded or missing filename):

```json
{
  "detail": "Uploaded file is empty (0 bytes)"
}
```

- **`401 Unauthorized`** (Missing or invalid Bearer token):

```json
{
  "detail": "Invalid or missing authentication token."
}
```

- **`415 Unsupported Media Type`** (Non-multipart form content type):

```json
{
  "detail": "Unsupported Media Type: Request must be multipart/form-data"
}
```

- **`422 Unprocessable Entity`** (Unextractable text or parameter error):

```json
{
  "error": true,
  "message": "Validation failed.",
  "details": [
    {
      "field": "body.file",
      "message": "Failed to extract readable text from the uploaded file.",
      "type": "value_error.extraction"
    }
  ]
}
```

---

#### `GET /api/v1/incidents`

**Summary:** Query recorded plagiarism incidents from the database with pagination support.
**HTTP Method:** `GET`
**Content-Type:** `application/json`
**Authentication:** `Bearer <TOKEN>`

##### Query Parameters

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | Integer | No | `50` | Maximum number of records to return (`1` to `500`) |
| `offset` | Integer | No | `0` | Number of initial records to skip for pagination |

##### Example Request (cURL)

```bash
curl -X GET "http://localhost:8000/api/v1/incidents?limit=10&offset=0" \
  -H "Authorization: Bearer dev-bearer-token"
```

##### Successful Response (`200 OK`)

```json
{
  "incidents": [
    {
      "incident_id": "INC-8F4A2C10B9E3",
      "document_a": "assignment_submission_alice.pdf",
      "document_b": "assignment_submission_bob.pdf",
      "similarity_score": 0.9415,
      "severity_rank": "High",
      "review_status": "Pending",
      "date_flagged": "2026-08-03T12:00:00Z",
      "threshold_at_time_of_flag": 0.59
    },
    {
      "incident_id": "INC-3D1E9F4A7C0B",
      "document_a": "lab_report_charlie.docx",
      "document_b": "reference_dataset_2025.pdf",
      "similarity_score": 0.7850,
      "severity_rank": "Medium",
      "review_status": "Resolved",
      "date_flagged": "2026-08-02T15:30:00Z",
      "threshold_at_time_of_flag": 0.59
    }
  ],
  "limit": 10,
  "offset": 0,
  "count": 2
}
```

##### Error Responses

- **`401 Unauthorized`**:

```json
{
  "detail": "Invalid or missing authentication token."
}
```

- **`500 Internal Server Error`**:

```json
{
  "detail": "Failed to fetch incidents: Database operational lock error."
}
```

---

### 3. Health & Telemetry Probes

#### `GET /api/v1/healthz` / `GET /healthz`

**Summary:** Health probe endpoint for Kubernetes / Docker container orchestration.
**HTTP Method:** `GET`
**Authentication:** None (Public)

##### Example Request (cURL)

```bash
curl -X GET http://localhost:8000/api/v1/healthz
```

##### Successful Response (`200 OK`)

```json
{
  "status": "ok",
  "db": "connected",
  "memory": "ok",
  "db_size_bytes": 4194304,
  "db_size_mb": 4.0
}
```

##### Degraded Service Response (`503 Service Unavailable`)

```json
{
  "status": "degraded",
  "db": "disconnected",
  "memory": "unavailable",
  "db_size_bytes": 0,
  "db_size_mb": 0.0
}
```

---

#### `GET /health`

**Summary:** Light application liveness check endpoint.
**HTTP Method:** `GET`
**Authentication:** None (Public)

##### Example Request (cURL)

```bash
curl -X GET http://localhost:8000/health
```

##### Successful Response (`200 OK`)

```json
{
  "status": "healthy",
  "service": "Semantic Plagiarism Detector API",
  "version": "1.0.0"
}
```

---

#### `GET /metrics`

**Summary:** Export system performance telemetry in standard Prometheus format.
**HTTP Method:** `GET`
**Content-Type:** `text/plain; version=0.0.4; charset=utf-8`
**Authentication:** None (Public)

##### Example Request (cURL)

```bash
curl -X GET http://localhost:8000/metrics
```

##### Example Plain Text Output

```text
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 12.45
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 284160000
# HELP api_requests_total Total number of API requests handled.
# TYPE api_requests_total counter
api_requests_total{endpoint="/api/v1/scan",status="200"} 142
```

---

#### `GET /metrics/json`

**Summary:** Export operational metrics in JSON structure.
**HTTP Method:** `GET`
**Authentication:** None (Public)

##### Example Request (cURL)

```bash
curl -X GET http://localhost:8000/metrics/json
```

##### Successful Response (`200 OK`)

```json
{
  "requests": {
    "total": 142,
    "scans_completed": 118,
    "incidents_flagged": 24
  },
  "performance": {
    "avg_scan_time_seconds": 0.42,
    "active_vector_dimension": 384
  }
}
```

---

### 4. System Administration & Rate Limiting

#### `GET /api/v1/rate_limit`

**Summary:** Retrieve current IP rate limit status.
**HTTP Method:** `GET`
**Authentication:** None (Public)

##### Example Request (cURL)

```bash
curl -X GET http://localhost:8000/api/v1/rate_limit
```

##### Successful Response (`200 OK`)

```json
{
  "limit": 100,
  "remaining": 85,
  "reset_in_seconds": 45
}
```

---

#### `GET /api/v1/version`

**Summary:** Fetch active API software version.
**HTTP Method:** `GET`
**Authentication:** None (Public)

##### Example Request (cURL)

```bash
curl -X GET http://localhost:8000/api/v1/version
```

##### Successful Response (`200 OK`)

```json
{
  "version": "1.0.0",
  "status": "active"
}
```

---

#### `POST /api/v1/clear`

**Summary:** Purge all stored documents, chunk vectors, and incidents from SQLite, reset the FAISS index, and invalidate Redis cache.
**HTTP Method:** `POST`
**Authentication:** `Bearer <TOKEN>` (Administrator role required)

##### Query Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `username` | String | Yes | Administrator username executing purge |

##### Example Request (cURL)

```bash
curl -X POST "http://localhost:8000/api/v1/clear?username=admin_user" \
  -H "Authorization: Bearer dev-bearer-token"
```

##### Successful Response (`200 OK`)

```json
{
  "status": "success",
  "message": "All documents, chunks, and plagiarism incidents have been cleared, and the FAISS index reset successfully."
}
```

##### Error Responses

- **`403 Forbidden`** (Non-administrator role):

```json
{
  "detail": "Forbidden: Only administrators are authorized to clear all documents."
}
```

---

## Client Integration Examples

### Python Integration (`requests` / `httpx`)

```python
import requests

API_BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "dev-bearer-token"

headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

# 1. Scan Document for Plagiarism
file_path = "sample_essay.pdf"
with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/pdf")}
    params = {"threshold": 0.60, "top_k": 5}
    response = requests.post(
        f"{API_BASE_URL}/api/v1/scan",
        headers=headers,
        files=files,
        params=params,
    )

scan_result = response.json()
print("Plagiarism Flagged:", scan_result.get("plagiarism_flagged"))
print("Matched Documents:", len(scan_result.get("matched_documents", [])))

# 2. Fetch Flagged Incidents History
incidents_res = requests.get(
    f"{API_BASE_URL}/api/v1/incidents",
    headers=headers,
    params={"limit": 20, "offset": 0},
)
print("Incident Records:", incidents_res.json())
```

---

### JavaScript / Node.js Integration (`fetch`)

```javascript
const fs = require('fs');
const FormData = require('form-data');
const fetch = require('node-fetch');

const API_BASE_URL = 'http://localhost:8000';
const BEARER_TOKEN = 'dev-bearer-token';

async function scanDocument(filePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  form.append('threshold', '0.65');
  form.append('top_k', '3');

  const response = await fetch(`${API_BASE_URL}/api/v1/scan`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${BEARER_TOKEN}`,
      ...form.getHeaders()
    },
    body: form
  });

  const data = await response.json();
  console.log('Scan Output:', JSON.stringify(data, null, 2));
}

scanDocument('student_submission.docx');
```

---

### cURL Command Reference Quicksheet

```bash
# 1. Authenticate & Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'

# 2. Upload Document for Plagiarism Scan
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Authorization: Bearer dev-bearer-token" \
  -F "file=@document.pdf" \
  -F "threshold=0.60" \
  -F "top_k=3"

# 3. Retrieve Flagged Incidents List
curl -X GET "http://localhost:8000/api/v1/incidents?limit=25&offset=0" \
  -H "Authorization: Bearer dev-bearer-token"

# 4. Check API Health Probe
curl -X GET http://localhost:8000/api/v1/healthz

# 5. Check Rate Limit Telemetry
curl -X GET http://localhost:8000/api/v1/rate_limit

# 6. Admin Clear All Documents & FAISS Index
curl -X POST "http://localhost:8000/api/v1/clear?username=admin" \
  -H "Authorization: Bearer dev-bearer-token"
```
