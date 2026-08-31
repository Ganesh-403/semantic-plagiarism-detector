# REST API Integration Guide for LMS Platforms

This guide provides step-by-step instructions for integrating the Semantic Plagiarism Detector API with popular Learning Management Systems (LMS) including Canvas, Moodle, and Blackboard.

## Overview

The Semantic Plagiarism Detector API allows LMS platforms to programmatically:
- Upload student assignments for plagiarism checking
- Compare submissions against a corpus of existing documents
- Receive real-time webhook alerts for high-similarity matches
- Process detailed similarity reports

### Quick Start

```bash
# Start the API server
uvicorn src.api.app:app --reload --port 8000

# Set your API token
export API_BEARER_TOKEN="your-secret-token"
```

## Table of Contents

- [API Endpoints](#api-endpoints)
- [Authentication](#authentication)
- [Uploading Documents for Scanning](#uploading-documents-for-scanning)
- [Processing Scan Results](#processing-scan-results)
- [Webhook Alerts](#webhook-alerts)
- [Error Handling](#error-handling)
- [LMS-Specific Integration Examples](#lms-specific-integration-examples)

---

## API Endpoints

### Health Check

Check if the API is running and ready.

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Semantic Plagiarism Detector API",
  "version": "1.0.0"
}
```

### Scan Document

Upload and analyze a document for plagiarism.

```
POST /api/v1/scan
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.59 | Similarity threshold for flagging plagiarism (0.0 to 1.0) |
| `top_k` | int | 3 | Number of top matching paragraph pairs per matched document (1 to 10) |

**Request Headers:**
| Header | Value |
|--------|-------|
| `Authorization` | `Bearer <your-api-token>` |

**Request Body:** Form data with `file` field

| Field | Type | Description |
|-------|------|-------------|
| `file` | multipart/form-data | Document file (.pdf, .docx, .txt) |

**Response:**
```json
{
  "filename": "student_essay.pdf",
  "word_count": 480,
  "chunk_count": 5,
  "plagiarism_flagged": true,
  "threshold_used": 0.59,
  "overall_document_similarity": 0.8523,
  "max_chunk_similarity": 0.9125,
  "matched_documents_count": 1,
  "matched_documents": [
    {
      "filename": "course_source_material.pdf",
      "document_similarity_score": 0.8523,
      "max_chunk_similarity_score": 0.9125,
      "severity": "🔴 High",
      "flagged_chunks": [
        {
          "uploaded_chunk": "Artificial Intelligence is rapidly reshaping higher education...",
          "matched_chunk": "AI models are transforming modern academic institutions...",
          "similarity_score": 0.9125
        }
      ]
    }
  ]
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Original filename uploaded |
| `word_count` | integer | Number of words in the document |
| `chunk_count` | integer | Number of text chunks created |
| `plagiarism_flagged` | boolean | Whether any matches were found |
| `threshold_used` | float | Threshold value used for comparison |
| `overall_document_similarity` | float | Document-level similarity score |
| `max_chunk_similarity` | float | Highest chunk-level similarity score |
| `matched_documents_count` | integer | Number of matched documents |
| `matched_documents` | array | List of matched documents with details |

**matched_documents fields:**
| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Name of matched document in corpus |
| `document_similarity_score` | float | Document-level cosine similarity |
| `max_chunk_similarity_score` | float | Highest chunk-to-chunk similarity |
| `severity` | string | "🔴 High" (≥0.90) or "🟡 Medium" (≥0.75) |
| `flagged_chunks` | array | List of matching paragraph pairs |

---

## Authentication

The API uses Bearer Token authentication. Include your API token in the `Authorization` header.

### Setting Your API Token

Set the `API_BEARER_TOKEN` environment variable before starting the server:

```bash
export API_BEARER_TOKEN="super-secret-token-12345"
```

**Default token (development only):** `dev-bearer-token`

**Important:** Change the default token in production!

### Header Format

```http
Authorization: Bearer your-api-token-here
```

### Authentication Errors

| Status Code | Response | Description |
|-------------|----------|-------------|
| 401 | `{"detail": "Invalid or missing authentication token."}` | Missing or invalid token |
| 403 | `{"detail": "Invalid or missing authentication token."}` | Token validation failed |

---

## Uploading Documents for Scanning

### Supported File Formats

- **PDF** (.pdf) - Native text extraction with OCR fallback for scanned documents
- **Word** (.docx) - Microsoft Word documents
- **Text** (.txt) - Plain text files
- **Markdown** (.md) - Markdown files
- **ZIP** (.zip) - Compressed archives containing supported files

### Scanning Workflow

1. Student submits assignment via LMS
2. LMS downloads file content
3. LMS sends file to plagiarism API
4. API returns similarity analysis

### Step-by-Step Example

#### 1. Student Submits Assignment

Student uploads `assignment1.pdf` to Canvas.

#### 2. LMS Retrieves File

The LMS makes the file available as bytes/stream.

#### 3. Send to Plagiarism API

**Python Example:**
```python
import requests

API_URL = "http://localhost:8000"
API_TOKEN = "your-api-token"


def scan_document(file_path: str, threshold: float = 0.59):
    url = f"{API_URL}/api/v1/scan"
    params = {"threshold": threshold}
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "application/pdf")}
        response = requests.post(url, headers=headers, params=params, files=files)

    return response.json()


result = scan_document("assignment1.pdf")
print(f"Plagiarism flagged: {result['plagiarism_flagged']}")
```

**JavaScript (Node.js) Example:**
```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

const API_URL = 'http://localhost:8000';
const API_TOKEN = 'your-api-token';

async function scanDocument(filePath, threshold = 0.59) {
  const url = `${API_URL}/api/v1/scan?threshold=${threshold}`;
  const headers = {
    'Authorization': `Bearer ${API_TOKEN}`
  };

  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));

  const response = await axios.post(url, form, { headers });
  return response.data;
}

scanDocument('assignment1.pdf')
  .then(result => {
    console.log(`Plagiarism flagged: ${result.plagiarism_flagged}`);
  });
```

**PHP Example:**
```php
<?php
$apiUrl = 'http://localhost:8000';
$apiToken = 'your-api-token';
$filePath = 'assignment1.pdf';

function scanDocument($filePath, $threshold = 0.59) {
    global $apiUrl, $apiToken;

    $url = $apiUrl . '/api/v1/scan?threshold=' . $threshold;
    $ch = curl_init();

    $postFields = [
        'file' => new CURLFile($filePath)
    ];

    $headers = [
        'Authorization: Bearer ' . $apiToken
    ];

    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $postFields);
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

    $response = curl_exec($ch);
    curl_close($ch);

    return json_decode($response, true);
}

$result = scanDocument($filePath);
echo "Plagiarism flagged: " . ($result['plagiarism_flagged'] ? 'Yes' : 'No') . "\n";
```

### Handling Large Files

For large files or batch scanning:

```python
# Python - Process multiple files
def scan_multiple_documents(file_paths: list, threshold: float = 0.59):
    results = {}
    for path in file_paths:
        try:
            result = scan_document(path, threshold)
            results[path] = result
        except Exception as e:
            results[path] = {"error": str(e)}
    return results
```

---

## Processing Scan Results

### Understanding the Response

#### Low Severity (No Flag)
```json
{
  "filename": "assignment1.pdf",
  "word_count": 450,
  "chunk_count": 4,
  "plagiarism_flagged": false,
  "threshold_used": 0.59,
  "overall_document_similarity": 0.32,
  "max_chunk_similarity": 0.45,
  "matched_documents_count": 0,
  "matched_documents": []
}
```

#### High Severity (Flagged)
```json
{
  "filename": "assignment2.pdf",
  "word_count": 520,
  "chunk_count": 6,
  "plagiarism_flagged": true,
  "threshold_used": 0.59,
  "overall_document_similarity": 0.87,
  "max_chunk_similarity": 0.92,
  "matched_documents_count": 1,
  "matched_documents": [
    {
      "filename": "reference_essay.pdf",
      "document_similarity_score": 0.87,
      "max_chunk_similarity_score": 0.92,
      "severity": "🔴 High",
      "flagged_chunks": [
        {
          "uploaded_chunk": "The quick brown fox jumps over the lazy dog.",
          "matched_chunk": "A fast brown fox leaps over a lazy dog.",
          "similarity_score": 0.92
        }
      ]
    }
  ]
}
```

### Processing Logic

#### Python
```python
def process_scan_result(result: dict):
    """Process a scan result and return severity information."""
    if not result.get("plagiarism_flagged"):
        return {"status": "clean", "score": result["max_chunk_similarity"]}

    flagged_matches = result["matched_documents"]
    highest_match = max(flagged_matches, key=lambda x: x["max_chunk_similarity_score"])

    severity = (
        "high" if highest_match["max_chunk_similarity_score"] >= 0.90 else "medium"
    )

    return {
        "status": "flagged",
        "score": highest_match["max_chunk_similarity_score"],
        "severity": severity,
        "matched_document": highest_match["filename"],
        "flagged_chunks": highest_match["flagged_chunks"],
    }
```

#### JavaScript
```javascript
function processScanResult(result) {
    if (!result.plagiarism_flagged) {
        return {
            status: 'clean',
            score: result.max_chunk_similarity
        };
    }

    const flaggedMatches = result.matched_documents;
    const highestMatch = flaggedMatches.reduce((max, m) =>
        m.max_chunk_similarity_score > max.max_chunk_similarity_score ? m : max
    );

    const severity = highestMatch.max_chunk_similarity_score >= 0.90 ? 'high' : 'medium';

    return {
        status: 'flagged',
        score: highestMatch.max_chunk_similarity_score,
        severity: severity,
        matchedDocument: highestMatch.filename,
        flaggedChunks: highestMatch.flagged_chunks
    };
}
```

#### PHP
```php
function processScanResult($result) {
    if (!$result['plagiarism_flagged']) {
        return [
            'status' => 'clean',
            'score' => $result['max_chunk_similarity']
        ];
    }

    $flaggedMatches = $result['matched_documents'];
    $highestMatch = array_reduce($flaggedMatches, function($carry, $item) {
        return ($item['max_chunk_similarity_score'] > $carry['max_chunk_similarity_score'])
            ? $item : $carry;
    });

    $severity = $highestMatch['max_chunk_similarity_score'] >= 0.90 ? 'high' : 'medium';

    return [
        'status' => 'flagged',
        'score' => $highestMatch['max_chunk_similarity_score'],
        'severity' => $severity,
        'matchedDocument' => $highestMatch['filename'],
        'flaggedChunks' => $highestMatch['flagged_chunks']
    ];
}
```

### Storing Results in LMS Database

```python
# Example: Store results in Canvas LMS gradebook
def store_in_gradebook(student_id, assignment_id, result, canvas_api):
    score = result["max_chunk_similarity"]
    status = "flagged" if result["plagiarism_flagged"] else "clean"

    # Update LMS grade or status
    canvas_api.update_submission_status(
        assignment_id,
        student_id,
        {"graded": True, "score": score, "plagiarism_status": status},
    )
```

---

## Webhook Alerts

### Overview

Webhooks notify external systems in real-time when high-similarity matches (≥90%) are detected. This enables immediate notification to instructors.

### Configuration

Set the `PLAGIARISM_WEBHOOK_URL` environment variable:

```bash
export PLAGIARISM_WEBHOOK_URL="YOUR-WEBHOOK-URL-HERE"
```

**Supported Webhook Types:**
- **Slack Webhooks** - Standard Slack incoming webhooks
- **Discord Webhooks** - Discord webhook URLs
- **Custom Webhooks** - Any HTTP endpoint accepting JSON

### Webhook Message Format

```json
{
  "text": "🚨 *Plagiarism Alert!* Student document *student1.pdf* matches *student2.pdf* by *95.5%*.\nReview details here: http://localhost:8501",
  "content": "🚨 *Plagiarism Alert!* Student document *student1.pdf* matches *student2.pdf* by *95.5%*.\nReview details here: http://localhost:8501"
}
```

### Webhook Message Structure

| Field | Description |
|-------|-------------|
| `text` | Slack-compatible message format |
| `content` | Discord-compatible message format |
| Both fields contain the same message for maximum compatibility. |

### Custom Webhook Integration

If you need custom webhook behavior, modify the payload in `src/core/webhook.py`:

```python
# In webhook.py, modify the payload dict:
payload = {
    "custom_field": "value",
    "similarity": similarity,
    "doc_a": doc_a,
    "doc_b": doc_b,
}
```

### Webhook URL Examples

Replace the placeholders with your actual webhook URL:

**Slack:**
```
https://hooks.slack.com/services/{YOUR-SUBDOMAIN}/{YOUR-ID}/{YOUR-TOKEN}
```

**Discord:**
```
https://discord.com/api/webhooks/{YOUR-WEBHOOK-ID}/{YOUR-TOKEN}
```

**Custom HTTP Endpoint:**
```
https://your-lms.com/api/plagiarism/alerts
```

---

## Error Handling

### Common Errors

#### 400 Bad Request
```json
{
  "detail": "Uploaded file is empty."
}
```
**Cause:** File upload contains no data.

#### 422 Unprocessable Entity
```json
{
  "detail": "Failed to extract readable text from the uploaded file."
}
```
**Cause:** File format is not supported or text extraction failed.

#### 401/403 Unauthorized
```json
{
  "detail": "Invalid or missing authentication token."
}
```
**Cause:** Missing or invalid API token.

### Retry Strategy

```python
import time
import requests
from requests.exceptions import RequestException


def scan_with_retry(file_path: str, max_retries: int = 3, timeout: int = 30):
    """Scan with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path, f)}
                response = requests.post(
                    f"{API_URL}/api/v1/scan",
                    headers={"Authorization": f"Bearer {API_TOKEN}"},
                    files=files,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2**attempt  # Exponential backoff
            time.sleep(wait_time)
    return None
```

### Validation Example

```python
def validate_scan_result(result: dict) -> bool:
    """Validate that scan result contains expected fields."""
    required_fields = ["filename", "word_count", "chunk_count", "plagiarism_flagged"]
    return all(field in result for field in required_fields)


# Usage
result = scan_document("assignment.pdf")
if not validate_scan_result(result):
    print("Error: Invalid response from API")
    return
```

---

## LMS-Specific Integration Examples

### Canvas LMS Integration

Canvas uses the Outcomes API for external tool integration.

#### Prerequisites

1. Canvas admin access
2. API credentials from Canvas
3. Your plagiarism detector running and accessible

#### Setup Steps

1. **Register External Tool in Canvas**

```python
# Create external tool via Canvas API
import requests

CANVAS_API_URL = "https://your-institution.instructure.com"
CANVAS_TOKEN = "your-canvas-token"

external_tool_config = {
    "name": "Semantic Plagiarism Detector",
    "privacy_level": "public",
    "domain": "localhost:8000",
    "callback_url": "http://localhost:8000/api/v1/scan",
    "configuration": """
        <basic-lti-launch-request>
            <lti_version>LTI-1p0</lti_version>
            <lti_message_type>ContentItemSelectionRequest</lti_message_type>
            <resource_link_id>{resource_link_id}</resource_link_id>
        </basic-lti-launch-request>
    """,
}

headers = {
    "Authorization": f"Bearer {CANVAS_TOKEN}",
    "Content-Type": "application/json",
}

response = requests.post(
    f"{CANVAS_API_URL}/api/v1/accounts/1/external_tools",
    json={"external_tool": external_tool_config},
    headers=headers,
)
```

2. **Submit Assignment to Plagiarism API**

```python
def submit_to_plagiarism_detector(canvas_submission, api_token):
    """Download assignment from Canvas and scan for plagiarism."""
    # Download file from Canvas
    file_url = canvas_submission["attachments"][0]["url"]
    file_response = requests.get(file_url)

    # Scan with plagiarism detector
    scan_response = requests.post(
        "http://localhost:8000/api/v1/scan",
        headers={"Authorization": f"Bearer {api_token}"},
        files={
            "file": (
                canvas_submission["attachments"][0]["filename"],
                file_response.content,
            )
        },
    )

    return scan_response.json()
```

#### Canvas Integration Diagram

```
Student submits assignment
         |
         v
   Canvas receives file
         |
         v
  LMS calls plagiarism API
         |
         v
  API scans and returns result
         |
         v
  LMS stores in gradebook
```

### Moodle Integration

Moodle uses the Atto editor and submission plugins.

#### Setup Steps

1. **Configure Web Service Protocol**

Enable Web Services in Moodle:
- Site admin → Advanced features → Enable web services
- Site admin → Plugins → Web services → Enable REST protocol

2. **Create External Service**

```php
// Create Moodle external service
$service = [
    'name' => 'Plagiarism Detector Service',
    'enabled' => 1,
    'restrictedusers' => 0,
    'components' => [],
    'requiredcapability' => '',
    'shortname' => 'plagiarism_detector'
];

// Use Moodle's external API to create service
```

3. **Submit to Plagiarism API**

```php
function submit_to_plagiarism_detector($filepath, $apikey) {
    $url = 'http://localhost:8000/api/v1/scan';

    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, 1);

    $cfile = new CURLFile($filepath);
    $post = ['file' => $cfile];

    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Authorization: Bearer ' . $apikey
    ]);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $post);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

    $response = curl_exec($ch);
    curl_close($ch);

    return json_decode($response, true);
}
```

### Blackboard Integration

Blackboard uses the Building Blocks API.

#### Setup Steps

1. **Register Building Block**

```xml
<!-- build.xml for Blackboard Building Block -->
<building-block>
    <name>Plagiarism Detector</name>
    <description>Submit assignments for plagiarism checking</description>
    <version>1.0</version>
    <publisher>Your Institution</publisher>
    <contact>support@your institution.edu</contact>
    <context-path>/plagiarism-detector</context-path>
</building-block>
```

2. **Submission Integration**

```java
// Blackboard Java integration
public class PlagiarismSubmission {

    public PlagiarismResult submitToPlagiarismDetector(String filePath) {
        try {
            URL url = new URL("http://localhost:8000/api/v1/scan");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            conn.setRequestMethod("POST");
            conn.setRequestProperty("Authorization", "Bearer " + apiKey);
            conn.setDoOutput(true);

            // Upload file and get response
            // ... (file upload logic)

        } catch (Exception e) {
            // Handle error
        }
    }
}
```

---

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_BEARER_TOKEN` | Authentication token for API | `dev-bearer-token` | Yes |
| `PLAGIARISM_WEBHOOK_URL` | URL for plagiarism alert webhooks | None | No |
| `APP_BASE_URL` | Base URL for dashboard links | `http://localhost:8501` | No |

### Example .env File

```bash
# API Configuration
API_BEARER_TOKEN=super-secret-token-12345

# Webhook Configuration (Slack or Discord webhook URL)
PLAGIARISM_WEBHOOK_URL=YOUR-WEBHOOK-URL-HERE

# Application URL (for webhook links)
APP_BASE_URL=https://dashboard.your-institution.edu
```

---

## Security Best Practices

### 1. Use HTTPS in Production

Never expose the API over HTTP in production:

```bash
# Use a reverse proxy like Nginx with SSL
server {
    listen 443 ssl;
    server_name plagiarism-api.your-institution.edu;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
    }
}
```

### 2. Secure API Token

- Store tokens in environment variables, not code
- Use different tokens for dev/staging/production
- Rotate tokens regularly
- Never commit `.env` files to version control

### 3. Rate Limiting

Implement rate limiting in production:

```python
# Add rate limiting middleware
from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/scan")
@limiter.limit("10/minute")
async def scan_document(request: Request, ...):
    # ... scan logic
```

### 4. File Size Limits

Set reasonable file size limits:

```bash
# In your server configuration
upload_max_filesize = 10M
post_max_size = 10M
```

### 5. CORS Configuration

Restrict CORS origins in production:

```python
# In src/api/app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-lms.com", "https://canvas.your-institution.edu"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

---

## Testing Your Integration

### Manual Testing

1. **Test Health Endpoint**
```bash
curl http://localhost:8000/health
```

2. **Test Document Upload**
```bash
curl -X POST "http://localhost:8000/api/v1/scan" \
  -H "Authorization: Bearer your-token" \
  -F "file=@test.pdf"
```

### Automated Testing

```python
# tests/test_integration.py
import pytest
import requests


@pytest.fixture
def api_token():
    return "test-token"


def test_health_check():
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_scan_document(api_token):
    # Create test file
    test_content = b"This is a test document for plagiarism checking."

    response = requests.post(
        "http://localhost:8000/api/v1/scan",
        headers={"Authorization": f"Bearer {api_token}"},
        files={"file": ("test.txt", test_content)},
    )

    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "word_count" in data
```

---

## Troubleshooting

### Issue: "Invalid or missing authentication token"

**Solution:** Verify `API_BEARER_TOKEN` matches in both server and client.

### Issue: "Failed to extract readable text"

**Solution:** Check file format is supported (.pdf, .docx, .txt, .md, .zip). Try re-uploading the original file.

### Issue: No matches found (all clean)

**Solution:** Ensure corpus documents are indexed. Run the plagiarism detector dashboard first to build the index.

### Issue: Webhook not sending

**Solution:**
1. Verify `PLAGIARISM_WEBHOOK_URL` is set
2. Check webhook URL is accessible
3. Review server logs for request failures

### Issue: High memory usage

**Solution:**
1. Limit `top_k` parameter
2. Reduce `threshold` to scan fewer documents
3. Restart server after large batches

---

## Additional Resources

- [API Source Code](../src/api/app.py)
- [Webhook Implementation](../src/core/webhook.py)
- [README](../README.md)
- [Daily Summary Guide](./daily_summary_email_setup.md)

---

*Last updated: August 2026*
