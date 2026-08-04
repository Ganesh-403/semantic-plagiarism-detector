# Authentication Guide

This guide explains how to authenticate with the Semantic Plagiarism Detector REST API, obtain an access token, configure two-factor authentication (2FA), and include authentication credentials in API requests.

## Authentication Overview

The REST API uses Bearer token authentication for protected endpoints. Clients must include a valid access token in the `Authorization` header when making authenticated requests.

Example:

```http
Authorization: Bearer <access_token>
```

---

## Obtaining an Access Token

Authenticate using your account credentials to receive an access token for authenticated API requests.

### Endpoint

```text
POST /api/v1/auth/login
```

### Request Body

```json
{
  "email": "user@example.com",
  "password": "password"
}
```

### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
        "email":"user@example.com",
        "password":"password"
      }'
```

### Python (`requests`) Example

```python
import requests

url = "http://localhost:8000/api/v1/auth/login"

payload = {
    "email": "user@example.com",
    "password": "password"
}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```

Store the returned access token securely and include it in subsequent API requests.

---

## Using the Access Token

Include the access token in the `Authorization` header.

### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@sample.pdf" \
  -F "threshold=0.59" \
  -F "top_k=3"
```

### Python (`requests`) Example

```python
import requests

url = "http://localhost:8000/api/v1/scan"

headers = {
    "Authorization": "Bearer <access_token>"
}

files = {
    "file": open("sample.pdf", "rb")
}

data = {
    "threshold": 0.59,
    "top_k": 3
}

response = requests.post(
    url,
    headers=headers,
    files=files,
    data=data
)

print(response.status_code)
print(response.json())
```

---

## Two-Factor Authentication (2FA)

After signing in, enroll a Time-based One-Time Password (TOTP) authenticator application to add an additional layer of account security.

Typical enrollment process:

1. Open your account security settings.
2. Start the 2FA enrollment process.
3. Scan the displayed QR code or manually enter the provided TOTP secret into your authenticator application.
4. Enter the generated one-time code to verify enrollment.
5. Save the recovery codes in a secure location.

---

## Recovery Codes

Recovery codes provide account access if your authenticator device is unavailable.

Recommendations:

- Store recovery codes securely.
- Keep them offline whenever possible.
- Use a recovery code only when you cannot generate a TOTP code.
- Replace or regenerate recovery codes if they are exposed.

> **Note:** Recovery code support depends on the authentication system configured for your deployment. If recovery codes are available, store them securely and use them only when you cannot access your TOTP authenticator.

---

## Authentication Best Practices

- Always use HTTPS in production.
- Never share or expose access tokens.
- Store tokens securely.
- Do not commit secrets or tokens to source control.
- Regenerate credentials immediately if they are compromised.