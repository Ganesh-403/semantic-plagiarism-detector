# REST API Authentication

All protected endpoints within the scanning engine require authentication via JSON Web Tokens (JWT). Follow the instructions below to acquire an access token and attach it to your API requests.

## 1. Authenticate to Get an Access Token

Send a `POST` request with your user credentials to the `/api/v1/auth/login` route. A successful request returns a Bearer access token.

### Using cURL
```bash
curl -X POST "https://openprep.ai" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "your_username",
       "password": "your_password"
     }'
```

### Using Python (`requests`)
```python
import requests

auth_url = "https://openprep.ai"
auth_payload = {
    "username": "your_username",
    "password": "your_password"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(auth_url, json=auth_payload, headers=headers)
response.raise_for_status() # Raises an exception for HTTP error statuses

token_data = response.json()
access_token = token_data.get("access_token")
print("Access token retrieved successfully.")
```

---

## 2. Access Protected Scan Endpoints

Include your access token in the `Authorization` header prefixed with `Bearer ` for any subsequent requests to secured endpoints like `/api/v1/scans/start`.

### Using cURL
```bash
curl -X POST "https://openprep.ai" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{
       "target_url": "https://example.com",
       "scan_profile": "full_audit"
     }'
```

### Using Python (`requests`)
```python
import requests

scan_url = "https://openprep.ai"
scan_headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}
scan_payload = {
    "target_url": "https://example.com",
    "scan_profile": "full_audit"
}

response = requests.post(scan_url, json=scan_payload, headers=scan_headers)

if response.status_code == 200:
    print("Scan initiated successfully:")
    print(response.json())
else:
    print(f"Failed to initiate scan. Status code: {response.status_code}")
    print(response.text)
```

---

## Testing with Swagger UI

When developing and auditing endpoints locally, you can use the interactive Swagger UI panel hosted at [http://localhost:8000/docs](http://localhost:8000/docs) to fire live requests against your workspace.

Secured endpoints require a valid JSON Web Token (JWT) Bearer token to authorize access. Follow these steps to authenticate your browser session:

### 🔐 How to Authorize Your Session

1. **Open the Documentation Core**: Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
2. **Locate the Security Action Hook**: Click the lock icon button labeled **"Authorize"** positioned at the top right header section of the page.
3. **Inject the Authorization Token**: 
   * In the modal popup window, locate the text input field labeled **Value**.
   * Enter your token using the exact format: `Bearer <your_jwt_token_here>`
   * *Example*: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
4. **Lock the Session Configuration**: Click the **Authorize** button within the modal window, then click **Close**.

Now, all subsequent interactive endpoint requests dispatched via the UI will automatically append the correct tracking header (`Authorization: Bearer <token>`) to your API request parameters.

### 🧪 Triggering an Interactive Request

* Expand any locked API route container (indicated by a closed lock icon).
* Click the **"Try it out"** button in the top right of the route container.
* Populate any required query parameters or JSON body payloads.
* Press the blue **"Execute"** button to fire the network request and review the server's response.
