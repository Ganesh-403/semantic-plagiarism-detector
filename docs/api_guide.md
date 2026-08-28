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
