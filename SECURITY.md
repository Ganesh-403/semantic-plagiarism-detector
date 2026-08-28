# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main | ✅ |
| latest release | ✅ |
| 1.0.x | ✅ |
| < 1.0 | ❌ |
| older releases | ❌ |

## Reporting a Vulnerability

We take the security of the **Semantic Plagiarism Detection System** seriously. If you believe you have discovered a security vulnerability in this project, please follow the procedure below to report it to us.

### How to Report

**Please do NOT open a public GitHub issue for security vulnerabilities.** We request that you do not disclose the vulnerability publicly before a fix is released.

Instead, please report vulnerabilities privately using **GitHub Private Vulnerability Reporting**.

To report a vulnerability:

1. Navigate to the **Security** tab of this repository.
2. Click on **Advisories** in the left sidebar (under Vulnerability reporting).
3. Click on the **Report a vulnerability** button.
4. Provide a detailed description of the vulnerability and its potential impact.
5. Include step-by-step instructions to reproduce the issue (including sample input files or payload snippets).
6. Provide any proposed mitigations or code patches if you have them.

The maintainers will investigate, validate, and coordinate disclosure.

### Response Timeline

- **Expected first response time:** **within 48 hours**.
- **Vulnerability Assessment:** Within 5 business days.
- **Patch & Advisory Release:** Depends on severity, typically within 14 days.

Thank you for helping keep our application and users safe!

## File Upload Hardening & Validation

To help protect the application from malicious or unsafe file uploads, all uploaded files should be validated and sanitized before processing or storage.

### File Upload Sanitization

- Sanitize filenames to remove unsafe or unexpected characters.
- Generate server-side filenames instead of relying on user-provided names.
- Validate the file's MIME type in addition to its extension.
- Scan uploaded files for malicious content whenever practical.
- Store uploaded files outside the web root whenever possible.

### File Size Bounds

- Enforce a maximum file size limit for all uploads.
- Reject files that exceed the configured size limit.
- Apply stricter size limits for specific file types when appropriate to reduce resource usage.

### Extension Validation

- Use an allowlist of permitted file extensions.
- Reject executable or potentially dangerous file types such as `.exe`, `.bat`, `.cmd`, `.sh`, `.php`, and `.js` unless explicitly required.
- Do not rely solely on file extensions; verify that the file content matches the expected format.
- Normalize filenames before validation to prevent bypass techniques.

### Additional Recommendations

- Validate uploaded files before processing or storage.
- Log failed upload validation attempts for monitoring and auditing.
- Restrict upload functionality to authorized users where applicable.
- Keep file validation libraries and dependencies up to date.

## Redis Security & Access Control

To protect cache data, session states, and FAISS indices from unauthorized access, configure your Redis production instances using the following security best practices.

### 1. Transport Layer Security (TLS) Encryption

- **Encrypt Traffic in Transit:** Enable TLS encryption (`rediss://` protocol) for all connections between the application server and the Redis host to prevent packet sniffing.
- **Client Certificate Verification:** Configure Redis to require client certificates (`tls-auth-clients yes`) to ensure only authorized application nodes can establish connections.

### 2. Password Protection & Authentication

- **Require Strong Passwords:** Set a complex, high-entropy password in `redis.conf` using the `requirepass` directive.
- **Environment Variables:** Inject the Redis password into the application container using secure secrets (e.g., environment variables) rather than hardcoding credentials in config files.

### 3. Access Control Lists (ACLs)

- **Least Privilege Access:** Utilize Redis ACLs (available in Redis 6.0+) to define strict permissions instead of using a global administrator user.
- **Restricted Users:** Create a dedicated user for the plagiarism detector that is only allowed access to the specific keyspaces it uses:

  ```redis
  user spd_app on >StrongPassword ~spd:v1:* +@all -@dangerous

```

* **Disable Unused Commands:** Block high-risk commands such as `FLUSHALL`, `FLUSHDB`, `KEYS`, `CONFIG`, and `SHUTDOWN` for the application user.

## Security Contact

For security vulnerabilities (such as SSRF, SQLi, or RCE), please report them using the repository's **Security > Advisories > Report a vulnerability** feature, or contact project maintainers privately. Please do not disclose security vulnerabilities through public GitHub issues.

```

```
