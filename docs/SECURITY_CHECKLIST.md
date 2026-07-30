# Production Security Checklist

## 1. HTTPS / Transport Security
(cover: terminating TLS at nginx, redirecting HTTP→HTTPS, valid cert renewal,
HSTS header, disabling weak TLS versions/ciphers)

## 2. Secret Management
(cover: never committing .env, rotating API_BEARER_TOKEN and SMTP_PASSWORD,
using a secrets manager instead of plain .env in production, restricting
REDIS_PASSWORD access, least-privilege service accounts)

## 3. HTTP Security Headers
(cover: adding Content-Security-Policy, X-Frame-Options, X-Content-Type-Options,
Referrer-Policy, Strict-Transport-Security in nginx.conf)

## 4. Network & Infrastructure Hardening
(cover: not exposing Redis port publicly, restricting nginx to necessary ports,
running containers as non-root, keeping base images patched)

## 5. Application-Level Hardening
(cover: enforcing 2FA for admin accounts, session expiry settings,
SSRF/MIME/file-upload protections already in src/security/, rate limiting)

## 6. Monitoring & Logging
(cover: centralizing logs, alerting on repeated auth failures, audit log retention)