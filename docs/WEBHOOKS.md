# Webhook Integration Security

This document describes the security measures currently implemented for the
outbound plagiarism alert webhook, and outlines planned improvements that are
not yet part of the codebase.

> **Scope note:** This webhook is **outbound only**. The application sends a
> POST request to a URL you configure (`PLAGIARISM_WEBHOOK_URL`) — there is no
> inbound webhook receiver/endpoint in this project. Any future signature
> support described below refers to *signing the outgoing request* so the
> receiving service can verify it originated from this application.

## Overview

When a plagiarism check flags a pair of documents with high similarity
(≥ 90%), the application dispatches an alert to a configured webhook URL —
typically a Slack or Discord incoming webhook. This lets administrators get
notified in near real-time without polling the dashboard.

The dispatch is asynchronous: it runs on a background thread so the
Streamlit UI is never blocked waiting on the network request.

## Configuration

| Environment Variable     | Required | Description                                                        |
|---------------------------|----------|----------------------------------------------------------------------|
| `PLAGIARISM_WEBHOOK_URL` | Yes      | The destination webhook URL (must be `https://`).                   |
| `APP_BASE_URL`           | No       | Base URL used to build the "review details" link in the alert message. Defaults to `http://localhost:8501`. |

If `PLAGIARISM_WEBHOOK_URL` is not set, webhook dispatch is silently skipped
(a warning is logged) — this is not treated as a fatal error.

## Payload Structure

The webhook sends a single JSON payload compatible with both Slack (which
reads the `text` field) and Discord (which reads `content`):

```json
{
  "text": "🚨 Plagiarism Alert! Student document *essay_a.pdf* matches *essay_b.pdf* by *93.4%*.\nReview details here: http://localhost:8501",
  "content": "🚨 Plagiarism Alert! Student document *essay_a.pdf* matches *essay_b.pdf* by *93.4%*.\nReview details here: http://localhost:8501"
}
```

Both fields always carry the same message; sending both lets the same
payload work unmodified against either platform's expected schema.

## Rate Limiting

To avoid flooding a webhook endpoint (and to reduce the blast radius of a
misconfiguration), dispatch is capped **per webhook URL**:

- Maximum **5 dispatches per rolling 60-second window**.
- Tracked in-memory per process using a timestamp deque; once the limit is
  hit, further alerts are dropped (and logged) until the window rolls
  forward.

This is an in-memory limiter, so it resets on process restart and is not
shared across multiple app instances/replicas.

## SSRF Protections

Before any webhook request is sent, the destination URL is validated by
`SSRFProtector.validate_webhook_url()` (`src/security/ssrf_protector.py`).
A URL must pass **all** of the following checks or the dispatch is blocked
and logged as a security event:

1. **HTTPS only.** Any non-`https` scheme (including plain `http`) is
   rejected outright.
2. **Hostname required.** URLs without a resolvable hostname are rejected.
3. **DNS resolution with caching.** The hostname is resolved to an IP via
   `socket.getaddrinfo`. Resolved IPs are cached for 5 minutes per hostname
   to avoid repeated lookups and reduce exposure to slow-DNS-based denial of
   service.
4. **Private IPv4 subnet blocking.** The resolved IP is checked against
   RFC1918 ranges and rejected if it falls inside any of:
   - `10.0.0.0/8`
   - `172.16.0.0/12`
   - `192.168.0.0/16`
5. **Additional address-class checks.** The resolved IP is also rejected if
   it is loopback, private (general `is_private` check, covering IPv6
   private ranges too), link-local, multicast, or unspecified (`0.0.0.0` /
   `::`).

If any check fails, an `SSRFSecurityException` is raised, the dispatch is
aborted, and the failure is logged with the offending URL and reason —
the request is never sent.

**Note:** this does not currently perform a distinct DNS-rebinding
re-check at connect time (i.e., re-resolving and re-validating immediately
before the TCP connection, after the initial validation) — validation
happens once against the cached/resolved IP.

## Error Handling

- SSRF validation failures are caught and logged; the function returns
  `False` without raising further.
- Standard network failures (timeouts, connection errors, non-2xx
  responses via `raise_for_status()`) are caught via
  `requests.exceptions.RequestException` and logged; they never propagate
  up and block the plagiarism-detection pipeline.
- The outer async dispatch wrapper (`dispatch_plagiarism_alert`) also
  catches any unexpected exception from the send routine so a webhook
  failure can never crash the background thread.

## What Is *Not* Currently Implemented

The following are **not** present in the codebase today. They are listed
here so this document doesn't imply protections that don't exist:

- No HMAC (or any other) cryptographic signature is attached to outgoing
  requests. There is no `X-Hub-Signature-256`-style header or equivalent.
- No replay-protection mechanism (timestamp header, nonce, etc.) on
  outgoing requests.
- No incoming webhook receiver/endpoint exists in this project — this
  document only covers the outbound alert dispatch described above.

## Planned Security Improvements

These are recommended future enhancements, not yet implemented:

- **HMAC request signing.** Sign the outgoing JSON payload with a shared
  secret (e.g. `HMAC-SHA256(payload_body, secret)`) and attach it as a
  custom header (e.g. `X-Plagiarism-Signature-256`), so a receiving
  service — if it's a custom endpoint rather than Slack/Discord — can
  verify the alert genuinely originated from this application and wasn't
  spoofed or tampered with in transit.
- **Timestamp-based replay protection**, paired with signing, to prevent
  a captured request from being resent later.
- **Secret rotation support** for the signing key, independent of
  `PLAGIARISM_WEBHOOK_URL`.

If you're relying on this webhook to trigger downstream actions in a
system you control (rather than just posting a Slack/Discord message),
treat the payload as **unauthenticated** until HMAC signing above is
implemented.
