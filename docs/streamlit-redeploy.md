# Redeploy Streamlit Community Cloud

Use these steps after the follow-up changes have been merged into `main`.
The local smoke checks do not update the hosted site.

## Preserve the current deployment

In [Community Cloud](https://share.streamlit.io/), record the app's URL/subdomain,
repository, branch, main-file path and secrets. Download database backups and
exports before deleting an app or changing its state directory. Community Cloud
local files are not durable storage; use external backups for anything that must
survive a rebuild.

## Deployment settings

- Repository: `Ganesh-403/semantic-plagiarism-detector`
- Branch: `main`
- Main file: `app/streamlit_app.py`
- Python: `3.13` matches the local verification environment. The CI matrix also
  covers `3.11` and `3.12`; confirm those checks before choosing either version.
- Keep `requirements.txt`, `requirements-no-torch.txt` and `packages.txt` at the
  repository root. Do not deploy `requirements-no-torch.txt` alone: the app needs
  the embedding-model dependencies in `requirements.txt`.

Community Cloud documents these fields and advanced settings in its
[deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Secrets

Add root-level TOML secrets in the app's Advanced settings or Settings → Secrets:

```toml
APP_ENV = "production"
ADMIN_BOOTSTRAP_PASSWORD = "replace-with-a-unique-strong-admin-password"  # pragma: allowlist secret
JWT_SECRET_KEY = "replace-with-a-random-secret-of-at-least-32-characters"  # pragma: allowlist secret
SEMANTIC_PLAGIARISM_MODEL = "all-MiniLM-L6-v2"
PRELOAD_EMBEDDING_MODEL = "false"
ENABLE_EMBEDDED_API = "false"
CROSS_ENCODER_RERANKING_ENABLED = "false"
```

Replace both secret placeholders before saving. Generate the JWT secret with
`python -c "import secrets; print(secrets.token_urlsafe(48))"` on your own machine.
Preserve an existing JWT secret during a routine redeploy to avoid invalidating
active sessions. Root-level Streamlit secrets are exposed as environment variables.

`ADMIN_BOOTSTRAP_PASSWORD` creates `admin` only when that account is absent. It does
not reset an existing account. After setting up durable state, remove the bootstrap
password. If the deployment intentionally starts with empty ephemeral state after
each rebuild, it needs a controlled bootstrap process again.

Redis is optional. Leave `REDIS_HOST`, `REDIS_URL` and `REDIS_PASSWORD` unset when no
Redis service is provisioned. For a managed service, use its authenticated TLS
`REDIS_URL`; do not invent a Redis host or password just to pass startup checks.

The smaller model reduces memory use for English comparisons. For multilingual
embedding without translation, choose a compatible multilingual model deliberately
and rebuild existing corpus embeddings/indexes after changing models.

## Password recovery and account maintenance

For email recovery, accounts must use their email address as their username.
Configure a real SMTP service that supports STARTTLS on port 587. Add these
root-level secrets using that service's actual values:

```toml
SMTP_SERVER = "your-smtp-server"
SMTP_PORT = "587"
SMTP_USERNAME = "your-smtp-account"
SMTP_PASSWORD = "your-smtp-password"  # pragma: allowlist secret
SMTP_FROM = "your-verified-sender-address"
```

The login page's **Password recovery** form sends a single-use token that expires
in 15 minutes. Without SMTP settings, the page directs users to an administrator.
An administrator can reset another account to a temporary password in user
management; the user must change that password before entering the dashboard.
Changing a password invalidates earlier tokens and signs the current browser out.

Startup migrates the authentication database through schema version 20. Preserve
the existing users database along with the corpus, indexes and configuration.
If the standalone API uses LTI, preserve its private signing key in the configured
data directory too. Restoring just the corpus database does not restore accounts.

## Apply the update

A push to the deployed branch normally updates the app; dependency-file changes
trigger dependency installation. Open the app's **Manage app** panel and inspect
its logs. If it remains stale, use **Reboot app** from the app menu, following the
[reboot instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app).

Changing the Python version requires deleting and redeploying the app. First save
its configuration and data, then recreate it with the settings above. This is
Streamlit's documented [Python-upgrade process](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).
Do not delete the app simply to apply an ordinary source change.

## Check the result

1. Confirm dependency installation finishes and the login page loads.
2. Log in with an existing account or the newly bootstrapped `admin` account.
3. Compare two small text files containing similar paragraphs. The first analysis
   may download model weights; watch the logs until that completes.
4. Confirm similarity results render and rerunning does not duplicate the documents.
5. Confirm the administrator's system-health tab renders. With no Redis service,
   a cache-unavailable status is expected and the app should remain usable.

If it fails, retain the first traceback and dependency-installation error from the
logs. The [recovery validation report](recovery-validation.md) distinguishes local
verification from checks that still need the hosted environment.
