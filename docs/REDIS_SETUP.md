# Redis Setup Guide

## Overview

The Semantic Plagiarism Detection System supports Redis as an **optional** service for session caching and rate limiting. If Redis is unavailable, the application automatically falls back to local in-memory state, allowing contributors to run the application without Redis during development.

This guide explains how to run Redis locally using Docker, configure the required environment variables, enable authentication when needed, verify the connection, and troubleshoot common issues.

---

# Prerequisites

Before setting up Redis, ensure the following are installed:

- Docker Desktop
- Docker CLI

Verify Docker is installed:

```bash
docker --version
```

---

# Running Redis Locally with Docker

Pull the official Redis image:

```bash
docker pull redis
```

Start a Redis container:

```bash
docker run -d --name redis -p 6379:6379 redis
```

### Command Explanation

| Option | Description |
|---------|-------------|
| `-d` | Run the container in detached mode |
| `--name redis` | Name the container `redis` |
| `-p 6379:6379` | Expose Redis on the default port |

Verify that Redis is running:

```bash
docker ps
```

You should see a container named **redis** with port **6379** exposed.

---

# Running with Docker Compose

The project also provides Docker Compose support. Running the following command starts the Streamlit application together with the optional Redis service.

```bash
docker compose up --build
```

Stop all services:

```bash
docker compose down
```

Remove associated volumes:

```bash
docker compose down -v
```

---

# Redis Configuration

Redis configuration is provided through environment variables in `.env`.

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Complete Redis connection URL. If set, it overrides the individual Redis settings. |
| `REDIS_HOST` | Redis server hostname. |
| `REDIS_PORT` | Redis server port. |
| `REDIS_DB` | Redis database number. |
| `REDIS_PASSWORD` | Password used when Redis authentication is enabled. |

Example configuration:

```env
REDIS_URL=
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### Connection Priority

The application resolves Redis configuration in the following order:

1. `REDIS_URL`
2. `REDIS_HOST`
3. `REDIS_PORT`
4. `REDIS_DB`
5. `REDIS_PASSWORD`

If `REDIS_URL` is provided, the remaining Redis settings are ignored.

---

# Authentication

Redis does not require authentication by default.

To start Redis with password protection:

```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis redis-server --requirepass yourpassword
```

Configure the same password in your `.env` file:

```env
REDIS_PASSWORD=yourpassword
```

Connect using Redis CLI:

```bash
redis-cli -a yourpassword
```

---

# Verifying the Connection

Open the Redis CLI:

```bash
redis-cli
```

If authentication is enabled:

```bash
redis-cli -a yourpassword
```

Test the connection:

```text
PING
```

Expected response:

```text
PONG
```

---

# Useful Docker Commands

Start Redis:

```bash
docker start redis
```

Stop Redis:

```bash
docker stop redis
```

Restart Redis:

```bash
docker restart redis
```

View logs:

```bash
docker logs redis
```

Remove the container:

```bash
docker rm redis
```

---

# Troubleshooting

## Redis container is not running

Check running containers:

```bash
docker ps
```

If the container is stopped:

```bash
docker start redis
```

---

## Unable to connect to Redis

Verify:

- Docker Desktop is running.
- Redis container is running.
- `REDIS_HOST` points to the correct host.
- `REDIS_PORT` matches the exposed port.
- `REDIS_URL` is correctly formatted if used.

---

## Authentication failed

Error:

```text
NOAUTH Authentication required.
```

Solution:

- Ensure `REDIS_PASSWORD` matches the password configured when Redis was started.
- Connect using:

```bash
redis-cli -a yourpassword
```

---

## Port 6379 already in use

Start Redis on another port:

```bash
docker run -d --name redis -p 6380:6379 redis
```

Update your environment configuration:

```env
REDIS_PORT=6380
```

---

## Check Redis logs

If Redis does not start correctly:

```bash
docker logs redis
```

Review the logs for startup or configuration errors.

---

# Connection Debugging Checklist

Before reporting a Redis issue, verify the following:

- Docker Desktop is running.
- Redis container is running (`docker ps`).
- Environment variables are configured correctly.
- `REDIS_URL` is valid or `REDIS_HOST` and `REDIS_PORT` are correctly configured.
- `REDIS_PASSWORD` matches the Redis server configuration if authentication is enabled.
- `redis-cli` connects successfully.
- `PING` returns `PONG`.

---

# Summary

Redis is an optional service for this project that enhances session caching and rate limiting. Contributors can run Redis locally with Docker or Docker Compose, configure the required environment variables through `.env`, and use the troubleshooting steps above to diagnose connection issues. If Redis is unavailable, the application continues to operate using its built-in in-memory fallback.
