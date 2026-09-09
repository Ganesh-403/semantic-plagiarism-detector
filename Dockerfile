# Use the supported Python 3.11 Debian runtime, including patch security updates.
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies (these will NOT be in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY requirements*.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab stopwords

# Final stage - runtime image (NO build tools)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install ONLY runtime dependencies (no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -m -g appuser appuser && \
    mkdir -p /state && chown -R appuser:appuser /app /state

EXPOSE 8501
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
