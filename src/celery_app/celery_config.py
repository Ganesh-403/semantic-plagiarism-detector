import os
from celery import Celery

broker_url = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "semantic_plagiarism_detector",
    broker=broker_url,
    backend=result_backend,
    include=["src.celery_app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
)
