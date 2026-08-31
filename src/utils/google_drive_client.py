import logging
import random
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class GoogleDriveResilientClient:
    """Wrapper encapsulating exponential backoff with jitter for Google Drive operations."""

    @staticmethod
    def execute_with_backoff(
        api_callable: Callable[..., Any],
        *args: Any,
        max_retries: int = 5,
        base_delay_seconds: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """
        Executes a Google Drive API function hook with exponential backoff and jitter.
        Retries occur specifically upon receiving HTTP 429 (Too Many Requests)
        and HTTP 503 (Service Unavailable) error structures.
        """
        retries = 0
        while True:
            try:
                # Dispatch the real API call execution path
                return api_callable(*args, **kwargs)

            except Exception as exc:
                # Dynamically inspect standard Google API client exception status codes
                status_code = getattr(exc, "resp", None) and getattr(
                    exc.resp, "status", None
                )

                # Fallback check mapping alternative library structure signatures
                if not status_code:
                    status_code = getattr(exc, "status_code", None)

                # Evaluate if error matches rate limits or temporary cluster dropouts
                if status_code in (429, 503) and retries < max_retries:
                    retries += 1

                    # Core Math: Exponential backoff progression = base_delay * (2 ^ retry_count)
                    backoff_delay = base_delay_seconds * (2**retries)

                    # Decoupling Jitter: Apply uniform random variance between 0 and 1000ms
                    # This prevents synchronized cluster retry spikes (thundering herd problem)
                    jitter = random.uniform(0.0, 1.0)
                    total_sleep_time = backoff_delay + jitter

                    logger.warning(
                        f"Google Drive API hit transient status {status_code}. "
                        f"Initiating retry retry {retries}/{max_retries} "
                        f"in {total_sleep_time:.2f} seconds..."
                    )

                    time.sleep(total_sleep_time)
                else:
                    # Cascade exceptions upwards immediately if non-retryable or bounds are blown
                    raise exc
