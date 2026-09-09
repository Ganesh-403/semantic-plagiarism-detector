"""Measure retained allocations across repeated real upload/cleanup operations."""
import asyncio
import gc
import io
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import UploadFile
from src.utils.file_streaming import stream_upload_file_to_disk

MEMORY_DELTA_LIMIT_MB = 10.0

async def exercise_uploads(count):
    for _ in range(count):
        upload = UploadFile(io.BytesIO(b"sample text " * 100_000), filename="sample.txt")
        path = await stream_upload_file_to_disk(upload)
        try:
            assert Path(path).stat().st_size == 1_200_000
        finally:
            Path(path).unlink()
            await upload.close()


def main():
    asyncio.run(exercise_uploads(5))
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    asyncio.run(exercise_uploads(100))
    gc.collect()
    after = tracemalloc.take_snapshot()
    delta = sum(s.size_diff for s in after.compare_to(before, "lineno")) / 1024**2
    tracemalloc.stop()
    print(f"Retained upload allocations: {delta:.2f} MiB (limit {MEMORY_DELTA_LIMIT_MB})")
    return int(delta > MEMORY_DELTA_LIMIT_MB)

if __name__ == "__main__":
    sys.exit(main())
