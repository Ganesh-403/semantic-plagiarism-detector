import argparse
import tracemalloc
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def benchmark_memory(pdf_path: str):
    tracemalloc.start()
    try:
        from src.core.document_parser import extract_text_from_pdf
        extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"Error parsing PDF: {e}", file=sys.stderr)
        return
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        print(f"Peak Memory: {peak_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark memory footprint of PDF parsing")
    parser.add_argument("pdf_path", help="Path to the PDF file to benchmark")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"File not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    benchmark_memory(args.pdf_path)
