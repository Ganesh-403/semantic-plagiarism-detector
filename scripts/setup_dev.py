import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version() -> bool:
    if sys.version_info < (3, 10):
        print(f"[FAIL] Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.10+ is required.")
        return False
    return True


def create_directories() -> bool:
    try:
        for d in ["data", "logs"]:
            (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"[FAIL] Failed to create directories: {e}")
        return False


def setup_env() -> bool:
    env_file = PROJECT_ROOT / ".env"
    example_file = PROJECT_ROOT / ".env.example"
    
    if env_file.exists():
        return True
    
    if not example_file.exists():
        print("[WARN] .env.example is missing. Cannot create .env.")
        return False
        
    try:
        shutil.copy(example_file, env_file)
        return True
    except Exception as e:
        print(f"[FAIL] Failed to copy .env.example to .env: {e}")
        return False


def install_dependencies() -> bool:
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print("[FAIL] requirements.txt not found.")
        return False
        
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("[FAIL] Failed to install dependencies.")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"[FAIL] Exception during dependency installation: {e}")
        return False


def setup_nltk() -> bool:
    try:
        import nltk
        corpora = [
            "punkt",
            "punkt_tab",
            "averaged_perceptron_tagger",
            "maxent_ne_chunker",
            "words",
            "stopwords",
        ]
        for corpus in corpora:
            success = nltk.download(corpus, quiet=True)
            if not success:
                print(f"[FAIL] Failed to download NLTK corpus: {corpus}")
                return False
        return True
    except Exception as e:
        print(f"[FAIL] Exception during NLTK download: {e}")
        return False


def initialize_databases() -> bool:
    try:
        from src.db.auth import init_db as init_auth_db
        from src.db.corpus_db import init_corpus_db
        from src.db.incidents import init_incident_db

        init_auth_db()
        init_corpus_db()
        init_incident_db()
        return True
    except Exception as e:
        print(f"[FAIL] Exception during DB initialization: {e}")
        return False


def check_tesseract() -> bool:
    try:
        tess = shutil.which("tesseract")
        if not tess:
            print("[WARN] Tesseract not found.")
            return False
        result = subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 and result.returncode != 1:
            print(f"[WARN] Tesseract found but failed to run: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"[WARN] Error checking Tesseract: {e}")
        return False


def print_readiness_report(results: dict) -> None:
    print("\n========================================")
    print(" Development Environment Readiness")
    print("========================================")
    print()
    
    python_ok = results.get('python', False)
    deps_ok = results.get('deps', False)
    nltk_ok = results.get('nltk', False)
    dirs_ok = results.get('dirs', False)
    env_ok = results.get('env', False)
    db_ok = results.get('db', False)
    tess_ok = results.get('tesseract', False)

    print(f"[{'OK' if python_ok else 'FAIL'}] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"[{'OK' if deps_ok else 'FAIL'}] Dependencies installed")
    print(f"[{'OK' if nltk_ok else 'FAIL'}] NLTK corpora available")
    print(f"[{'OK' if dirs_ok else 'FAIL'}] data/ and logs/ directories")
    print(f"[{'OK' if env_ok else 'WARN'}] .env configured")
    print(f"[{'OK' if db_ok else 'FAIL'}] SQLite databases initialized")
    print(f"[{'OK' if tess_ok else 'WARN'}] Tesseract detected")
    print()
    print("-" * 40)
    
    is_ready = all([python_ok, deps_ok, nltk_ok, dirs_ok, db_ok])
    if is_ready:
        print("Environment is ready for development.")
    else:
        print("Environment is NOT ready. Please fix the failures above.")
    print("-" * 40)


def main():
    results = {}
    
    results['python'] = check_python_version()
    if not results['python']:
        sys.exit(1)
        
    results['dirs'] = create_directories()
    results['env'] = setup_env()
    
    results['deps'] = install_dependencies()
    if not results['deps']:
        print_readiness_report(results)
        sys.exit(1)
        
    results['nltk'] = setup_nltk()
    results['db'] = initialize_databases()
    results['tesseract'] = check_tesseract()
    
    print_readiness_report(results)
    
    if not all([results['python'], results['deps'], results['nltk'], results['dirs'], results['db']]):
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
