import sqlite3
import json
from pathlib import Path
from contextlib import closing
from typing import List, Dict, Any, Optional

class MultimodalCorpusDB:
    """Persists image hashes and equation ASTs for cross-document matching."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.init_db()

    def init_db(self) -> None:
        """Initialize the multimodal corpus database schema."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    image_index INTEGER NOT NULL,
                    phash TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equation_asts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    equation_index INTEGER NOT NULL,
                    ast_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_image_hash(self, document_id: str, image_index: int, phash: str) -> None:
        """Saves a computed perceptual hash for an image."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO image_hashes (document_id, image_index, phash) VALUES (?, ?, ?)",
                (document_id, image_index, phash)
            )
            conn.commit()

    def save_equation_ast(self, document_id: str, equation_index: int, ast_tree: Dict[str, Any]) -> None:
        """Saves a parsed AST for an equation."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO equation_asts (document_id, equation_index, ast_json) VALUES (?, ?, ?)",
                (document_id, equation_index, json.dumps(ast_tree))
            )
            conn.commit()

    def get_all_image_hashes(self) -> List[Dict[str, Any]]:
        """Retrieves all image hashes."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM image_hashes").fetchall()
            return [dict(row) for row in rows]

    def get_all_equation_asts(self) -> List[Dict[str, Any]]:
        """Retrieves all equation ASTs."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM equation_asts").fetchall()
            result = []
            for row in rows:
                r = dict(row)
                r['ast_tree'] = json.loads(r['ast_json'])
                result.append(r)
            return result
