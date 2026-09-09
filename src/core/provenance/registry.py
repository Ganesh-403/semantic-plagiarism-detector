import sqlite3
import hashlib
from web3 import Web3


class BlockchainProvenanceRegistry:
    def __init__(self, db_path="provenance.db", rpc_url="https://polygon-rpc.com"):
        self.db_path = db_path
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS provenance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    file_hash TEXT,
                    tx_hash TEXT
                )
            ''')
            
    def compute_hash(self, file_path):
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def mint_receipt(self, document_id, file_path, mock_tx=True):
        file_hash = self.compute_hash(file_path)
        
        # In a real environment, this would sign and send a transaction via web3
        if mock_tx:
            tx_hash = "0x" + hashlib.md5(file_hash.encode()).hexdigest() * 2
        else:
            tx_hash = "0xPENDING"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO provenance (document_id, file_hash, tx_hash) VALUES (?, ?, ?)",
                (document_id, file_hash, tx_hash)
            )
            
        return tx_hash
        
    def verify_integrity(self, document_id, file_path, mock_tx=True):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT file_hash, tx_hash FROM provenance WHERE document_id = ? ORDER BY id DESC LIMIT 1",
                (document_id,)
            ).fetchone()
            
        if not row:
            return False, "No provenance record found."
            
        stored_hash, tx_hash = row
        current_hash = self.compute_hash(file_path)
        
        if stored_hash == current_hash:
            return True, f"Verified on blockchain. TX: {tx_hash}"
        else:
            return False, "Document has been altered."

