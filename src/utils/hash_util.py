import hashlib

def calculate_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate the SHA256 hash of a file efficiently by reading it in chunks.
    
    Args:
        file_path: Path to the file to hash.
        chunk_size: Size of the chunks to read (default 1MB).
        
    Returns:
        The hexadecimal SHA256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()
