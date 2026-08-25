import io
from typing import List

try:
    import imagehash
    from PIL import Image
except ImportError:
    pass

class ImagePHashEngine:
    """Extracts images and computes perceptual hashes for rotation-invariant matching."""

    @staticmethod
    def extract_images_from_pdf(pdf_path: str) -> List['Image.Image']:
        """
        Extracts images from a PDF. 
        Note: Currently a stub for actual PDF extraction logic, typically done with fitz/PyMuPDF.
        """
        return []

    @staticmethod
    def compute_phash(image: 'Image.Image') -> str:
        """Computes a perceptual hash for an image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        hash_val = imagehash.phash(image)
        return str(hash_val)

    @staticmethod
    def compute_rotation_invariant_phash(image: 'Image.Image') -> str:
        """
        Computes a perceptual hash that attempts to be rotation invariant.
        """
        rotations = [
            image,
            image.rotate(90, expand=True),
            image.rotate(180, expand=True),
            image.rotate(270, expand=True)
        ]
        hashes = [str(imagehash.phash(rot.convert('RGB'))) for rot in rotations]
        return min(hashes)

    @staticmethod
    def phash_distance(hash1: str, hash2: str) -> int:
        """Computes Hamming distance between two perceptual hashes."""
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
