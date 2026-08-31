"""
src/security/bloom_filter_exchange.py
-------------------------------------
Cryptographic packaging for LSH band exchange.

Packages LSH bands and cryptographically signs them for secure peer-to-peer
exchange between institutional nodes in a federated plagiarism detection network.
"""

import hashlib
import hmac
import json
import logging
from typing import List, Dict, Any, Optional
import base64

logger = logging.getLogger(__name__)


def package_lsh_bands(
    document_id: str,
    lsh_bands: list[bytes],
    institution_id: str,
    secret_key: str
) -> dict[str, Any]:
    """Package and cryptographically sign LSH bands for exchange.
    
    Args:
        document_id: Unique identifier for the document.
        lsh_bands: List of LSH band byte strings.
        institution_id: ID of the institution generating the signature.
        secret_key: Shared secret key for HMAC signing.
        
    Returns:
        Dictionary containing the payload and HMAC signature.
    """
    # Base64 encode the bands for JSON serialization
    encoded_bands = [base64.b64encode(band).decode('utf-8') for band in lsh_bands]
    
    payload = {
        "document_id": document_id,
        "institution_id": institution_id,
        "lsh_bands": encoded_bands
    }
    
    # Serialize payload to canonical JSON string for signing
    payload_str = json.dumps(payload, sort_keys=True)
    
    # Generate HMAC-SHA256 signature
    signature = hmac.new(
        secret_key.encode('utf-8'),
        payload_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return {
        "payload": payload,
        "signature": signature
    }


def verify_lsh_package(
    package: dict[str, Any],
    secret_key: str
) -> bool:
    """Verify the cryptographic signature of an LSH package.
    
    Args:
        package: The package dictionary containing 'payload' and 'signature'.
        secret_key: The shared secret key used for verification.
        
    Returns:
        True if the signature is valid, False otherwise.
    """
    if "payload" not in package or "signature" not in package:
        return False
        
    payload_str = json.dumps(package["payload"], sort_keys=True)
    expected_sig = hmac.new(
        secret_key.encode('utf-8'),
        payload_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_sig, package["signature"])
