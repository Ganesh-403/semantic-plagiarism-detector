# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
src/api/endpoints/federation.py
-------------------------------
FastAPI router for Federated Plagiarism Detection.

Handles signature uploads, downloads, and federated similarity queries
using MinHash and LSH bands.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.core.federated_minhash import (
    estimate_jaccard_similarity,
    generate_lsh_bands,
    generate_minhash_signature,
)
from src.db.federation_registry_db import (
    initialize_federation_db,
    register_trusted_node,
    store_federated_signature,
)
from src.security.bloom_filter_exchange import package_lsh_bands, verify_lsh_package

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/federation", tags=["Federation"])

initialize_federation_db()


class MinHashRequest(BaseModel):
    """Schema for generating MinHash signatures."""

    text: str = Field(..., min_length=10)
    num_hashes: int = Field(128, ge=32, le=512)
    shingle_size: int = Field(3, ge=2, le=5)
    bands: int = Field(16, ge=4, le=64)
    rows_per_band: int = Field(8, ge=2, le=16)


class UploadRequest(BaseModel):
    """Schema for uploading signed LSH packages."""

    document_id: str
    institution_id: str
    lsh_bands: list[str]


@router.post("/minhash")
async def generate_signature(request: MinHashRequest):
    """Generate MinHash signature and LSH bands for a text document."""
    sig = generate_minhash_signature(
        request.text, request.num_hashes, request.shingle_size
    )
    bands = generate_lsh_bands(sig, request.bands, request.rows_per_band)

    # Convert bands to hex strings for JSON response
    hex_bands = [b.hex() for b in bands]

    return {
        "document_length": len(request.text),
        "signature_length": request.num_hashes,
        "lsh_bands": hex_bands,
    }


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_signature(request: UploadRequest):
    """Upload and store LSH bands from a trusted institutional node."""
    success = store_federated_signature(
        request.document_id, request.institution_id, request.lsh_bands
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to store signature.")

    return {"status": "success", "document_id": request.document_id}
