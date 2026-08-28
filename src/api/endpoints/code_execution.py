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
src/api/endpoints/code_execution.py
-----------------------------------
FastAPI router for Dynamic Code Execution Sandboxing.

Provides REST endpoints to submit code for sandboxed behavioral analysis
and retrieve execution traces.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.execution_trace_analyzer import generate_behavioral_hash
from src.db.sandbox_logs_db import (
    find_behavioral_clones,
    initialize_sandbox_db,
    log_execution_trace,
)
from src.security.code_sandbox import execute_code_sandbox

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/code-execution", tags=["Code Execution"])

initialize_sandbox_db()


class ExecutionRequest(BaseModel):
    """Schema for code execution requests."""

    code: str = Field(..., min_length=1)
    test_cases: Optional[list[dict[str, Any]]] = None
    timeout: float = Field(5.0, ge=0.1, le=30.0)
    max_memory_mb: int = Field(256, ge=64, le=1024)


class ExecutionResponse(BaseModel):
    """Schema for code execution responses."""

    submission_hash: str
    behavioral_hash: str
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    memory_limit_exceeded: bool
    test_results: list[dict[str, Any]]
    behavioral_clones: list[str]


@router.post("/execute", response_model=ExecutionResponse)
async def execute_code(request: ExecutionRequest):
    """Submit code for sandboxed execution and behavioral analysis."""
    # Generate a hash of the submitted code for tracking
    submission_hash = hashlib.sha256(request.code.encode("utf-8")).hexdigest()

    # Execute in sandbox
    trace = execute_code_sandbox(
        code=request.code,
        test_cases=request.test_cases,
        timeout=request.timeout,
        max_memory_mb=request.max_memory_mb,
    )

    # Generate behavioral hash
    behavioral_hash = generate_behavioral_hash(
        trace["stdout"], trace["stderr"], trace["test_results"], trace["return_code"]
    )

    # Log the trace
    log_execution_trace(submission_hash, behavioral_hash, trace)

    # Find behavioral clones in the database
    clones = find_behavioral_clones(
        behavioral_hash, exclude_submission_hash=submission_hash
    )

    return ExecutionResponse(
        submission_hash=submission_hash,
        behavioral_hash=behavioral_hash,
        stdout=trace["stdout"],
        stderr=trace["stderr"],
        return_code=trace["return_code"],
        execution_time=trace["execution_time"],
        memory_limit_exceeded=trace["memory_limit_exceeded"],
        test_results=trace["test_results"],
        behavioral_clones=clones,
    )
