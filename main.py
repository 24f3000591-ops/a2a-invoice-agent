import os
import json
import hashlib
import asyncio
import re
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Response, HTTPException, Header, status
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="A2A Invoice Agent")

# Environment & Configuration
BASE_URL = os.getenv("BASE_URL", "https://your-app.onrender.com/a2a").rstrip("/")
ORIGIN = "/".join(BASE_URL.split("/")[:3]) if "://" in BASE_URL else BASE_URL
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# In-Memory Storage
# Tasks Store: { (principal, task_id): TaskDict }
tasks_db: Dict[tuple, Dict[str, Any]] = {}

# Idempotency Store: { (principal, message_id): {"hash": str, "task_id": str} }
idempotency_db: Dict[tuple, Dict[str, Any]] = {}

# Package Decision Cache: { canonical_hash: dict_decision }
package_cache: Dict[str, Dict[str, Any]] = {}

# Fine-Grained Locks for Task Race Conditions: { task_id: asyncio.Lock }
task_locks: Dict[str, asyncio.Lock] = {}
global_lock = asyncio.Lock()


async def get_task_lock(task_id: str) -> asyncio.Lock:
    async with global_lock:
        if task_id not in task_locks:
            task_locks[task_id] = asyncio.Lock()
        return task_locks[task_id]


def canonical_json_bytes(obj: Any) -> bytes:
    """Produces compact key-sorted JSON bytes for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')


def hash_message(message_obj: Dict[str, Any]) -> str:
    """Hashes recursively key-sorted compact JSON of the message only."""
    return hashlib.sha256(canonical_json_bytes(message_obj)).hexdigest()


def canonical_package_hash(pkg: Dict[str, Any]) -> str:
    """Hashes canonical content of an invoice package for decision caching."""
    return hashlib.sha256(canonical_json_bytes(pkg)).hexdigest()


def verify_headers(a2a_version: Optional[str], content_type: Optional[str] = None, check_content_type: bool = False):
    """Strict protocol header checks for A2A 1.0 HTTP+JSON binding."""
    if a2a_version != "1.0":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header 'A2A-Version: 1.0' is required."
        )
    if check_content_type:
        if not content_type or "application/a2a+json" not in content_type.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content-Type must be 'application/a2a+json'."
            )


def get_principal(authorization: Optional[str]) -> str:
    """Extracts and verifies Bearer token for multi-tenant isolation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authorization token."
        )
    token = authorization.split("Bearer ")[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token cannot be empty."
        )
    return token


# LLM Reasoning Layer with Evidence Extraction
async def analyze_package_with_llm(pkg: Dict[str, Any]) -> Dict[str, Any]:
    pkg_hash = canonical_package_hash(pkg)
    if pkg_hash in package_cache:
        return package_cache[pkg_hash]

    prompt = f"""You are an expert invoice reconciliation auditor. Analyze this invoice package carefully and determine the exact business action required.

Actions:
- "settle_invoice": Valid, reconciled, and within autonomous authority limit.
- "request_approval": Commercially valid, but exceeds delegated authority limit.
- "hold_invoice": Payment pauses until a stated verification completes.
- "reject_duplicate": The same commercial invoice was already paid.
- "open_exception": Material records conflict and need an exception workflow.

Invoice Package Data:
{json.dumps(pkg, indent=2)}

Rules:
1. Identify vendorName, invoiceNumber, amountMinor (int), currency.
2. Find the controlling paragraph determining the action and extract EXACTLY THREE decisive bracketed evidence references (e.g. ["[REF-123]", "[REF-456]", "[REF-789]"]). Ignore cover sheet, archive examples, or training decoys.
3. Write a rationale between 60 and 1500 characters. You MUST explicitly state the action name and cite at least two evidence references.

Return raw JSON matching this schema:
{{
  "action": "<one_action_above>",
  "facts": {{
    "vendorName": "...",
    "invoiceNumber": "...",
    "amountMinor": 12345,
    "currency": "INR"
  }},
  "evidenceRefs": ["[...]", "[...]", "[...]"],
  "rationale": "..."
}}
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPIPE_TOKEN}"
    }
    
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise finance auditor AI. Return raw JSON without markdown formatting."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(
                f"{AIPIPE_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                json=body
            )
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            
            # Clean Markdown code block fences safely
            cleaned = re.sub(r"^```(?:json)?\s*|\s*
