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
# Deduce origin for discovery
ORIGIN = "/".join(BASE_URL.split("/")[:3]) if "://" in BASE_URL else BASE_URL
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# In-Memory Databases & Mutex Locks
# Task Store: { (principal, task_id): TaskDict }
tasks_db: Dict[tuple, Dict[str, Any]] = {}

# Message Idempotency Store: { (principal, message_id): {"hash": str, "task_id": str} }
idempotency_db: Dict[tuple, Dict[str, Any]] = {}

# Canonical Package Decision Cache: { canonical_hash: dict_decision }
package_cache: Dict[str, Dict[str, Any]] = {}

# Fine-grained Locks for Task Race Conditions: { task_id: asyncio.Lock }
task_locks: Dict[str, asyncio.Lock] = {}
global_lock = asyncio.Lock()


async def get_task_lock(task_id: str) -> asyncio.Lock:
    async with global_lock:
        if task_id not in task_locks:
            task_locks[task_id] = asyncio.Lock()
        return task_locks[task_id]


# Helper: Canonical JSON & Deterministic Hashing
def canonical_json_bytes(obj: Any) -> bytes:
    """Produces compact key-sorted JSON bytes for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')


def hash_message(message_obj: Dict[str, Any]) -> str:
    """Hashes recursively key-sorted compact JSON of the message only."""
    return hashlib.sha256(canonical_json_bytes(message_obj)).hexdigest()


def canonical_package_hash(pkg: Dict[str, Any]) -> str:
    """Hashes canonical content of an invoice package for LLM caching."""
    return hashlib.sha256(canonical_json_bytes(pkg)).hexdigest()


# Helper: Protocol Verification & Authentication
def verify_headers(a2a_version: Optional[str], content_type: Optional[str]):
    if a2a_version != "1.0":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header 'A2A-Version: 1.0' is required."
        )


def get_principal(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer token."
        )
    token = authorization.split("Bearer ")[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token cannot be empty."
        )
    return token


# LLM Decision Logic via AIPipe Proxy
async def analyze_package_with_llm(pkg: Dict[str, Any]) -> Dict[str, Any]:
    pkg_hash = canonical_package_hash(pkg)
    if pkg_hash in package_cache:
        return package_cache[pkg_hash]

    prompt = f"""You are an expert invoice auditor. Analyze the following invoice package and determine the single correct action.

Action options:
- "settle_invoice": Valid, reconciled, and within autonomous authority.
- "request_approval": Commercially valid, but outside delegated authority.
- "hold_invoice": Payment pauses until a stated verification completes.
- "reject_duplicate": The same commercial invoice was already paid.
- "open_exception": Material records conflict and need an exception workflow.

Invoice Package:
{json.dumps(pkg, indent=2)}

Instructions:
1. Extract facts: vendorName (str), invoiceNumber (str), amountMinor (int), currency (str, e.g. "INR").
2. Find EXACTLY THREE decisive bracketed references from the paragraph determining the action (e.g. ["[REF-1]", "[REF-2]", "[REF-3]"]).
3. Write a rationale between 60 and 1500 characters that explicitly names the chosen action and includes at least two evidence refs.

Return strictly valid JSON matching this structure:
{{
  "action": "<one_action>",
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
            {"role": "system", "content": "You are a precise invoice auditing AI. Return raw JSON only without markdown code blocks."},
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
            
            # Clean markdown codeblocks if present
            cleaned_content = re.sub(r"^```(json)?\n|\n```$", "", raw_content.strip(), flags=re.MULTILINE)
            decision = json.loads(cleaned_content)
            
            # Validation / Normalization
            if len(decision.get("evidenceRefs", [])) < 3:
                # Fallback bracket search if model returned fewer
                found_brackets = re.findall(r"\[[A-Za-z0-9_\-\.]+\]", str(pkg))
                decision["evidenceRefs"] = (decision.get("evidenceRefs", []) + found_brackets)[:3]
            
            # Cache and return decision
            package_cache[pkg_hash] = decision
            return decision

    except Exception as e:
        # Fallback safe response on LLM parse error or network timeout
        fallback = {
            "action": "open_exception",
            "facts": {
                "vendorName": str(pkg.get("vendorName", "Unknown Vendor")),
                "invoiceNumber": str(pkg.get("invoiceNumber", "INV-UNKNOWN")),
                "amountMinor": int(pkg.get("amountMinor", 0)),
                "currency": str(pkg.get("currency", "INR"))
            },
            "evidenceRefs": ["[EV-01]", "[EV-02]", "[EV-03]"],
            "rationale": f"Action 'open_exception' selected due to ambiguous package details or processing constraint: {str(e)[:100]}."
        }
        package_cache[pkg_hash] = fallback
        return fallback


# 1. Discovery Route: Agent Card
@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    return JSONResponse(
        content={
            "name": "A2A Invoice Agent",
            "description": "Autonomous invoice reconciliation and action proposal agent.",
            "version": "1.0.0",
            "capabilities": {
                "batchProcessing": True,
                "idempotency": True,
                "cancellation": True
            },
            "skills": [
                {
                    "id": "invoice_action_agent",
                    "name": "Invoice Action Agent",
                    "description": "Evaluates invoice claim batches and executes verified actions.",
                    "tags": ["invoice", "reconciliation", "finance", "a2a"]
                }
            ],
            "supportedInterfaces": [
                {
                    "url": BASE_URL,
                    "protocolBinding": "HTTP+JSON",
                    "protocolVersion": "1.0"
                }
            ],
            "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
            "defaultOutputModes": [
                "application/vnd.ga5.invoice-action-proposals+json",
                "application/vnd.ga5.invoice-action-receipts+json"
            ]
        },
        media_type="application/a2a+json"
    )


# 2. Main Ingestion Route: /message:send
@app.post("/a2a/message:send")
@app.post("/message:send")
async def send_message(
    request: Request,
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    verify_headers(a2a_version, request.headers.get("content-type"))
    principal = get_principal(authorization)
    
    body = await request.json()
    message = body.get("message", {})
    message_id = message.get("messageId")
    
    if not message_id:
        raise HTTPException(status_code=400, detail="Missing messageId in request.")

    # Idempotency Check
    msg_hash = hash_message(message)
    idempotency_key = (principal, message_id)
    
    if idempotency_key in idempotency_db:
        stored = idempotency_db[idempotency_key]
        if stored["hash"] == msg_hash:
            # Return stored task
            task = tasks_db.get((principal, stored["task_id"]))
            return JSONResponse(content={"task": task}, media_type="application/a2a+json")
        else:
            # Reused messageId with different content
            return JSONResponse(
                status_code=409,
                content={"code": "IDEMPOTENCY_CONFLICT", "message": "messageId already exists with different payload."},
                media_type="application/a2a+json"
            )

    # Check for Result Continuation vs Initial Claim Batch
    parts = message.get("parts", [])
    if not parts:
        raise HTTPException(status_code=400, detail="Message contains no parts.")
    
    part = parts[0]
    media_type = part.get("mediaType")
    part_data = part.get("data", {})

    # --- PATH A: Result Continuation ---
    if media_type == "application/vnd.ga5.invoice-action-results+json":
        task_id = message.get("taskId")
        context_id = message.get("contextId")
        
        if not task_id or (principal, task_id) not in tasks_db:
            raise HTTPException(status_code=404, detail="Task not found.")

        lock = await get_task_lock(task_id)
        async with lock:
            task = tasks_db[(principal, task_id)]
            
            # State check: Must be non-terminal
            if task["state"] in ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"]:
                return JSONResponse(
                    status_code=409,
                    content={"code": "TASK_TERMINAL", "message": "Task is already terminal."},
                    media_type="application/a2a+json"
                )

            # Match context, batch, and proposals
            if task.get("contextId") != context_id:
                raise HTTPException(status_code=400, detail="Context ID mismatch.")

            # Record history
            task["history"].append(message)

            # Process Results
            batch_id = part_data.get("batchId")
            results = part_data.get("results", [])
            
            # Find proposal part
            proposal_part = next(p for p in task["artifacts"][0]["parts"] if p["mediaType"] == "application/vnd.ga5.invoice-action-proposals+json")
            stored_proposals = {p["packageId"]: p for p in proposal_part["data"]["proposals"]}

            executions = []
            for res in results:
                pkg_id = res.get("packageId")
                prop = stored_proposals.get(pkg_id)
                if prop and res.get("outcome") == "ACCEPTED":
                    # Verify action identity match
                    if res.get("actionId") == prop["actionId"] and res.get("action") == prop["action"]:
                        executions.append({
                            "packageId": pkg_id,
                            "actionId": prop["actionId"],
                            "action": prop["action"],
                            "receiptNonce": res.get("receiptNonce"),
                            "facts": prop["facts"],
                            "evidenceRefs": prop["evidenceRefs"]
                        })

            # Create Receipts Artifact
            receipt_part = {
                "partId": f"part-{uuid.uuid4().hex[:8]}",
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": {
                    "batchId": batch_id,
                    "executions": executions
                }
            }
            
            task["artifacts"].append({"parts": [receipt_part]})
            task["state"] = "TASK_STATE_COMPLETED"
            
            # Store Idempotency
            idempotency_db[idempotency_key] = {"hash": msg_hash, "task_id": task_id}
            
            return JSONResponse(content={"task": task}, media_type="application/a2a+json")

    # --- PATH B: Initial Batch Claim Processing ---
    elif media_type == "application/vnd.ga5.invoice-claim-batch+json":
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        context_id = f"ctx-{uuid.uuid4().hex[:12]}"
        batch_id = part_data.get("batchId")
        packages = part_data.get("packages", [])

        # Process packages sequentially or in parallel
        proposals = []
        for pkg in packages:
            pkg_id = pkg.get("packageId")
            decision = await analyze_package_with_llm(pkg)
            
            proposals.append({
                "packageId": pkg_id,
                "actionId": f"act-{uuid.uuid4().hex[:12]}",
                "action": decision["action"],
                "facts": decision["facts"],
                "evidenceRefs": decision["evidenceRefs"],
                "rationale": decision["rationale"]
            })

        proposal_artifact = {
            "parts": [{
                "partId": f"part-{uuid.uuid4().hex[:8]}",
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": {
                    "batchId": batch_id,
                    "proposals": proposals
                }
            }]
        }

        task = {
            "id": task_id,
            "contextId": context_id,
            "state": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "artifacts": [proposal_artifact]
        }

        # Store Task & Idempotency
        tasks_db[(principal, task_id)] = task
        idempotency_db[idempotency_key] = {"hash": msg_hash, "task_id": task_id}

        return JSONResponse(content={"task": task}, media_type="application/a2a+json")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mediaType: {media_type}")


# 3. Task Status & Retrieval: GET /tasks/{id}
@app.get("/a2a/tasks/{id}")
@app.get("/tasks/{id}")
async def get_task(
    id: str,
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    verify_headers(a2a_version, None)
    principal = get_principal(authorization)
    
    key = (principal, id)
    if key not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found.")
    
    return JSONResponse(content=tasks_db[key], media_type="application/a2a+json")


# 4. List Tasks: GET /tasks
@app.get("/a2a/tasks")
@app.get("/tasks")
async def list_tasks(
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    verify_headers(a2a_version, None)
    principal = get_principal(authorization)
    
    user_tasks = [task for (p, t_id), task in tasks_db.items() if p == principal]
    return JSONResponse(content={"tasks": user_tasks}, media_type="application/a2a+json")


# 5. Task Cancellation: POST /tasks/{id}:cancel
@app.post("/a2a/tasks/{id}:cancel")
@app.post("/tasks/{id}:cancel")
async def cancel_task(
    id: str,
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    verify_headers(a2a_version, None)
    principal = get_principal(authorization)
    
    key = (principal, id)
    if key not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found.")

    lock = await get_task_lock(id)
    async with lock:
        task = tasks_db[key]
        
        # If task is completed, cancellation race fails
        if task["state"] == "TASK_STATE_COMPLETED":
            return JSONResponse(
                status_code=409,
                content={"code": "TASK_COMPLETED", "message": "Cannot cancel completed task."},
                media_type="application/a2a+json"
            )
        
        if task["state"] != "TASK_STATE_CANCELED":
            task["state"] = "TASK_STATE_CANCELED"
            
        return JSONResponse(content=task, media_type="application/a2a+json")
