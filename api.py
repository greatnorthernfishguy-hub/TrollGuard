"""
TrollGuard — FastAPI REST API

HTTP interface for agent integration.  Exposes TrollGuard's scanning
capabilities as REST endpoints so AI agents, web frameworks, and
external tools can call TrollGuard without importing it as a library.

Endpoints:
    POST /scan/text     — Scan raw text (web pages, chat, API responses)
    POST /scan/file     — Scan a file on disk
    GET  /health        — Health check + NG-Lite stats
    GET  /stats         — Detailed pipeline and sentry telemetry
    POST /quarantine/review — Submit human review for quarantined items

The API is the primary integration point for runtime protection:
an agent's web search tool, chat handler, or API client sends text
to POST /scan/text and gets back sanitized content with the threat
verdict.

PRD reference: Section 7.3 — Integration with OpenClaw

# ---- Changelog ----
# [2026-02-19] Claude (Opus 4.6) — Grok security audit: non-blocking scans + background tasks.
#   What: Wrapped blocking scan calls in asyncio.to_thread() so ML
#         inference and file I/O don't block the event loop.  Added
#         POST /scan/file/async endpoint that accepts a file scan
#         request and returns a task_id immediately; the scan runs in
#         a background thread and results are polled via GET /task/{id}.
#   Why:  Grok flagged that synchronous scan calls in async endpoints
#         block the entire uvicorn event loop, stalling all concurrent
#         requests during a deep audit (Layer 3 Swarm can take seconds).
#
#   Claude's note: Grok suggested making the core scan() method itself
#   async.  That would require rewriting the entire pipeline, VectorSentry,
#   and SwarmAudit to be async — high effort, low value since the real
#   bottleneck is CPU-bound ML inference (NumPy/sklearn) which doesn't
#   benefit from async I/O.  asyncio.to_thread() is the right tool here:
#   it moves blocking work to a thread pool while keeping the event loop
#   free.  Full async rewrite deferred to Phase 3 if profiling shows
#   I/O-bound bottlenecks.
#
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: FastAPI application with scan/text, scan/file, health, stats,
#         and quarantine review endpoints.  Singleton pipeline instance
#         with NG-Lite + peer bridge, shared across requests.
#   Why:  Agents need an HTTP interface to call TrollGuard at runtime.
#         Library-level import (sentinel_client.sanitize()) works for
#         Python agents, but HTTP works for everything — Go agents,
#         Node.js agents, shell scripts, other modules in the E-T
#         ecosystem.  The systemd service in install.sh runs this.
#   Settings: Host/port from config.yaml (default 127.0.0.1:7438).
#         Bound to localhost by default — not exposed to the network.
#         Change to 0.0.0.0 in config.yaml only if you need remote
#         access (and add authentication first).
#   How:  Singleton TrollGuardPipeline created at startup, reused
#         across all requests.  NG-Lite state is saved periodically
#         and on shutdown via lifespan context manager.
# -------------------
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("trollguard.api")

# ---------------------------------------------------------------------------
# Singleton pipeline (created at startup, shared across requests)
# ---------------------------------------------------------------------------

_pipeline = None
_config = None
_sentry = None
_quarantine = None
_startup_time = 0.0


def _get_pipeline():
    """Lazy import to avoid circular deps at module level."""
    global _pipeline, _config, _sentry, _quarantine, _startup_time
    if _pipeline is None:
        from main import TrollGuardPipeline, load_config
        from sentinel_core.vector_sentry import VectorSentry
        from sentinel_core.quarantine_logger import QuarantineLogger

        _config = load_config()
        _pipeline = TrollGuardPipeline(_config)

        # Dedicated sentry instance for the API (shares NG-Lite with pipeline)
        sentry_config = _config.get("runtime_sentry", {})
        _sentry = VectorSentry(config=sentry_config, ng_lite=_pipeline._ng_lite)

        # Quarantine logger
        q_path = _config.get("persistence", {}).get("quarantine_path", "quarantine.json")
        _quarantine = QuarantineLogger(q_path)

        _startup_time = time.time()

    return _pipeline


# ---------------------------------------------------------------------------
# Lifespan: startup/shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("TrollGuard API starting up...")
    _get_pipeline()
    logger.info("Pipeline initialized, API ready on port %s",
                (_config or {}).get("api", {}).get("port", 7438))
    yield
    # Shutdown: persist NG-Lite state
    logger.info("TrollGuard API shutting down, saving state...")
    if _pipeline is not None:
        _pipeline.save_ng_state()
    logger.info("State saved. Goodbye.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TrollGuard",
    description="The Open-Source Immune System for AI Agents",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class ScanTextRequest(BaseModel):
    """Request body for POST /scan/text."""
    text: str = Field(..., description="Raw text to scan (HTML, chat, API response, etc.)")
    source: str = Field("unknown", description="Label for origin (e.g. 'web_search', 'user_chat', 'api_response')")
    mode: Optional[str] = Field(None, description="Override sentry mode: 'redact', 'block', 'report_only'")

class ScanTextResponse(BaseModel):
    """Response from POST /scan/text."""
    verdict: str = Field(..., description="SAFE, SUSPICIOUS, or MALICIOUS")
    sanitized_text: str = Field(..., description="Text after redaction/blocking applied")
    max_score: float = Field(..., description="Highest per-chunk threat score [0.0, 1.0]")
    chunks_scanned: int = Field(..., description="Number of chunks processed")
    flagged_chunks: int = Field(..., description="Chunks scoring above safe_ceiling")
    scan_time_ms: float = Field(..., description="Processing time in milliseconds")
    source: str = Field("", description="Echo of the source label")

class ScanFileRequest(BaseModel):
    """Request body for POST /scan/file."""
    file_path: str = Field(..., description="Absolute path to the file to scan")

class ScanFileResponse(BaseModel):
    """Response from POST /scan/file."""
    file_path: str
    file_hash: str
    final_verdict: str
    layers_run: List[int]
    total_time_ms: float
    fail_fast_triggered: bool
    layer_results: List[Dict[str, Any]]

class ReviewRequest(BaseModel):
    """Request body for POST /quarantine/review."""
    incident_id: str = Field(..., description="Quarantine incident ID")
    verdict: str = Field(..., description="'false_positive', 'true_positive', 'false_negative', or 'confirmed_threat'")
    reviewer: str = Field("api_user", description="Who is submitting the review")

class AsyncScanResponse(BaseModel):
    """Response from POST /scan/file/async — returns immediately."""
    task_id: str = Field(..., description="Unique task ID for polling results")
    status: str = Field("pending", description="Task status: pending, running, completed, failed")

class TaskStatusResponse(BaseModel):
    """Response from GET /task/{task_id}."""
    task_id: str
    status: str = Field(..., description="pending, running, completed, or failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Scan result when completed")
    error: Optional[str] = Field(None, description="Error message if failed")

class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: str
    uptime_seconds: float
    ng_lite_connected: bool
    peer_bridge_connected: bool
    pipeline_ready: bool


# ---------------------------------------------------------------------------
# Background task store (in-memory, bounded)
# ---------------------------------------------------------------------------

_task_results: Dict[str, Dict[str, Any]] = {}
_TASK_STORE_MAX = 200


def _prune_task_store() -> None:
    """Keep task store bounded by evicting oldest completed tasks."""
    if len(_task_results) <= _TASK_STORE_MAX:
        return
    completed = [
        (tid, t) for tid, t in _task_results.items()
        if t["status"] in ("completed", "failed")
    ]
    completed.sort(key=lambda x: x[1].get("finished_at", 0))
    to_remove = len(_task_results) - _TASK_STORE_MAX
    for tid, _ in completed[:to_remove]:
        del _task_results[tid]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/scan/text", response_model=ScanTextResponse)
async def scan_text(request: ScanTextRequest):
    """Scan raw text for prompt injection and adversarial content.

    This is the primary endpoint for runtime protection.  Send any
    text the agent encounters — web search results, user messages,
    API responses, repository contents — and get back sanitized text
    with the threat verdict.

    The sanitized_text field has dangerous content removed based on
    the operating mode:
      - report_only: text returned unchanged (logging only)
      - redact: suspicious/malicious chunks replaced with
        [TROLLGUARD: CONTENT REDACTED]
      - block: empty string if any chunk is malicious

    Usage from an agent's web search tool:
        response = requests.post("http://localhost:7438/scan/text", json={
            "text": raw_html,
            "source": "web_search"
        })
        safe_text = response.json()["sanitized_text"]
    """
    pipeline = _get_pipeline()

    def _do_text_scan():
        # Allow per-request mode override
        if request.mode and _sentry is not None:
            original_mode = _sentry.config["mode"]
            _sentry.config["mode"] = request.mode
        else:
            original_mode = None

        result = _sentry.scan(request.text, source=request.source)

        # Restore original mode
        if original_mode is not None:
            _sentry.config["mode"] = original_mode

        # Quarantine if malicious
        if result.verdict.value == "MALICIOUS" and _quarantine is not None:
            best_chunk = max(result.flagged_chunks, key=lambda c: c.score) if result.flagged_chunks else None
            _quarantine.log_incident(
                source=request.source,
                trigger_engine="vector_sentry_api",
                layer=4,
                vector_score=result.max_score,
                raw_text=request.text[:5000],
                vector_embedding=best_chunk.embedding.tolist() if best_chunk and best_chunk.embedding is not None else None,
            )
        return result

    # Run blocking ML scan in thread pool to avoid stalling the event loop
    result = await asyncio.to_thread(_do_text_scan)

    return ScanTextResponse(
        verdict=result.verdict.value,
        sanitized_text=result.sanitized_text,
        max_score=round(result.max_score, 4),
        chunks_scanned=result.chunks_scanned,
        flagged_chunks=len(result.flagged_chunks),
        scan_time_ms=round(result.scan_time_ms, 2),
        source=request.source,
    )


@app.post("/scan/file", response_model=ScanFileResponse)
async def scan_file(request: ScanFileRequest):
    """Scan a file through the full 4-layer pipeline.

    Runs Layers 0-3 (Emergency Stop, Cisco Static Analysis, Sentinel ML,
    Swarm Audit) with Fail Fast architecture.  Use this for scanning
    skill files, plugins, and repositories before installation.

    For runtime text scanning (web search results, chat, etc.),
    use POST /scan/text instead — it's faster and purpose-built for
    streaming content.
    """
    pipeline = _get_pipeline()

    if not Path(request.file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    # Run blocking pipeline in thread pool
    report = await asyncio.to_thread(pipeline.scan_file, request.file_path)

    return ScanFileResponse(
        file_path=report.file_path,
        file_hash=report.file_hash,
        final_verdict=report.final_verdict.value,
        layers_run=report.layers_run,
        total_time_ms=round(report.total_time_ms, 2),
        fail_fast_triggered=report.fail_fast_triggered,
        layer_results=[
            {
                "layer": lr.layer,
                "name": lr.name,
                "verdict": lr.verdict.value,
                "score": round(lr.score, 4),
                "elapsed_ms": round(lr.elapsed_ms, 2),
            }
            for lr in report.layer_results
        ],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check.  Use this for monitoring and load balancer probes."""
    pipeline = _get_pipeline()

    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _startup_time, 1),
        ng_lite_connected=pipeline._ng_lite is not None,
        peer_bridge_connected=pipeline._peer_bridge is not None,
        pipeline_ready=True,
    )


@app.get("/stats")
async def stats():
    """Detailed telemetry: ecosystem tier, NG-Lite state, sentry counters, quarantine stats."""
    pipeline = _get_pipeline()

    result: Dict[str, Any] = {
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }

    # Prefer unified ecosystem stats when available
    if pipeline._eco is not None:
        result["ecosystem"] = pipeline._eco.stats()
    else:
        if pipeline._ng_lite is not None:
            result["ng_lite"] = pipeline._ng_lite.get_stats()
        if pipeline._peer_bridge is not None:
            result["peer_bridge"] = pipeline._peer_bridge.get_stats()

    if _sentry is not None:
        result["sentry"] = _sentry.get_stats()

    if _quarantine is not None:
        result["quarantine"] = _quarantine.get_stats()

    return result


@app.post("/quarantine/review")
async def quarantine_review(request: ReviewRequest):
    """Submit a human review for a quarantined incident.

    This feeds the active learning loop (PRD §8.2.2).  Marking
    incidents as false_positive or confirmed_threat improves the
    classifier over time.
    """
    if _quarantine is None:
        raise HTTPException(status_code=503, detail="Quarantine logger not initialized")

    valid_verdicts = {"false_positive", "true_positive", "false_negative", "confirmed_threat"}
    if request.verdict not in valid_verdicts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid verdict. Must be one of: {', '.join(valid_verdicts)}",
        )

    success = _quarantine.mark_reviewed(
        incident_id=request.incident_id,
        review=request.verdict,
        reviewer=request.reviewer,
    )

    if not success:
        raise HTTPException(status_code=404, detail=f"Incident not found: {request.incident_id}")

    # Check if enough reviews have accumulated to trigger retraining
    retrain_status = _quarantine.check_retrain_ready()
    result = {
        "status": "reviewed",
        "incident_id": request.incident_id,
        "verdict": request.verdict,
        "retrain_ready": retrain_status["ready"],
    }

    if retrain_status["ready"]:
        logger.info(
            "Retrain recommended: %d reviewed samples (threshold=%d)",
            retrain_status["training_samples"],
            retrain_status["threshold"],
        )
        result["retrain_info"] = retrain_status

    return result


@app.post("/scan/file/async", response_model=AsyncScanResponse)
async def scan_file_async(request: ScanFileRequest):
    """Submit a file scan as a background task.

    Returns immediately with a task_id.  Poll GET /task/{task_id}
    for results.  Use this for deep audits (Layer 3 Swarm) that
    may take several seconds — the caller isn't blocked waiting.
    """
    pipeline = _get_pipeline()

    if not Path(request.file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    task_id = str(uuid.uuid4())
    _task_results[task_id] = {"status": "pending", "result": None, "error": None}
    _prune_task_store()

    async def _run_scan():
        _task_results[task_id]["status"] = "running"
        try:
            report = await asyncio.to_thread(pipeline.scan_file, request.file_path)
            _task_results[task_id]["status"] = "completed"
            _task_results[task_id]["finished_at"] = time.time()
            _task_results[task_id]["result"] = {
                "file_path": report.file_path,
                "file_hash": report.file_hash,
                "final_verdict": report.final_verdict.value,
                "layers_run": report.layers_run,
                "total_time_ms": round(report.total_time_ms, 2),
                "fail_fast_triggered": report.fail_fast_triggered,
                "layer_results": [
                    {
                        "layer": lr.layer,
                        "name": lr.name,
                        "verdict": lr.verdict.value,
                        "score": round(lr.score, 4),
                        "elapsed_ms": round(lr.elapsed_ms, 2),
                    }
                    for lr in report.layer_results
                ],
            }
        except Exception as e:
            _task_results[task_id]["status"] = "failed"
            _task_results[task_id]["finished_at"] = time.time()
            _task_results[task_id]["error"] = str(e)
            logger.error("Background scan failed for %s: %s", request.file_path, e)

    asyncio.create_task(_run_scan())

    return AsyncScanResponse(task_id=task_id, status="pending")


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Poll for the result of an async scan task."""
    if task_id not in _task_results:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = _task_results[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error"),
    )


@app.get("/retrain/status")
async def retrain_status():
    """Check if enough reviewed quarantine data has accumulated for retraining.

    Use this endpoint in a cron job or CI pipeline to decide whether
    to trigger train_model.py.  The active learning loop:
      1. Incidents are quarantined during scanning
      2. Humans review them via POST /quarantine/review
      3. This endpoint signals when the reviewed pool is large enough
      4. External process runs train_model.py with the new data
    """
    if _quarantine is None:
        raise HTTPException(status_code=503, detail="Quarantine logger not initialized")

    status = _quarantine.check_retrain_ready()
    status["training_data_available"] = len(_quarantine.export_training_data())
    return status
