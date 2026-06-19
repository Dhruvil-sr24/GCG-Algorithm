 
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM

from api.schemas import (
    AttackRequest, JobResponse, JobStatusResponse,
    JobStatus, AttackResultData,
)
from gcg.attack import GCGAttack, GCGResult
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
 
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_jobs: dict[str, "_JobRecord"] = {}


class _JobRecord:
    def __init__(self, job_id: str, goal: str, target_str: str):
        self.job_id = job_id
        self.goal = goal
        self.target_str = target_str
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.result: Optional[GCGResult] = None
        self.error: Optional[str] = None
        self._cancelled = False

 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg.model.dtype, torch.float32)

    logger.info(f"Loading {cfg.model.name} on {cfg.model.device} ({cfg.model.dtype})...")
    t0 = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        device_map=cfg.model.device if cfg.model.device != "cpu" else None,
    )
    if cfg.model.device == "cpu":
        _model = _model.to("cpu")
    _model.eval()

    logger.info(f"Model loaded in {time.time() - t0:.1f}s — ready.")
    yield
    del _model, _tokenizer

 
app = FastAPI(
    title="GCG Adversarial Testing API",
    description=(
        "Adversarial suffix generation using the Greedy Coordinate Gradient algorithm "
        "(Zou et al., 2023). White-box attack against HuggingFace CausalLMs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

 
async def _run_attack(
    job_id: str,
    num_steps: Optional[int],
    suffix_len: Optional[int],
):
    record = _jobs[job_id]
    record.status = JobStatus.RUNNING

    try:
        loop = asyncio.get_event_loop()
        attack = GCGAttack(
            model=_model,
            tokenizer=_tokenizer,
            goal=record.goal,
            target_str=record.target_str,
            num_steps=num_steps,
            suffix_len=suffix_len,
        )
        result: GCGResult = await loop.run_in_executor(None, attack.run)

        if record._cancelled:
            record.status = JobStatus.FAILED
            record.error = "Cancelled"
            return

        record.result = result
        record.status = JobStatus.COMPLETED

    except Exception as exc:
        record.status = JobStatus.FAILED
        record.error = str(exc)
        logger.exception(f"[Job {job_id}] Failed: {exc}")


def _result_to_schema(r: GCGResult) -> AttackResultData:
    return AttackResultData(
        goal=r.goal,
        target_str=r.target_str,
        success=r.success,
        best_suffix=r.best_suffix,
        best_loss=r.best_loss,
        best_generated=r.best_generated,
        steps_run=r.steps_run,
        steps_to_success=r.steps_to_success,
        total_elapsed_sec=r.total_elapsed_sec,
        loss_history=r.loss_history,
    )

 
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "model": cfg.model.name,
        "device": cfg.model.device,
        "active_jobs": sum(1 for j in _jobs.values() if j.status == JobStatus.RUNNING),
    }


@app.post("/v1/attack", response_model=JobResponse, status_code=202, tags=["Attack"])
async def submit_attack(req: AttackRequest, background_tasks: BackgroundTasks): 
    job_id = str(uuid.uuid4())
    record = _JobRecord(job_id=job_id, goal=req.goal, target_str=req.target_str)
    _jobs[job_id] = record

    background_tasks.add_task(
        _run_attack, job_id=job_id,
        num_steps=req.num_steps,
        suffix_len=req.suffix_len,
    )

    logger.info(f"Job queued: {job_id} | goal={req.goal[:60]!r}")
    return JobResponse(
        job_id=job_id, status=JobStatus.QUEUED,
        created_at=record.created_at, goal=req.goal,
    )


@app.get("/v1/attack/{job_id}/status", response_model=JobStatusResponse, tags=["Attack"])
async def get_status(job_id: str): 
    record = _jobs.get(job_id)
    if not record:
        raise HTTPException(404, f"Job {job_id!r} not found")
    return JobStatusResponse(
        job_id=job_id, status=record.status, created_at=record.created_at
    )


@app.get("/v1/attack/{job_id}", response_model=JobResponse, tags=["Attack"])
async def get_result(job_id: str): 
    record = _jobs.get(job_id)
    if not record:
        raise HTTPException(404, f"Job {job_id!r} not found")

    result_data = None
    if record.status == JobStatus.COMPLETED and record.result:
        result_data = _result_to_schema(record.result)

    return JobResponse(
        job_id=job_id, status=record.status,
        created_at=record.created_at, goal=record.goal,
        result=result_data, error=record.error,
    )


@app.delete("/v1/attack/{job_id}", tags=["Attack"])
async def cancel_job(job_id: str): 
    record = _jobs.get(job_id)
    if not record:
        raise HTTPException(404, f"Job {job_id!r} not found")
    if record.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(409, f"Job already {record.status.value}")
    record._cancelled = True
    return {"job_id": job_id, "message": "Cancellation requested"}
