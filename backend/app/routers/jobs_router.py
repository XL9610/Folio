"""Router for async ingestion jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..repositories.ingestion_job_repository import get_job, list_jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job_endpoint(job_id: str) -> dict:
    """Return a single ingestion job's status."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.get("")
async def list_jobs_endpoint(notebook_id: str = "default") -> list[dict]:
    """List ingestion jobs for a notebook, newest first."""
    return list_jobs(notebook_id)
