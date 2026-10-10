"""Shared helpers for routers."""

from fastapi import HTTPException

from ..repositories.notebook_repository import get_notebook, get_or_create_default


def resolve_notebook_id(notebook_id: str) -> str:
    """Resolve a notebook id, expanding the 'default' alias, else 404."""
    if notebook_id == "default":
        return str(get_or_create_default()["id"])
    found = get_notebook(notebook_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return str(found["id"])
