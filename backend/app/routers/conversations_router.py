"""FastAPI router — save/load chat conversations."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db.session import get_connection
from ..repositories.notebook_repository import get_or_create_default

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class SaveConversationRequest(BaseModel):
    id: str = ""
    notebook_id: str = "default"
    title: str = ""
    messages: list[dict] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    id: str
    title: str
    messages: list[dict]
    source_document_ids: list[str]
    created_at: str = ""


@router.post("", response_model=ConversationResponse)
async def save_conversation(request: SaveConversationRequest):
    if request.notebook_id == "default":
        nb = get_or_create_default()
        notebook_id = nb["id"]
    else:
        notebook_id = request.notebook_id

    conv_id = request.id or str(uuid.uuid4())
    conn = get_connection()

    if request.id:
        owner = conn.execute(
            "SELECT notebook_id FROM saved_conversations WHERE id = ?", (request.id,)
        ).fetchone()
        if owner is not None and owner["notebook_id"] != notebook_id:
            raise HTTPException(
                status_code=409, detail="Conversation id belongs to a different notebook."
            )

    conn.execute(
        """INSERT INTO saved_conversations
           (id, notebook_id, title, messages_json, source_document_ids)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             title = excluded.title,
             messages_json = excluded.messages_json,
             source_document_ids = excluded.source_document_ids""",
        (
            conv_id,
            notebook_id,
            request.title,
            json.dumps(request.messages, ensure_ascii=False),
            json.dumps(request.source_document_ids),
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM saved_conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    d = dict(row)
    return ConversationResponse(
        id=d["id"],
        title=d["title"],
        messages=json.loads(d["messages_json"]),
        source_document_ids=json.loads(d["source_document_ids"]),
        created_at=d["created_at"],
    )


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(notebook_id: str = "default"):
    if notebook_id == "default":
        nb = get_or_create_default()
        notebook_id = nb["id"]
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM saved_conversations WHERE notebook_id = ? ORDER BY created_at DESC",
        (notebook_id,),
    ).fetchall()
    return [
        ConversationResponse(
            id=r["id"],
            title=r["title"],
            messages=json.loads(r["messages_json"]),
            source_document_ids=json.loads(r["source_document_ids"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conn = get_connection()
    cur = conn.execute(
        "DELETE FROM saved_conversations WHERE id = ?", (conversation_id,)
    )
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}
