"""Chat API endpoints for conversation-based interactions."""
import asyncio
import json
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from backend.db.database import get_db
from backend.db.models import Conversation, Message, MessageRole
from backend.services.chat_service import chat, chat_stream
from backend.schemas.schemas import ChatRequest, ChatResponse, MessageResponse

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)


@router.post("/{conversation_id}/chat", response_model=ChatResponse)
def send_message(
    conversation_id: UUID,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Send a message in a conversation and get a response.
    Messages are saved to the database for context.
    """
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message
    user_msg = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=request.message
    )
    db.add(user_msg)
    db.commit()

    # Get LLM response
    response_content = chat(db, conversation_id, request.message)

    # Save assistant message
    assistant_msg = Message(
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=response_content
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(
        message=MessageResponse(
            id=cast(UUID, assistant_msg.id),
            role=assistant_msg.role.value,
            content=cast(str, assistant_msg.content),
            created_at=cast(datetime, assistant_msg.created_at)
        ),
        conversation_id=conversation_id
    )


async def run_in_thread(func, *args):
    """Run a sync function in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


async def chat_stream_generator(
    conversation_id: UUID,
    user_message: str,
    db: Session
) -> AsyncGenerator[dict, None]:
    """Generate SSE events for streaming chat."""
    # Save user message first
    user_msg = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=user_message
    )
    db.add(user_msg)
    db.commit()

    yield {"event": "start", "data": json.dumps({"status": "streaming"})}
    await asyncio.sleep(0)

    full_content = ""
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def stream_to_queue():
        for token in chat_stream(db, conversation_id, user_message):
            queue.put_nowait(token)
        queue.put_nowait(None)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, stream_to_queue)

    while True:
        try:
            token = await asyncio.wait_for(queue.get(), timeout=0.1)
            if token is None:
                break
            full_content += token
            yield {"event": "token", "data": json.dumps({"text": token})}
        except asyncio.TimeoutError:
            await asyncio.sleep(0.01)
            continue

    # Save assistant message
    assistant_msg = Message(
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=full_content.strip()
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    yield {
        "event": "complete",
        "data": json.dumps({
            "message_id": str(assistant_msg.id),
            "content": full_content.strip()
        })
    }


@router.post("/{conversation_id}/chat/stream")
async def send_message_stream(
    conversation_id: UUID,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Send a message and stream the response using Server-Sent Events.
    """
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return EventSourceResponse(
        chat_stream_generator(conversation_id, request.message, db),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
