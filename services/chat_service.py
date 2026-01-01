"""Chat service for conversation-based LLM interactions using Groq."""
from collections.abc import Generator
from typing import cast
from uuid import UUID

from openai import OpenAI
from sqlalchemy.orm import Session

from backend.db.models import Conversation, Message, MessageRole, Image, DetectedObject, ExtractedText
from backend.config import GROQ_API_KEY

GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize Groq client
_groq_client = None
if GROQ_API_KEY:
    _groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

# Conversational prompt for general chat (no evidence)
CHAT_SYSTEM_PROMPT = """You are a helpful forensic detective assistant. 
Be concise and direct. Keep responses under 100 words unless asked for detail."""

# Forensic analysis prompt for when there's actual evidence
FORENSIC_SYSTEM_PROMPT = """You are a forensic analyst. Analyze evidence and provide brief, actionable insights.
Keep responses SHORT (under 150 words).
Speak naturally as a forensic detective explaining the situation to a colleague.
Do NOT use headers like 'Analysis:' or 'Hypothesis:'. Do NOT list findings as bullet points unless necessary for clarity.
Be professional and direct."""


def _confidence_label(confidence: float) -> str:
    """Convert confidence score to a label for the LLM."""
    if confidence >= 0.8:
        return "[HIGH]"
    elif confidence >= 0.5:
        return "[MEDIUM]"
    else:
        return "[LOW]"


def _build_evidence_context(db: Session, conversation_id: UUID) -> str:
    """Build evidence context from all images in the conversation."""
    images = db.query(Image).filter(Image.conversation_id == conversation_id).all()

    if not images:
        return ""

    all_objects: list[str] = []
    all_texts: list[str] = []

    for image in images:
        image_id = cast(UUID, image.id)

        # Get detected objects
        objects = db.query(DetectedObject).filter(DetectedObject.image_id == image_id).all()
        for obj in objects:
            label = f"{_confidence_label(cast(float, obj.confidence))} {obj.label}"
            all_objects.append(label)

        # Get extracted texts
        texts = db.query(ExtractedText).filter(ExtractedText.image_id == image_id).all()
        for text in texts:
            conf_val = cast(float, text.confidence) if text.confidence is not None else 0.7
            label = f"{_confidence_label(conf_val)} \"{text.text}\""
            all_texts.append(label)

    if not all_objects and not all_texts:
        return "Analysis completed. No specific threats, objects, or text were detected in this image."

    objects_str = ", ".join(all_objects) if all_objects else "No objects detected"
    texts_str = ", ".join(all_texts) if all_texts else "No text extracted"

    return f"""Evidence from the scene:
Objects detected: {objects_str}
Text found: {texts_str}"""


def _build_message_history(
    db: Session,
    conversation_id: UUID,
    include_evidence: bool = True
) -> list[dict]:
    """Build the full message history for the LLM."""
    # Check if there's evidence in this conversation
    evidence_context = ""
    if include_evidence:
        evidence_context = _build_evidence_context(db, conversation_id)

    # Use forensic prompt if there's evidence, otherwise conversational
    if evidence_context:
        system_prompt = FORENSIC_SYSTEM_PROMPT
    else:
        system_prompt = CHAT_SYSTEM_PROMPT

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Add evidence context as the first user message if there's evidence
    if evidence_context:
        messages.append({
            "role": "user",
            "content": f"{evidence_context}\n\nAnalyze this evidence and provide your initial assessment."
        })

    # Get all conversation messages
    db_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at).all()

    for msg in db_messages:
        messages.append({
            "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
            "content": cast(str, msg.content)
        })

    return messages


def chat(
    db: Session,
    conversation_id: UUID,
    user_message: str
) -> str:
    """
    Send a message in a conversation and get a response.
    Maintains full conversation history for context.
    """
    if not _groq_client:
        return "[ERROR] Groq API key not configured."

    # Build message history
    messages = _build_message_history(db, conversation_id)

    # Add the new user message
    messages.append({"role": "user", "content": user_message})

    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
        content = response.choices[0].message.content or ""
        return " ".join(content.split())  # Clean up whitespace
    except Exception as e:
        return f"Unable to generate response: {str(e)}"


def chat_stream(
    db: Session,
    conversation_id: UUID,
    user_message: str
) -> Generator[str, None, None]:
    """
    Stream a chat response token by token using Groq.
    """
    if not _groq_client:
        yield "[ERROR] Groq API Key not configured."
        return

    # Build message history
    messages = _build_message_history(db, conversation_id)

    # Add the new user message
    messages.append({"role": "user", "content": user_message})

    try:
        stream = _groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            stream=True
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        yield f"[ERROR] Groq Error: {str(e)}"
