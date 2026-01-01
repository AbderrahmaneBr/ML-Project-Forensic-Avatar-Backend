"""NLP service for forensic hypothesis generation using Groq LLM."""
from collections.abc import Generator

from openai import OpenAI

from backend.config import GROQ_API_KEY

GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize Groq client (using OpenAI SDK)
_groq_client = None
if GROQ_API_KEY:
    _groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

SYSTEM_PROMPT = """You are a forensic analyst AI. Analyze evidence precisely and concisely.

For images: List objects detected, transcribe any text found, then provide a brief hypothesis.

Keep responses SHORT (under 150 words). Be direct and professional. No dramatic narration."""


def _confidence_label(confidence: float) -> str:
    """Convert confidence score to a label for the LLM."""
    if confidence >= 0.8:
        return "[HIGH]"
    elif confidence >= 0.5:
        return "[MEDIUM]"
    else:
        return "[LOW]"


def _build_prompt(
    detected_objects: list[dict],
    extracted_texts: list[dict],
    context: str | None = None
) -> str:
    """Build the user prompt for hypothesis generation."""
    objects_str = ", ".join([
        f"{_confidence_label(obj['confidence'])} {obj['label']}"
        for obj in detected_objects
    ]) if detected_objects else "No objects detected"

    texts_str = ", ".join([
        f"{_confidence_label(text.get('confidence', 0.7))} \"{text['text']}\""
        for text in extracted_texts
    ]) if extracted_texts else "No text extracted"

    context_str = f"\n\nAdditional Context: {context}" if context else ""

    return f"""Analyze this crime scene evidence and provide a narrated analysis:

Objects at the scene: {objects_str}

Text found at the scene: {texts_str}{context_str}

Provide your strongest hypothesis about what occurred based on this evidence."""


def generate_hypothesis(
    detected_objects: list[dict],
    extracted_texts: list[dict],
    context: str | None = None
) -> dict:
    """
    Generate a single forensic hypothesis based on detected objects and extracted text.
    Uses Groq LLM for inference.

    Returns the strongest hypothesis as a single dict with content and confidence.
    """
    if not _groq_client:
        return {
            "content": "Groq API key not configured. Cannot generate hypothesis.",
            "confidence": 0.0
        }

    user_prompt = _build_prompt(detected_objects, extracted_texts, context)

    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
        )

        content = response.choices[0].message.content or ""
        clean_content = " ".join(content.split())

        return {
            "content": clean_content,
            "confidence": 0.8
        }

    except Exception as e:
        return {
            "content": f"Unable to generate hypothesis: {str(e)}",
            "confidence": 0.0
        }


def generate_hypotheses_stream(
    detected_objects: list[dict],
    extracted_texts: list[dict],
    context: str | None = None
) -> Generator[str, None, None]:
    """
    Stream forensic hypotheses using Groq LLM.
    Yields each token as it's generated.
    """
    if not _groq_client:
        yield "[ERROR] Groq API key not configured. Cannot generate hypothesis."
        return

    user_prompt = _build_prompt(detected_objects, extracted_texts, context)

    try:
        stream = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except Exception as e:
        yield f"[ERROR] Groq error: {str(e)}"
