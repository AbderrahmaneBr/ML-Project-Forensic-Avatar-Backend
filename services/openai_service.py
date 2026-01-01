"""Service for OpenAI GPT-4o Vision analysis."""
from collections.abc import Generator
from openai import OpenAI
from backend.config import OPENAI_API_KEY

_client = None
if OPENAI_API_KEY:
    _client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a forensic detective AI.
Analyze the image provided.
1. Describe any objects, weapons, or people you see.
2. Transcribe any visible text.
3. Provide a brief theory on what is happening.

Format your response as a natural, spoken assessment to a colleague.
Keep it concise (under 100 words). Do NOT use headers like 'Analysis:' or 'Hypothesis:'."""

def generate_analysis(image_url: str, context: str | None = None) -> str:
    """Generate a full analysis of the image using GPT-4o."""
    if not _client:
        return "OpenAI API key not configured."

    user_content = [
        {"type": "text", "text": "Analyze this image." + (f"\nContext: {context}" if context else "")},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]

    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error regenerating analysis: {str(e)}"

def generate_analysis_stream(image_url: str, context: str | None = None) -> Generator[str, None, None]:
    """Stream the analysis from GPT-4o."""
    if not _client:
        yield "OpenAI API key not configured."
        return

    user_content = [
        {"type": "text", "text": "Analyze this image." + (f"\nContext: {context}" if context else "")},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]

    try:
        stream = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=300,
            stream=True
        )
        
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        yield f"Error streaming analysis: {str(e)}"
