"""TTS API endpoint."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from backend.services.tts_service import generate_speech_stream

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # 'male', 'female', or specific voice name


@router.post("/", response_class=StreamingResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech audio via Groq TTS.
    
    Args:
        text: The text to convert
        voice: 'male', 'female', or a specific voice name
    
    Returns streaming audio (mp3).
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        audio_stream = generate_speech_stream(
            request.text.strip(),
            voice=request.voice or "male"
        )
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
