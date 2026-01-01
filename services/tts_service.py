"""Text-to-Speech service using Groq Playai TTS."""
import os
import requests
from typing import Generator

# Import config first to ensure .env is loaded
import backend.config  # noqa: F401

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_TTS_MODEL = "playai-tts"

# Voice configurations for male/female
GROQ_VOICES = {
    "male": "Angelo-PlayAI",
    "female": "Aaliyah-PlayAI",
}

DEFAULT_GENDER = "male"


def get_voice_for_gender(gender: str) -> str:
    """Get the appropriate voice for the given gender."""
    gender = gender.lower()
    return GROQ_VOICES.get(gender, GROQ_VOICES["male"])


def generate_speech_stream(text: str, voice: str = "male") -> Generator[bytes, None, None]:
    """
    Generate speech audio from text using Groq TTS.
    
    Args:
        text: The text to convert to speech.
        voice: The voice persona (or gender like 'male'/'female').
        
    Yields:
        Audio chunk bytes (mp3).
    """
    # Resolve voice to actual ID if needed
    if voice in ["male", "female"]:
        voice_id = get_voice_for_gender(voice)
    else:
        voice_lower = voice.lower()
        if voice_lower in ["male", "female"]:
            voice_id = get_voice_for_gender(voice_lower)
        else:
            voice_id = voice

    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set")
        yield b""
        return

    # Groq TTS endpoint
    url = "https://api.groq.com/openai/v1/audio/speech"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_TTS_MODEL,
        "input": text,
        "voice": voice_id,
        "response_format": "mp3"
    }

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code != 200:
            try:
                error_data = response.json()
                message = error_data.get("error", {}).get("message", "Unknown error")
            except:
                message = response.text
            print(f"Groq TTS API Error ({response.status_code}): {message}")
            yield b""
            return

        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                yield chunk

    except Exception as e:
        print(f"Groq TTS Connection Error: {e}")
        yield b""
