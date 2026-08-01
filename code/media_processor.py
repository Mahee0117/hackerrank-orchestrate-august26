"""
media_processor.py
------------------
Detects and processes media attached to an incoming message.

Public API
----------
    process_media(msg, data) → str

    Returns a plain-text description of the media content (or an empty
    string if the message has no media). The caller can inject this text
    into the existing context/prompt pipeline without knowing anything
    about how it was produced.

Architecture
------------
    Image messages  → Gemma4 Vision (Ollama multimodal)
    Voice messages  → SpeechTranscriber strategy
                      • WhisperTranscriber  — if openai-whisper is installed
                      • FallbackTranscriber — structured placeholder otherwise

All failures are caught and returned as descriptive strings so the
pipeline never crashes due to a missing file or a model error.
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Project layout ─────────────────────────────────────────────────────────────
_CODE_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _CODE_DIR.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"

# Ollama model used for vision extraction (same model as the text pipeline)
VISION_MODEL = "gemma4"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(val) -> str:
    """Return empty string for NaN/None, otherwise stripped str."""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _resolve_media_path(relative_path: str) -> Path | None:
    """
    Resolve a media file path from images.csv / voice_notes.csv.

    The CSV stores paths relative to the dataset directory
    (e.g. 'media/images/img_001.jpg').  We resolve against DATASET_DIR.
    Returns None if the file does not exist.
    """
    if not relative_path:
        return None
    full = DATASET_DIR / relative_path
    return full if full.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Image handler — Gemma4 Vision via Ollama
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTRACTION_PROMPT = """\
You are a WhatsApp message content extractor.

This image was received as a WhatsApp message. It may be a:
- Promotional poster or advertisement
- Payment slip or transaction receipt
- Official notice or announcement
- Screenshot of a conversation or website
- Scam or phishing poster
- Event invitation

Extract ALL of the following that are present:
1. Main heading or title
2. Sender or brand name
3. Key body text (offers, warnings, deadlines, amounts)
4. Call-to-action (e.g. "Click here", "Pay now", "Register", "OTP")
5. Any URLs, phone numbers, or account details shown
6. Any urgency or risk signals (e.g. "LIMITED TIME", "ACT NOW", suspicious links)

Return a compact structured summary. Do NOT describe visual design elements
(colors, fonts, layout). Focus ONLY on the informational content.
If the image is unclear or contains no readable text, say so briefly.
"""


def _extract_image_text(image_path: Path) -> str:
    """
    Send an image to Gemma4 Vision via Ollama and return extracted text.

    Uses the Ollama `images` field to pass a base64-encoded JPEG/PNG.
    The response is the raw text extraction — no JSON wrapping needed here.
    """
    import ollama  # imported here so the rest of the module works without it

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = ollama.chat(
            model=VISION_MODEL,
            options={"temperature": 0.1},   # low temp for factual extraction
            messages=[
                {
                    "role": "user",
                    "content": _IMAGE_EXTRACTION_PROMPT,
                    "images": [b64],
                }
            ],
        )

        extracted = response["message"]["content"].strip()
        logger.info("Image extracted: %s … (%d chars)", extracted[:60], len(extracted))
        return extracted

    except Exception as exc:
        logger.warning("Vision extraction failed for %s: %s", image_path, exc)
        return f"[Image extraction failed: {exc}]"


def _process_image(msg: pd.Series, data: dict) -> str:
    """
    Look up the image file for this message and extract its text content.

    Lookup path: msg.media_id → images.csv → file_path → disk
    """
    media_id = _safe(msg.get("media_id", ""))
    if not media_id:
        return "[Image message: no media_id found]"

    images_df: pd.DataFrame = data.get("images", pd.DataFrame())
    if images_df.empty:
        return f"[Image message: images.csv not loaded — media_id={media_id}]"

    row = images_df[images_df["image_id"] == media_id]
    if row.empty:
        return f"[Image message: media_id={media_id} not found in images.csv]"

    relative_path = _safe(row.iloc[0].get("file_path", ""))
    image_path    = _resolve_media_path(relative_path)

    if image_path is None:
        return f"[Image file not found on disk: {relative_path}]"

    extracted = _extract_image_text(image_path)
    return f"[EXTRACTED IMAGE CONTENT]\n{extracted}"


# ─────────────────────────────────────────────────────────────────────────────
# Voice handler — SpeechTranscriber strategy
# ─────────────────────────────────────────────────────────────────────────────

class SpeechTranscriber(ABC):
    """Abstract base for speech-to-text backends."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return a text transcription of the audio file."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""
        ...


class WhisperTranscriber(SpeechTranscriber):
    """
    Transcribes audio using OpenAI Whisper (local, offline).

    Activated automatically when `openai-whisper` is installed:
        pip install openai-whisper

    Uses the 'base' model for speed. Change model_name to 'small'
    or 'medium' for higher accuracy at the cost of more memory.
    """

    MODEL_NAME = "base"

    def __init__(self) -> None:
        import whisper  # noqa: F401 — intentionally deferred
        self._whisper = whisper

    @property
    def name(self) -> str:
        return f"WhisperTranscriber(model={self.MODEL_NAME})"

    def transcribe(self, audio_path: Path) -> str:
        try:
            model  = self._whisper.load_model(self.MODEL_NAME)
            result = model.transcribe(str(audio_path))
            text   = result.get("text", "").strip()
            logger.info("Whisper transcribed: %s … (%d chars)", text[:60], len(text))
            return text if text else "[Voice note: transcription produced no text]"
        except Exception as exc:
            logger.warning("Whisper transcription failed: %s", exc)
            return f"[Whisper transcription failed: {exc}]"


class FallbackTranscriber(SpeechTranscriber):
    """
    Used when no speech-to-text backend is installed.

    Returns a structured placeholder that tells the LLM this is a voice
    message so it can still route using conversation context, sender trust,
    and historical behaviour — rather than just having an empty field.

    To upgrade: install openai-whisper and re-run. No code change needed.
    """

    @property
    def name(self) -> str:
        return "FallbackTranscriber(no STT engine installed)"

    def transcribe(self, audio_path: Path) -> str:
        filename = audio_path.name
        return (
            f"[Voice note: {filename}]\n"
            "[No speech-to-text engine is available. "
            "Route this message based on conversation context, "
            "sender trust, and historical behaviour. "
            "Install openai-whisper to enable automatic transcription.]"
        )


def get_transcriber() -> SpeechTranscriber:
    """
    Factory: return the best available SpeechTranscriber.

    Priority:
      1. WhisperTranscriber — if openai-whisper is importable
      2. FallbackTranscriber — safe no-op placeholder

    Adding a new backend: implement SpeechTranscriber and add it here.
    """
    try:
        import whisper  # noqa: F401
        logger.info("Using WhisperTranscriber")
        return WhisperTranscriber()
    except ImportError:
        logger.info("openai-whisper not found — using FallbackTranscriber")
        return FallbackTranscriber()


def _process_voice(msg: pd.Series, data: dict) -> str:
    """
    Look up the audio file for this message and transcribe it.

    Lookup path: msg.media_id → voice_notes.csv → file_path → disk
    """
    media_id = _safe(msg.get("media_id", ""))
    if not media_id:
        return "[Voice message: no media_id found]"

    voice_df: pd.DataFrame = data.get("voice_notes", pd.DataFrame())
    if voice_df.empty:
        return f"[Voice message: voice_notes.csv not loaded — media_id={media_id}]"

    row = voice_df[voice_df["voice_note_id"] == media_id]
    if row.empty:
        return f"[Voice message: media_id={media_id} not found in voice_notes.csv]"

    relative_path = _safe(row.iloc[0].get("file_path", ""))
    audio_path    = _resolve_media_path(relative_path)

    if audio_path is None:
        return f"[Audio file not found on disk: {relative_path}]"

    transcriber = get_transcriber()
    transcript  = transcriber.transcribe(audio_path)
    return f"[VOICE NOTE TRANSCRIPT — {transcriber.name}]\n{transcript}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_media(msg: pd.Series, data: dict) -> str:
    """
    Detect media type and return extracted text content.

    Parameters
    ----------
    msg  : one row from messages.csv
    data : dict returned by loader.load_data()

    Returns
    -------
    str
        - Empty string      — message has no media (text-only)
        - Extracted text    — image content extracted by Gemma4 Vision
        - Transcription     — voice note text from Whisper (or fallback)
        - Fallback string   — descriptive error if extraction fails

    The caller injects this into the prompt pipeline as-is.
    """
    media_type = _safe(msg.get("media_type", ""))

    if media_type == "image":
        return _process_image(msg, data)

    if media_type == "voice":
        return _process_voice(msg, data)

    # No media — text-only message
    return ""
