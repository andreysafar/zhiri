"""Text-to-speech via gTTS. Outputs OGG Opus bytes for Telegram voice messages."""
import io
import logging

from gtts import gTTS

logger = logging.getLogger(__name__)

# Telegram voice notes must be OGG Opus.
# pydub + ffmpeg handle the MP3→OGG conversion.
# If ffmpeg is not installed, falls back to sending raw MP3 as audio file.
try:
    from pydub import AudioSegment
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False


def synthesize(text: str) -> tuple[io.BytesIO, str]:
    """
    Returns (audio_buffer, mime_type).
    mime_type is 'audio/ogg' when ffmpeg is available, else 'audio/mpeg'.
    """
    # Trim very long texts — TTS of a full speech takes forever
    if len(text) > 900:
        text = text[:900] + "..."

    tts = gTTS(text=text, lang="ru", slow=False)

    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    if FFMPEG_AVAILABLE:
        try:
            audio = AudioSegment.from_mp3(mp3_buf)
            ogg_buf = io.BytesIO()
            audio.export(ogg_buf, format="ogg", codec="libopus")
            ogg_buf.seek(0)
            return ogg_buf, "audio/ogg"
        except Exception as e:
            logger.warning("OGG conversion failed (%s), falling back to MP3", e)
            mp3_buf.seek(0)

    return mp3_buf, "audio/mpeg"
