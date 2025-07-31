
from faster_whisper import WhisperModel

model = WhisperModel("small", compute_type="int8")

def transcribe_audio_whisper(audio_path: str):
    segments, info = model.transcribe(audio_path, beam_size=5, task="translate")
    full_text = ""
    for segment in segments:
        full_text += segment.text.strip() + " "
    return info.language, full_text.strip()
