"""
Speech-to-text via Whisper, used to obtain a transcript when the manifest's
`transcript` column is empty (e.g. at inference time on a brand-new clip).
"""

import whisper

from config import WHISPER_MODEL_SIZE

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _model


def transcribe(wav_path: str) -> str:
    model = _get_model()
    result = model.transcribe(wav_path, fp16=False)
    return result["text"].strip()
