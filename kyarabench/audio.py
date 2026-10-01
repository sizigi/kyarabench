"""The reference loader."""
import numpy as np
import soundfile as sf

SR = 16000
MAX_SECONDS = 10.0


def load_clip(path, sr=SR, max_seconds=MAX_SECONDS):
    """Read one restored clip as a mono float32 waveform at 16 kHz, exactly 10 s long.

    Longer clips are truncated and shorter ones are zero-padded, so every clip has the same
    length and a batch needs no padding of its own.

    Returns the waveform and how many of its samples are real audio, for models that accept
    a length or an attention mask.
    """
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    length = int(round(max_seconds * sr))
    out = np.zeros(length, dtype="float32")
    wav = wav[:length]
    out[:len(wav)] = wav
    return out, len(wav)


def clip_path(root, uuid):
    """Where a restored clip lives: <root>/<uuid[:2]>/<uuid>_restored.wav."""
    return f"{root}/{uuid[:2]}/{uuid}_restored.wav"
