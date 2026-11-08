"""Fixed-window diagnostic preprocessing, not production voice activity detection."""
from pathlib import Path
import hashlib
import wave

import numpy as np

RATE = 16_000
WINDOW = 160_000
MASK_FRAMES = 589
MODES = ("uniform", "valid", "activity")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_wav(path):
    with wave.open(str(path)) as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (RATE, 1, 2, "NONE"):
            raise ValueError("Expected uncompressed mono PCM16 at 16 kHz; convert explicitly first")
        audio = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").astype(np.float32) / 32768
    validate(audio)
    return audio


def validate(audio):
    if audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be a nonempty finite mono vector")


def activity(audio):
    """20 ms RMS, 10 ms hop, +/-50 ms hangover. Energy proxy, NOT a speech/identity label."""
    validate(audio)
    hop, width = 160, 320
    starts = np.arange(0, len(audio), hop)
    rms = np.asarray([np.sqrt(np.mean(audio[s:s + width] ** 2)) for s in starts])
    threshold = max(1e-4, float(rms.max()) * 0.08)
    active = rms > threshold
    expanded = active.copy()
    for shift in range(1, 6):
        expanded[shift:] |= active[:-shift]
        expanded[:-shift] |= active[shift:]
    return np.repeat(expanded, hop)[:len(audio)]


def crop(audio, seconds, alignment):
    """Crop by file start or energy onset; onset is computed without speaker labels."""
    if not np.isfinite(seconds) or not 0 < seconds <= 10:
        raise ValueError("Crop duration must be in (0, 10] seconds")
    start = 0
    if alignment == "onset":
        indices = np.flatnonzero(activity(audio))
        start = int(indices[0]) if len(indices) else 0
    elif alignment != "prefix":
        raise ValueError("Unknown crop alignment")
    end = min(len(audio), start + int(round(seconds * RATE)))
    return audio[start:end], start


def tensors(audio, mode):
    """589-point time mask over the full 10 s axis, as consumed by the pinned graph.

    The graph nearest-resizes this axis for weighted pooling. Valid/audio masks exclude
    padding from weights, NOT convolutional receptive fields or feature normalization.
    """
    validate(audio)
    if mode not in MODES:
        raise ValueError("Unknown mask mode")
    pcm = audio[:WINDOW]
    waveform = np.zeros((1, 1, WINDOW), np.float32)
    waveform[0, 0, :len(pcm)] = pcm
    indices = np.floor(np.arange(MASK_FRAMES) * WINDOW / MASK_FRAMES).astype(int)
    valid = indices < len(pcm)
    if mode == "uniform":
        mask = np.ones(MASK_FRAMES, np.float32)
    else:
        mask = valid.astype(np.float32)
        if mode == "activity":
            active = activity(pcm)
            mask[valid] *= active[indices[valid]]
    return {"waveform": waveform, "weights": mask.reshape(1, -1)}


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vector)
    if len(vector) != 256 or not np.isfinite(vector).all() or norm < 1e-12:
        raise ValueError("Expected a finite, nonzero 256-d embedding")
    return vector / norm
