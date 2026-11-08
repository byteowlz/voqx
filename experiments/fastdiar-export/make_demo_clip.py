"""Build a turn-taking demo clip from fictional synthetic voices (stdlib only).

    uv run --no-project python make_demo_clip.py --bank <voice-bank-dir> --out <new-dir>

The bank is a waqy/spqx VoiceDesign output folder with clips.jsonl, voices.json and wav/.
Writes demo.wav (16 kHz mono PCM16) and demo.json with the exact turn script. No real
person's voice is used; the script is ground truth only for this synthetic arrangement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

VOICES = [("de00", "Voice A"), ("de01", "Voice B"), ("de02", "Voice C")]
# (voice index, sentences in the turn). Long turns first so references fit; short turns later.
PLAN = [(0, 3), (1, 3), (2, 3), (0, 2), (1, 2), (2, 2), (0, 1), (1, 1), (2, 1), (0, 2), (2, 2), (1, 2)]
RATE, WITHIN, BETWEEN = 16000, 0.25, 0.7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=False, exist_ok=False)
    rows = [json.loads(line) for line in (args.bank / "clips.jsonl").read_text().splitlines() if line.strip()]
    meta = {v["voice"]: v for v in json.loads((args.bank / "voices.json").read_text())}
    pools = {vid: [r for r in rows if r["voice"] == vid and "-ctx-" in r["id"]] for vid, _ in VOICES}
    used = {vid: 0 for vid, _ in VOICES}
    pcm, turns, cursor = bytearray(), [], 0

    def silence(seconds: float) -> None:
        nonlocal cursor
        n = round(seconds * RATE)
        pcm.extend(b"\0\0" * n)
        cursor += n

    for index, count in PLAN:
        vid = VOICES[index][0]
        start, sentences = cursor, []
        for i in range(count):
            row = pools[vid][used[vid]]
            used[vid] += 1
            with wave.open(str(args.bank / "wav" / f"{row['id']}.wav"), "rb") as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE), row["id"]
                frames = w.readframes(w.getnframes())
            pcm.extend(frames)
            cursor += len(frames) // 2
            sentences.append({"id": row["id"], "text": row["text"]})
            if i < count - 1:
                silence(WITHIN)
        turns.append({"voice": vid, "start": start / RATE, "end": cursor / RATE, "sentences": sentences})
        silence(BETWEEN)

    with wave.open(str(args.out / "demo.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(pcm))
    script = {
        "format": "voqx-demo-turns/1",
        "note": "Synthetic arrangement of fictional VoiceDesign voices; no real person.",
        "seconds": cursor / RATE,
        "wav_sha256": hashlib.sha256((args.out / "demo.wav").read_bytes()).hexdigest(),
        "voices": [{"id": vid, "label": label, "description": meta[vid]["instruct"]} for vid, label in VOICES],
        "turns": turns,
    }
    (args.out / "demo.json").write_text(json.dumps(script, ensure_ascii=False, indent=1))
    print(json.dumps({"seconds": round(script["seconds"], 2), "turns": len(turns), "sha256": script["wav_sha256"]}))


if __name__ == "__main__":
    main()
