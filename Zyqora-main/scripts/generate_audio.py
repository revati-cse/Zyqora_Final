#!/usr/bin/env python3
"""Pre-generate VoiceSort clips OFFLINE with espeak-ng (placeholder quality).

    sudo apt install espeak-ng          # once, before going offline
    python scripts/generate_audio.py    # writes audio/en/*.wav and audio/ta/*.wav
    python scripts/generate_audio.py --force --lang ta

espeak-ng is a local formant synthesizer: no network, no model download. Its
Tamil voice is intelligible but robotic. Recordings by a fluent Tamil speaker
(16-bit PCM WAV, same file names) should replace these clips when available;
existing clips are never overwritten without --force.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave

import _bootstrap  # noqa: F401

from zyqora.config import load_yaml
from zyqora.paths import AUDIO_DIR, DEFAULT_VOICE_CONFIG
from zyqora.voicesort import LANGUAGES, clip_path

ESPEAK_VOICE = {"en": "en-gb", "ta": "ta"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=LANGUAGES, nargs="*", default=list(LANGUAGES))
    ap.add_argument("--speed", type=int, default=140, help="espeak-ng words per minute")
    ap.add_argument("--force", action="store_true", help="overwrite existing clips (e.g. human recordings)")
    args = ap.parse_args()

    exe = shutil.which("espeak-ng")
    if not exe:
        print("espeak-ng not found. Install it (sudo apt install espeak-ng) or record clips manually.", file=sys.stderr)
        return 2
    messages = load_yaml(DEFAULT_VOICE_CONFIG)["messages"]
    for lang in args.lang:
        for key, texts in messages.items():
            out = clip_path(AUDIO_DIR, lang, key)
            if out.exists() and not args.force:
                print(f"keep  {out.relative_to(AUDIO_DIR.parent)} (exists)")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [exe, "-v", ESPEAK_VOICE[lang], "-s", str(args.speed), "-w", str(out), texts[lang]]
            subprocess.run(cmd, check=True, capture_output=True)
            with wave.open(str(out)) as w:
                dur = w.getnframes() / w.getframerate()
            print(f"wrote {out.relative_to(AUDIO_DIR.parent)} ({dur:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
