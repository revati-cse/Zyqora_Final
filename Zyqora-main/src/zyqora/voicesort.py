"""VoiceSort: offline Tamil / English instructions from pre-generated audio clips.

* No online TTS, no model downloads, no network.
* Clips: audio/<lang>/<key lower-case>.wav (16-bit PCM WAV plays with ALSA ``aplay``
  on Raspberry Pi OS).
* If a clip or audio device is missing, the instruction is printed instead and
  classification continues; playback errors never abort the pipeline.

CLI:
    python -m zyqora.voicesort --lang ta --say RECYCLE
    python -m zyqora.voicesort --lang en --list
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import ConfigError, load_yaml
from .paths import AUDIO_DIR, DEFAULT_VOICE_CONFIG

LANGUAGES = ("en", "ta")
REQUIRED_SYSTEM_KEYS = ("MANUAL_REVIEW", "VERIFIED_CORRECT", "VERIFIED_INCORRECT")
PLAYER_ARGS = {"aplay": ["-q"], "paplay": [], "pw-play": []}


def clip_path(audio_dir: Path, lang: str, key: str) -> Path:
    return Path(audio_dir) / lang / f"{key.lower()}.wav"


@dataclass(frozen=True)
class Announcement:
    key: str
    language: str
    text: str
    played: bool
    detail: str


class VoiceSort:
    def __init__(
        self,
        language: str,
        config_path: Path = DEFAULT_VOICE_CONFIG,
        audio_dir: Path = AUDIO_DIR,
        player: Optional[str] = None,
        echo: bool = True,
    ):
        if language not in (*LANGUAGES, "off"):
            raise ValueError(f"language must be one of {LANGUAGES} or 'off', got {language!r}")
        self.language = language
        self.audio_dir = Path(audio_dir)
        self.echo = echo
        cfg = load_yaml(config_path)
        self.messages: dict[str, dict[str, str]] = cfg.get("messages") or {}
        if not self.messages:
            raise ConfigError(f"{config_path}: 'messages' is empty")
        self.timeout = float(cfg.get("playback_timeout_s", 15))
        self.player = self._choose_player(player or cfg.get("player", "auto"))

    @staticmethod
    def _choose_player(preference: str) -> Optional[str]:
        if preference == "none":
            return None
        names = list(PLAYER_ARGS) if preference == "auto" else [preference]
        for name in names:
            if name not in PLAYER_ARGS:
                raise ValueError(f"unsupported audio player {name!r}; use one of {list(PLAYER_ARGS)} or auto/none")
            if shutil.which(name):
                return name
        return None

    @property
    def enabled(self) -> bool:
        return self.language != "off"

    def text_for(self, key: str) -> str:
        entry = self.messages.get(key)
        if not entry:
            raise KeyError(f"no VoiceSort message configured for '{key}'")
        lang = self.language if self.language in LANGUAGES else "en"
        return entry.get(lang) or entry.get("en", key)

    def announce(self, key: str) -> Announcement:
        if not self.enabled:
            return Announcement(key, "off", "", False, "voice disabled")
        text = self.text_for(key)
        clip = clip_path(self.audio_dir, self.language, key)
        if self.echo:
            print(f"  [VoiceSort:{self.language}] {text}")
        if not clip.is_file():
            return Announcement(key, self.language, text, False, f"clip missing: {clip}")
        if self.player is None:
            return Announcement(key, self.language, text, False, "no audio player found (aplay/paplay/pw-play)")
        cmd = [shutil.which(self.player) or self.player, *PLAYER_ARGS[self.player], str(clip)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            return Announcement(key, self.language, text, False, f"playback timed out after {self.timeout}s")
        except OSError as exc:
            return Announcement(key, self.language, text, False, f"playback failed: {exc}")
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").strip().splitlines()
            return Announcement(key, self.language, text, False,
                                f"{self.player} exit {proc.returncode}: {err[-1] if err else 'no audio device?'}")
        return Announcement(key, self.language, text, True, f"played {clip.name} via {self.player}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m zyqora.voicesort", description="Play a VoiceSort instruction offline.")
    ap.add_argument("--lang", choices=LANGUAGES, default="en")
    ap.add_argument("--say", help="message key, e.g. RECYCLE, COMPOST, RESIDUAL, MANUAL_REVIEW, VERIFIED_CORRECT")
    ap.add_argument("--list", action="store_true", help="list message keys and clip status")
    args = ap.parse_args(argv)
    vs = VoiceSort(args.lang, echo=False)
    if args.list or not args.say:
        for key in vs.messages:
            clip = clip_path(vs.audio_dir, args.lang, key)
            print(f"{key:20s} {'clip OK ' if clip.is_file() else 'NO CLIP '} {vs.text_for(key)}")
        print(f"player: {vs.player or 'none found'}")
        return 0
    try:
        a = vs.announce(args.say)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"[{a.language}] {a.text}\n -> {a.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
