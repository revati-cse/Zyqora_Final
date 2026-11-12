# audio/

Pre-generated VoiceSort clips: `audio/<en|ta>/<message key lower-case>.wav` (16-bit PCM WAV, mono).

Current clips were generated **offline with espeak-ng** (`python scripts/generate_audio.py`):
they are intelligible placeholders, and their quality has **not** been checked on the Pi speaker.
The Tamil text in `config/voice.yaml` still needs review by a fluent Tamil speaker.
Preferred: replace each file with a recording by a native speaker (same file name, WAV);
`generate_audio.py` never overwrites existing clips unless `--force` is given.
