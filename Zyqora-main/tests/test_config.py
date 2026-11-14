import re
import tempfile
import unittest
from pathlib import Path

from tests import helpers  # noqa: F401
from zyqora.config import ConfigError, load_app_config
from zyqora.imagenet_labels import LabelsError, load_labels
from zyqora.paths import AUDIO_DIR, DEFAULT_APP_CONFIG, DEFAULT_VOICE_CONFIG
from zyqora.voicesort import LANGUAGES, REQUIRED_SYSTEM_KEYS, VoiceSort, clip_path


class ConfigTests(unittest.TestCase):
    def test_shipped_app_config_is_valid(self):
        app = load_app_config(DEFAULT_APP_CONFIG)
        self.assertEqual(app.ecoguard.threshold, 0.40)
        self.assertFalse(app.ecoguard.calibrated)  # must not claim calibration without a sweep
        self.assertGreater(app.points_correct, 0)
        self.assertEqual(app.points_incorrect, 0)

    def test_bundled_labels(self):
        labels = load_labels()
        self.assertEqual(len(labels), 1000)
        self.assertEqual((labels[0], labels[898], labels[954]), ("tench", "water bottle", "banana"))

    def test_bad_configs_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "app.yaml"
            base = DEFAULT_APP_CONFIG.read_text()

            def setting(key: str, value: str) -> str:
                # Replace whatever value the local config has (e.g. num_threads tuned on the Pi).
                new, n = re.subn(rf"(?m)^(\s*{key}:)[^#\n]*", rf"\g<1> {value}", base, count=1)
                self.assertEqual(n, 1, f"{key} not found in app.yaml")
                return new

            for bad in (setting("threshold", "1.7"),
                        setting("threshold", "high"),
                        setting("num_threads", "0"),
                        setting("points_verified_correct", "-10"),
                        "model: [unclosed", "just a string"):
                p.write_text(bad)
                with self.assertRaises(ConfigError, msg=bad[:40]):
                    load_app_config(p)
            with self.assertRaises(ConfigError):
                load_app_config(Path(t) / "missing.yaml")
            lp = Path(t) / "labels.json"
            lp.write_text('["a", "b"]')
            with self.assertRaises(LabelsError):
                load_labels(lp)

    def test_voice_messages_and_clips_cover_all_actions(self):
        vs = VoiceSort("ta", echo=False)
        for key in (*REQUIRED_SYSTEM_KEYS, "RECYCLE", "COMPOST", "RESIDUAL"):
            for lang in LANGUAGES:
                self.assertTrue(VoiceSort(lang, echo=False).text_for(key))
                self.assertTrue(clip_path(AUDIO_DIR, lang, key).is_file(), f"{lang}/{key}")
        self.assertNotEqual(vs.text_for("RECYCLE"), VoiceSort("en", echo=False).text_for("RECYCLE"))

    def test_voice_failures_never_raise(self):
        vs = VoiceSort("en", DEFAULT_VOICE_CONFIG, audio_dir=Path("/nonexistent"), echo=False)
        a = vs.announce("RECYCLE")
        self.assertFalse(a.played)
        self.assertIn("clip missing", a.detail)
        off = VoiceSort("off", echo=False).announce("RECYCLE")
        self.assertFalse(off.played)
        silent = VoiceSort("en", player="none", echo=False).announce("COMPOST")
        self.assertFalse(silent.played)
        with self.assertRaises(ValueError):
            VoiceSort("fr")


if __name__ == "__main__":
    unittest.main()
