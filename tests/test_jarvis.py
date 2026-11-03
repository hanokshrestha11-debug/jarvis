import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import jarvis


class EntrypointTests(unittest.TestCase):
    def test_unsupported_python_retries_with_python314(self):
        argv = ["jarvis.py", "--example"]
        with patch.object(jarvis.sys, "version_info", (3, 15)), patch.object(
            jarvis.sys, "argv", argv
        ), patch.object(
            jarvis.shutil, "which", return_value="C:\\Python\\py.exe"
        ), patch.object(
            jarvis.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            self.assertEqual(jarvis.main(), 0)

        run.assert_called_once_with(
            ["C:\\Python\\py.exe", "-3.14", *argv],
            check=False,
        )

    def test_double_clap_opens_song_and_claude(self):
        with patch.object(jarvis, "play_song") as play_song, patch.object(
            jarvis, "open_claude_in_default_browser"
        ) as open_claude:
            jarvis.run_double_clap_actions()

        play_song.assert_called_once_with(jarvis.SONG_URI)
        open_claude.assert_called_once_with()

    def test_main_waits_for_clap_then_starts_voice_assistant(self):
        with patch.object(jarvis.sys, "version_info", (3, 14)), patch.object(
            jarvis, "_wait_for_double_clap", return_value=(0, 2)
        ) as wait_for_clap, patch(
            "levi_voice.run_voice_assistant", return_value=0
        ) as run_voice:
            self.assertEqual(jarvis.main(), 0)

        wait_for_clap.assert_called_once_with()
        run_voice.assert_called_once_with(
            input_device=2,
            welcome_enabled=jarvis.JARVIS_WELCOME_ENABLED,
            welcome_phrase=jarvis.JARVIS_WELCOME_PHRASE,
        )

    def test_main_does_not_start_voice_assistant_if_stopped_before_clap(self):
        with patch.object(jarvis.sys, "version_info", (3, 14)), patch.object(
            jarvis, "_wait_for_double_clap", return_value=(130, None)
        ), patch("levi_voice.run_voice_assistant") as run_voice:
            self.assertEqual(jarvis.main(), 0)

        run_voice.assert_not_called()

    def test_detected_double_clap_runs_song_and_browser_actions(self):
        frame_size = jarvis.block_samples()

        class FakeStream:
            def __init__(self, **_kwargs):
                clap = np.full((frame_size, 1), 0.05, dtype=np.float32)
                quiet = np.zeros((frame_size, 1), dtype=np.float32)
                self.frames = iter([(clap, False), (quiet, False), (clap, False)])

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _blocksize):
                return next(self.frames)

        with patch.object(jarvis, "_choose_input_device", return_value=2), patch.object(
            jarvis.sd, "InputStream", FakeStream
        ), patch.object(
            jarvis.time, "monotonic", side_effect=[10.0, 10.1, 10.2]
        ), patch.object(jarvis, "run_double_clap_actions") as actions:
            status, input_device = jarvis._wait_for_double_clap()

        self.assertEqual((status, input_device), (0, 2))
        actions.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
