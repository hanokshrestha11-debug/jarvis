import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
