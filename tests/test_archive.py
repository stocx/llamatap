"""Archiving a recording, which must never lose one.

Regressions covered:

F3  `llamatap clear` was a shell `: > file` with no archive at all, while its
    own help text and the README both promised one. Following the documented
    workflow destroyed the only copy.

F4  Destinations were stamped to the minute and created with `exist_ok=True`,
    then files moved in with `os.replace` — so two archives within the same
    minute silently overwrote each other.
"""

import json
import os
import subprocess
import unittest

from _support import ROOT, TapTest, load, write_records


REC_A = [{"t": "A", "path": "/v1/chat/completions", "status": 200,
          "req": {"messages": [{"role": "user", "content": "first run"}]},
          "resp": "{}", "meta": {}}]
REC_B = [{"t": "B", "path": "/v1/chat/completions", "status": 200,
          "req": {"messages": [{"role": "user", "content": "second run"}]},
          "resp": "{}", "meta": {}}]


class ArchiveHelper(TapTest):

    def setUp(self):
        super().setUp()
        self.A = load("archive")
        self.runs = os.environ["LLAMATAP_RUNS"]

    def test_moves_the_recording_aside(self):
        write_records(self.rec_path, REC_A)
        dest = self.A.archive([self.rec_path], self.runs)
        self.assertTrue(os.path.isdir(dest))
        moved = os.path.join(dest, os.path.basename(self.rec_path))
        self.assertTrue(os.path.exists(moved))
        self.assertFalse(os.path.exists(self.rec_path),
                         "the original should have been moved, not copied")
        with open(moved) as f:
            self.assertIn("first run", f.read())

    def test_two_archives_in_the_same_moment_both_survive(self):
        """F4 — the data-loss case. Same stamp, forced."""
        write_records(self.rec_path, REC_A)
        first = self.A.archive([self.rec_path], self.runs, stamp="fixed")
        write_records(self.rec_path, REC_B)
        second = self.A.archive([self.rec_path], self.runs, stamp="fixed")

        self.assertNotEqual(first, second,
                            "the second archive reused the first's directory")
        for d, phrase in ((first, "first run"), (second, "second run")):
            path = os.path.join(d, os.path.basename(self.rec_path))
            self.assertTrue(os.path.exists(path), f"{d} lost its recording")
            with open(path) as f:
                self.assertIn(phrase, f.read(),
                              f"{d} does not hold the run it archived")

    def test_nothing_to_archive_is_not_an_error(self):
        self.assertIsNone(self.A.archive([self.rec_path], self.runs))
        write_records(self.rec_path, [])
        self.assertIsNone(self.A.archive([self.rec_path], self.runs),
                          "an empty recording is not worth an archive")

    def test_archive_directory_is_private(self):
        write_records(self.rec_path, REC_A)
        dest = self.A.archive([self.rec_path], self.runs)
        self.assertEqual(os.stat(dest).st_mode & 0o777, 0o700,
                         "archives hold prompts and source; owner-only")

    def test_failure_raises_so_callers_do_not_truncate(self):
        write_records(self.rec_path, REC_A)
        with self.assertRaises(Exception):
            self.A.archive([self.rec_path], "/proc/cannot/create/here")
        self.assertTrue(os.path.exists(self.rec_path),
                        "the recording must survive a failed archive")


class CliClear(TapTest):
    """F3 — end to end through the dispatcher, which is what users run."""

    def run_cli(self, *args):
        env = dict(os.environ)
        env["LLAMATAP_UPSTREAM"] = "http://127.0.0.1:1"
        return subprocess.run([os.path.join(ROOT, "llamatap"), *args],
                              capture_output=True, text=True, env=env, timeout=60)

    def test_clear_archives_before_truncating(self):
        write_records(self.rec_path, REC_A)
        with open(os.environ["LLAMATAP_LOG"], "w") as f:
            f.write("a log line\n")

        r = self.run_cli("clear")
        self.assertEqual(r.returncode, 0, r.stderr)

        runs = os.environ["LLAMATAP_RUNS"]
        archives = os.listdir(runs) if os.path.isdir(runs) else []
        self.assertTrue(archives,
                        f"`llamatap clear` destroyed the recording without "
                        f"archiving it. stdout={r.stdout!r} stderr={r.stderr!r}")
        kept = os.path.join(runs, archives[0], os.path.basename(self.rec_path))
        self.assertTrue(os.path.exists(kept))
        with open(kept) as f:
            self.assertIn("first run", f.read())

    def test_clear_leaves_an_empty_recording_behind(self):
        write_records(self.rec_path, REC_A)
        self.run_cli("clear")
        self.assertEqual(os.path.getsize(self.rec_path), 0,
                         "clear should leave a fresh, empty recording")

    def test_clear_twice_in_one_second_keeps_both(self):
        write_records(self.rec_path, REC_A)
        self.run_cli("clear")
        write_records(self.rec_path, REC_B)
        self.run_cli("clear")
        runs = os.environ["LLAMATAP_RUNS"]
        found = []
        for d in os.listdir(runs):
            p = os.path.join(runs, d, os.path.basename(self.rec_path))
            if os.path.exists(p):
                with open(p) as f:
                    found.append(f.read())
        self.assertEqual(len(found), 2, f"expected two archives, got {found}")
        self.assertTrue(any("first run" in x for x in found), "first run lost")
        self.assertTrue(any("second run" in x for x in found), "second run lost")

    def test_clear_with_nothing_recorded_is_harmless(self):
        r = self.run_cli("clear")
        self.assertEqual(r.returncode, 0, r.stderr)


class ConsoleClear(TapTest):
    """F4 again, in the console's own path."""

    def test_console_clear_uses_the_shared_archiver(self):
        C = load("console")
        self.assertTrue(hasattr(C, "A") or hasattr(C, "archive_mod")
                        or "llamatap-archive" in open(
                            os.path.join(ROOT, "llamatap-console.py")).read(),
                        "the console still has its own archive implementation")


if __name__ == "__main__":
    unittest.main()
