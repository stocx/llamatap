"""End-to-end workflows, driven the way a user drives them.

These exist because the previous round of tests passed while four defects were
live. Every one of them was missed for the same reason: the test called a
helper directly instead of running the command. `append_record` worked
perfectly when called as the test called it, and raised TypeError every single
time the replay command called it.

So nothing here imports a function and checks its return value. Each test runs
the real entry point — the dispatcher, or the replay script as a subprocess —
against a mock upstream, and then looks at what is left on disk.

Defects covered:

1. P1  `llamatap replay` raised TypeError after every successful replay
       (`append_record` had two parameters called `path`), so the answer was
       never recorded and `--diff` comparisons stopped.
2. P2  `llamatap archive` MOVED the live recording, despite its help saying
       "without clearing".
3. P2  the response head was sent before the cleanup block, so a client that
       disconnected during `end_headers()` leaked the in-flight entry and left
       the upstream response open.
4. P2  `llamatap clear` recreated the recording and log as 0644 under a 022
       umask, and the proxy sets its mode only at creation — so the next whole
       session stayed world-readable.
"""

import http.client
import json
import os
import socket
import subprocess
import threading
import time
import unittest

from _support import CHAT_OK, ROOT, MockUpstream, TapTest, free_port, load, sse


SEED = [{
    "t": "2026-10-02T12:00:00", "path": "/v1/chat/completions", "status": 200,
    "req": {"model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "say hi"}]},
    "resp": "".join(sse("hi", " there")[0].decode() for _ in [0]),
    "meta": {},
}]


def seed(path, recs=SEED):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def records(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cli(*args, env_extra=None, timeout=120):
    env = dict(os.environ)
    env.setdefault("LLAMATAP_UPSTREAM", "http://127.0.0.1:1")
    if env_extra:
        env.update(env_extra)
    return subprocess.run([os.path.join(ROOT, "llamatap"), *args],
                          capture_output=True, text=True, env=env,
                          timeout=timeout)


class ReplayWorkflow(TapTest):
    """Defect 1 — run the command, not the helper."""

    def test_replay_records_its_answer(self):
        seed(self.rec_path)
        reply = json.dumps({"choices": [{"message": {"content": "hello back"},
                                         "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3,
                                      "completion_tokens": 2}}).encode()
        routes = {"/v1/chat/completions": {
            "status": 200, "headers": {"Content-Type": "application/json"},
            "chunks": [reply]}}
        with MockUpstream(routes) as up:
            r = subprocess.run(
                [os.path.join(ROOT, "llamatap-replay.py"), "1",
                 "--file", self.rec_path, "--to", up.url],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "LLAMATAP_UPSTREAM": up.url})

        self.assertEqual(r.returncode, 0,
                         f"replay failed.\nstdout={r.stdout}\nstderr={r.stderr}")
        self.assertNotIn("Traceback", r.stderr, r.stderr)
        recs = records(self.rec_path)
        self.assertEqual(len(recs), 2,
                         "the replay answer was not appended to the recording")
        self.assertEqual(recs[1]["meta"].get("replay_of"), 1)
        self.assertIn("hello back", recs[1]["resp"])

    def test_replayed_answer_reads_back(self):
        """The recorded replay must be parseable afterwards, which is the only
        reason to record it."""
        seed(self.rec_path)
        reply = json.dumps({"choices": [{"message": {"content": "x" * 300000},
                                         "finish_reason": "stop"}]}).encode()
        with MockUpstream({"/v1/chat/completions": {
                "status": 200, "chunks": [reply]}}) as up:
            r = subprocess.run(
                [os.path.join(ROOT, "llamatap-replay.py"), "1",
                 "--file", self.rec_path, "--to", up.url],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "LLAMATAP_UPSTREAM": up.url})
        self.assertEqual(r.returncode, 0, r.stderr)
        recs = records(self.rec_path)
        self.assertEqual(len(recs), 2)
        R = load("replay")
        text, _, _, _ = R.reply_of(recs[1]["resp"])
        self.assertEqual(len(text), 300000, "a long replay answer was truncated")

    def test_diff_shows_the_original(self):
        """F13 through the command: the original side must not be blank."""
        seed(self.rec_path)
        reply = json.dumps({"choices": [{"message": {"content": "new answer"},
                                         "finish_reason": "stop"}]}).encode()
        with MockUpstream({"/v1/chat/completions": {
                "status": 200, "chunks": [reply]}}) as up:
            r = subprocess.run(
                [os.path.join(ROOT, "llamatap-replay.py"), "1", "--diff",
                 "--file", self.rec_path, "--to", up.url],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "LLAMATAP_UPSTREAM": up.url})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("hi", r.stdout,
                      "the original streamed answer rendered as empty")


class ArchiveWorkflow(TapTest):
    """Defect 2 — `archive` must leave the live recording alone."""

    def test_archive_keeps_the_live_recording(self):
        seed(self.rec_path)
        with open(os.environ["LLAMATAP_LOG"], "w") as f:
            f.write("log line\n")

        r = cli("archive")
        self.assertEqual(r.returncode, 0, r.stderr)

        self.assertTrue(os.path.exists(self.rec_path),
                        "`archive` moved the live recording away — its help "
                        "says it archives WITHOUT clearing")
        self.assertGreater(os.path.getsize(self.rec_path), 0,
                           "the live recording was emptied")
        self.assertEqual(len(records(self.rec_path)), 1)

        runs = os.environ["LLAMATAP_RUNS"]
        dirs = os.listdir(runs)
        self.assertEqual(len(dirs), 1, f"expected one archive, got {dirs}")
        copied = os.path.join(runs, dirs[0], os.path.basename(self.rec_path))
        self.assertTrue(os.path.exists(copied), "nothing was archived")
        self.assertEqual(records(copied), records(self.rec_path),
                         "the archive does not match the live recording")

    def test_archived_copy_is_private(self):
        seed(self.rec_path)
        cli("archive")
        runs = os.environ["LLAMATAP_RUNS"]
        d = os.listdir(runs)[0]
        copied = os.path.join(runs, d, os.path.basename(self.rec_path))
        self.assertEqual(os.stat(copied).st_mode & 0o777, 0o600,
                         "an archived recording is more readable than the live one")

    def test_clear_still_moves(self):
        """`clear` is the one that should take the recording away."""
        seed(self.rec_path)
        r = cli("clear")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(os.path.getsize(self.rec_path), 0)
        runs = os.environ["LLAMATAP_RUNS"]
        d = os.listdir(runs)[0]
        kept = os.path.join(runs, d, os.path.basename(self.rec_path))
        self.assertEqual(len(records(kept)), 1)


class ClearPermissionsWorkflow(TapTest):
    """Defect 4 — the files `clear` leaves behind must be private.

    The proxy sets the mode at CREATION, so a 0644 file left by clear stays
    0644 for the whole next session.
    """

    def test_clear_recreates_owner_only(self):
        seed(self.rec_path)
        with open(os.environ["LLAMATAP_LOG"], "w") as f:
            f.write("x\n")
        old = os.umask(0o022)
        try:
            r = cli("clear")
        finally:
            os.umask(old)
        self.assertEqual(r.returncode, 0, r.stderr)
        for f in (self.rec_path, os.environ["LLAMATAP_LOG"]):
            with self.subTest(file=os.path.basename(f)):
                self.assertEqual(os.stat(f).st_mode & 0o777, 0o600,
                                 f"{os.path.basename(f)} is world-readable "
                                 f"after clear")

    def test_a_proxy_write_after_clear_stays_private(self):
        """The end-to-end shape of the defect: clear, then record."""
        seed(self.rec_path)
        old = os.umask(0o022)
        try:
            cli("clear")
        finally:
            os.umask(old)
        P = load("proxy")
        P.record("/v1/chat/completions", 200, b'{"messages":[]}', b"{}", {})
        self.assertEqual(os.stat(self.rec_path).st_mode & 0o777, 0o600,
                         "the session's recording stayed world-readable")


class DisconnectDuringHeaders(TapTest):
    """Defect 3 — a client that leaves before the body starts."""

    def _run(self, disconnect):
        """Drive one streamed request, optionally vanishing in the exact window.

        The window that matters is: the proxy has the request, has told the
        in-flight file about it, and is waiting on upstream headers. The client
        leaving THEN makes the failure land on send_response/end_headers.

        Synchronised on state, not on sleeps: the mock withholds its headers
        until `gate` opens, and the test waits for the in-flight entry to
        appear before disconnecting. An earlier version closed the socket
        immediately, which only proved the proxy handles an unreadable request
        body — a different code path entirely.
        """
        gate = threading.Event()
        routes = {"/v1/chat/completions": {
            "status": 200, "headers": {"Content-Type": "text/event-stream"},
            "chunks": sse(*[f"t{i} " for i in range(20)]),
            "chunked": True, "gate_headers": gate}}
        body = json.dumps({"model": "m", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]})
        with MockUpstream(routes) as up:
            os.environ["LLAMATAP_UPSTREAM"] = up.url
            port = free_port()
            os.environ["LLAMATAP_PORT"] = str(port)
            mod = load("proxy")
            srv = mod.QuietServer(("127.0.0.1", port), mod.Tap)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            inflight_path = os.environ["LLAMATAP_INFLIGHT"]
            rec_path = os.environ["LLAMATAP_REC"]

            def inflight_now():
                try:
                    with open(inflight_path) as f:
                        return json.load(f) or []
                except Exception:
                    return []

            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=10)
                s.sendall(
                    f"POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n{body}".encode())

                # Wait until the proxy has registered it and is blocked on
                # upstream headers. This is the window.
                deadline = time.time() + 20
                while time.time() < deadline and not inflight_now():
                    time.sleep(0.01)
                self.assertTrue(inflight_now(),
                                "the proxy never registered the request; the "
                                "test is not exercising the intended window")

                drained = []
                if disconnect:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                 b"\x01\x00\x00\x00\x00\x00\x00\x00")
                    s.close()
                else:
                    def drain():
                        try:
                            while True:
                                b = s.recv(65536)
                                if not b:
                                    break
                                drained.append(b)
                        except OSError:
                            pass
                    threading.Thread(target=drain, daemon=True).start()

                gate.set()      # upstream now sends headers and the stream

                while time.time() < deadline and not records(rec_path):
                    time.sleep(0.02)
                inflight = inflight_now()
                while time.time() < deadline and inflight:
                    time.sleep(0.02)
                    inflight = inflight_now()
                if not disconnect:
                    s.close()
                return inflight, records(rec_path)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_client_gone_before_headers_is_cleaned_up(self):
        inflight, recs = self._run(disconnect=True)
        self.assertEqual(inflight, [],
                         "the request is still in flight; the console would "
                         "show it running forever")
        self.assertEqual(len(recs), 1,
                         "the exchange was not recorded at all")

    def test_a_normal_client_is_unaffected(self):
        inflight, recs = self._run(disconnect=False)
        self.assertEqual(inflight, [])
        self.assertEqual(len(recs), 1)


if __name__ == "__main__":
    unittest.main()


class ClearEmptyFilePermissions(TapTest):
    """`clear` must make an EXISTING empty recording private too.

    The archive helper skips zero-byte files, so an empty recording is never
    moved aside — `clear` truncates it in place. `umask` only governs file
    CREATION, so a 0644 empty recording stayed 0644, and every capture of the
    next session with it. The previous permission test started from a
    non-empty file, which takes the archive path and therefore gets a freshly
    created, private file by accident.
    """

    def modes(self):
        return {os.path.basename(f): os.stat(f).st_mode & 0o777
                for f in (self.rec_path, os.environ["LLAMATAP_LOG"])}

    def clear_under_umask(self):
        old = os.umask(0o022)
        try:
            return cli("clear")
        finally:
            os.umask(old)

    def test_existing_empty_files_become_private(self):
        for f in (self.rec_path, os.environ["LLAMATAP_LOG"]):
            open(f, "w").close()
            os.chmod(f, 0o644)
        r = self.clear_under_umask()
        self.assertEqual(r.returncode, 0, r.stderr)
        for name, mode in self.modes().items():
            with self.subTest(file=name):
                self.assertEqual(mode, 0o600,
                                 f"{name} stayed world-readable after clear")

    def test_a_mix_of_empty_and_nonempty(self):
        seed(self.rec_path)                       # non-empty: archived away
        open(os.environ["LLAMATAP_LOG"], "w").close()   # empty: skipped
        for f in (self.rec_path, os.environ["LLAMATAP_LOG"]):
            os.chmod(f, 0o644)
        r = self.clear_under_umask()
        self.assertEqual(r.returncode, 0, r.stderr)
        for name, mode in self.modes().items():
            with self.subTest(file=name):
                self.assertEqual(mode, 0o600, f"{name} is world-readable")

    def test_the_whole_next_session_stays_private(self):
        """The shape of the defect: clear, then let the proxy write."""
        for f in (self.rec_path, os.environ["LLAMATAP_LOG"]):
            open(f, "w").close()
            os.chmod(f, 0o644)
        self.clear_under_umask()
        P = load("proxy")
        P.record("/v1/chat/completions", 200, b'{"messages":[]}', b"{}", {})
        self.assertEqual(os.stat(self.rec_path).st_mode & 0o777, 0o600,
                         "the session's captures are world-readable")

    def test_a_writer_tightens_a_loose_file_it_did_not_create(self):
        """A recording at a custom path, outside the 0700 state directory,
        may already exist with loose permissions. O_CREAT's mode applies only
        at creation, so the writer has to fix it."""
        open(self.rec_path, "w").close()
        os.chmod(self.rec_path, 0o644)
        P = load("proxy")
        P.record("/v1/chat/completions", 200, b'{"messages":[]}', b"{}", {})
        self.assertEqual(os.stat(self.rec_path).st_mode & 0o777, 0o600)
