"""Shared test scaffolding.

Three things every test here needs:

1. A loader. The modules are executable scripts with hyphens in their names,
   so they cannot be imported normally.
2. A curses stub, so `llamatap-console.py` imports without a terminal.
3. Isolation. Every module reads its paths from the environment AT IMPORT
   TIME, so a test that wants its own recording must set the environment and
   then load a FRESH copy of the module. `tap_env()` and `load()` together do
   that; no test may use the real recording.
"""

import contextlib
import http.server
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import types
import unittest

# LLAMATAP_TEST_ROOT lets the same tests run against a DIFFERENT copy of the
# tree — which is how a regression test is shown failing against the commit
# that had the bug, without touching the working tree.
ROOT = os.environ.get("LLAMATAP_TEST_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _curses_stub():
    """Enough of curses for import and for calling render functions headlessly.

    Only constants and colour lookups are needed: the drawing calls take a
    screen object, which tests supply themselves.
    """
    ns = types.SimpleNamespace(
        color_pair=lambda n: n << 8,
        pair_number=lambda a: (a >> 8) & 0xff,
        A_COLOR=0xff00,
        A_BOLD=0, A_REVERSE=0, A_UNDERLINE=0, A_DIM=0, A_NORMAL=0,
        # Narrow on purpose. It used to be `Exception`, so the `except
        # curses.error` guards around every draw swallowed real bugs —
        # an AttributeError inside an attribute helper made the whole
        # statistics panel disappear and no test said a word.
        error=type("error", (Exception,), {}),
        ALL_MOUSE_EVENTS=0, REPORT_MOUSE_POSITION=0,
        KEY_UP=259, KEY_DOWN=258, KEY_PPAGE=339, KEY_NPAGE=338,
        KEY_HOME=262, KEY_END=360, KEY_ENTER=343, KEY_BACKSPACE=263,
        KEY_MOUSE=409, KEY_RESIZE=410,
        BUTTON1_PRESSED=4, BUTTON4_PRESSED=65536, BUTTON5_PRESSED=2097152,
        COLOR_BLACK=0, COLOR_RED=1, COLOR_GREEN=2, COLOR_YELLOW=3,
        COLOR_BLUE=4, COLOR_MAGENTA=5, COLOR_CYAN=6, COLOR_WHITE=7,
        # draw() calls these; without them a test that drives the real draw()
        # dies in the stub rather than telling you anything about the code.
        curs_set=lambda n: 0,
        napms=lambda ms: None,
        beep=lambda: None,
        flash=lambda: None,
        doupdate=lambda: None,
    )
    return ns


def load(stem, fresh=True):
    """Load `llamatap-<stem>.py` as a module.

    `fresh=True` (the default) loads a new copy every call, which is the point:
    module-level constants capture the environment, so a test that changed the
    environment needs a module that has not already read it.
    """
    path = os.path.join(ROOT, f"llamatap-{stem}.py") if stem != "cli" \
        else os.path.join(ROOT, "llamatap")
    name = f"_llamatap_{stem}_{id(path)}_{len(sys.modules)}" if fresh else f"_llamatap_{stem}"
    if not fresh and name in sys.modules:
        return sys.modules[name]
    sys.modules.setdefault("curses", _curses_stub())
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # argv belongs to the test runner; these scripts parse it when run as main
    # and some read it at import to decide on archiving.
    old_argv, sys.argv = sys.argv, [path]
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


@contextlib.contextmanager
def tap_env(**extra):
    """A private state directory, exported the way the dispatcher exports it.

    Yields the directory. Everything a module might write — recording, log,
    in-flight file, archives — lands inside it and is removed afterwards.
    """
    keys = ("LLAMATAP_STATE", "LLAMATAP_REC", "LLAMATAP_LOG",
            "LLAMATAP_INFLIGHT", "LLAMATAP_RUNS", "LLAMATAP_UPSTREAM",
            "LLAMATAP_PORT", "LLAMATAP_CONFIG", "LLAMATAP_SSH_HOST",
            "LLAMATAP_SWAP_CONFIG", "LLAMATAP_KEEP", "LLAMATAP_CAP")
    saved = {k: os.environ.get(k) for k in keys}
    saved.update({k: os.environ.get(k) for k in extra})
    # mkdtemp + a tolerant rmtree, not TemporaryDirectory. The proxy's handler
    # threads are daemons: one can still be publishing the in-flight file as
    # the directory is being removed, and TemporaryDirectory then raises
    # "Directory not empty" and fails the test that had already passed. A
    # leftover temp file is not a defect in llamatap.
    d = tempfile.mkdtemp(prefix="llamatap-test-")
    try:
        env = {
            "LLAMATAP_STATE": d,
            "LLAMATAP_REC": os.path.join(d, "llamatap-records.jsonl"),
            "LLAMATAP_LOG": os.path.join(d, "llamatap.log"),
            "LLAMATAP_INFLIGHT": os.path.join(d, "llamatap-inflight.json"),
            "LLAMATAP_RUNS": os.path.join(d, "runs"),
            # Point at a closed port by default: no test may reach the real
            # server by accident.
            "LLAMATAP_UPSTREAM": "http://127.0.0.1:1",
        }
        env.update({k: str(v) for k, v in extra.items()})
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(env)
        yield d
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockUpstream:
    """A stand-in model server.

    Scripted per path. A route is a dict:

        status   int
        headers  {name: value}
        chunks   [bytes, ...]        written one at a time, each flushed
        gate     threading.Event     if set on a route, the server waits for it
                                     before writing anything after `chunks[0]`

    `gate` is how streaming is tested without sleeping: the test asserts the
    client already holds the first chunk while the server is still waiting.
    """

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.requests = []          # [(method, path, headers, body)]
        self._lock = threading.Lock()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    return self.rfile.read(n)
                if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
                    out = b""
                    while True:
                        line = self.rfile.readline().strip()
                        if not line:
                            break
                        size = int(line.split(b";")[0], 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        out += self.rfile.read(size)
                        self.rfile.readline()
                    return out
                return b""

            def _serve(self, method):
                body = self._body()
                with outer._lock:
                    outer.requests.append(
                        (method, self.path, dict(self.headers), body))
                # Unknown paths 404. A mock that answers everything with a
                # cheerful 200 hides exactly the bug being tested: props()
                # trying the wrong route and appearing to succeed.
                route = outer.routes.get(self.path) or outer.routes.get("*")
                # gate_headers holds everything, including the status line, so
                # a test can act while the client is blocked waiting for them.
                if route and route.get("gate_headers") is not None:
                    route["gate_headers"].wait(20)
                if route is None:
                    msg = b'{"error":"no route in mock: ' + self.path.encode() + b'"}'
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    self.wfile.flush()
                    return
                chunks = list(route.get("chunks") or [b""])
                total = sum(len(c) for c in chunks)
                self.send_response(route.get("status", 200))
                for k, v in (route.get("headers") or {}).items():
                    self.send_header(k, v)
                if route.get("chunked"):
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Content-Length", str(total))
                self.end_headers()
                gate = route.get("gate")
                for i, c in enumerate(chunks):
                    if i == 1 and gate is not None:
                        gate.wait(10)
                    _ = c
                    if route.get("chunked"):
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(c), c))
                    else:
                        self.wfile.write(c)
                    self.wfile.flush()
                if route.get("chunked"):
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()

            def do_POST(self):
                self._serve("POST")

            def do_GET(self):
                self._serve("GET")

        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._srv.shutdown()
        self._srv.server_close()


#: a plain, non-streamed chat reply, for routes that just need to answer
CHAT_OK = {"status": 200,
           "headers": {"Content-Type": "application/json"},
           "chunks": [b'{"choices":[{"message":{"content":"ok"},'
                      b'"finish_reason":"stop"}],'
                      b'"usage":{"prompt_tokens":5,"completion_tokens":1}}']}


class RawUpstream:
    """An upstream that sends literal bytes, however malformed.

    MockUpstream speaks through BaseHTTPRequestHandler, which will not emit a
    Content-Length it then fails to honour — and that is exactly the case
    worth testing: a server that promises a body and dies partway through it.
    """

    def __init__(self, raw):
        self.raw = raw
        self.requests = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._stop = False

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                self.requests.append(conn.recv(65536))
                conn.sendall(self.raw)
                # Half-close rather than close(). close() straight after
                # sendall can emit an RST that discards the bytes still in
                # the kernel buffer, so the client sometimes saw none of the
                # partial body — flaky, and a property of this mock rather
                # than of the proxy. SHUT_WR delivers what was sent and then
                # signals EOF, which is what a server dying mid-body does.
                conn.shutdown(socket.SHUT_WR)
                conn.settimeout(5)
                try:
                    while conn.recv(65536):
                        pass
                except OSError:
                    pass
            except OSError:
                pass
            finally:
                conn.close()

    def __enter__(self):
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


#: 500 that promises 100 bytes of error body and delivers 13, then hangs up
TRUNCATED_ERROR = (b"HTTP/1.1 500 Internal Server Error\r\n"
                   b"Content-Type: application/json\r\n"
                   b"Content-Length: 100\r\n\r\npartial-error")


def sse(*events):
    """An SSE body from a list of delta strings, ending in [DONE]."""
    out = []
    for e in events:
        out.append(b"data: " + json.dumps(
            {"choices": [{"delta": {"content": e}}]}).encode() + b"\n\n")
    out.append(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
               b'"usage":{"prompt_tokens":10,"completion_tokens":3},'
               b'"timings":{"prompt_n":10,"predicted_n":3,'
               b'"predicted_per_second":50.0,"prompt_ms":100.0,'
               b'"predicted_ms":60.0,"prompt_per_second":100.0}}\n\n')
    out.append(b"data: [DONE]\n\n")
    return out


def write_records(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class TapTest(unittest.TestCase):
    """Base class: every subclass gets its own state directory."""

    def setUp(self):
        self._env = tap_env()
        self.state = self._env.__enter__()
        self.addCleanup(lambda: self._env.__exit__(None, None, None))

    @property
    def rec_path(self):
        return os.environ["LLAMATAP_REC"]
