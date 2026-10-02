"""The proxy over real HTTP: streaming, lifecycle, recording fidelity, framing.

Regressions covered:

F2  `resp.read(4096)` on a BufferedReader blocks until it has 4096 bytes or
    EOF, including across upstream chunks. Small SSE events accumulated, so a
    short reply reached the harness only once generation had finished.

F7  Only HTTPError was caught. A refused connection, a timeout or a client
    hangup skipped both `inflight_clear` and `record`, so the exchange was
    never recorded and the console showed it running forever.

F8  The request body was read from Content-Length only, while
    Transfer-Encoding was forwarded — so a chunked request reached upstream
    with that header and no body.

F18 The recorded request must be what the client sent, not what the shims
    forwarded.

Streaming is asserted by ORDERING, never by elapsed time: the mock withholds
everything after its first event until the test releases a gate, and the test
asserts the client already holds that first event. Under `read(4096)` it does
not; under `read1` it does. No sleeps, no timing thresholds.
"""

import http.client
import json
import os
import threading
import time
import unittest

from _support import CHAT_OK, MockUpstream, TapTest, free_port, load, sse


class ProxyHarness:
    """Runs llamatap-proxy's own server against a given upstream."""

    def __init__(self, upstream_url):
        os.environ["LLAMATAP_UPSTREAM"] = upstream_url
        self.port = free_port()
        os.environ["LLAMATAP_PORT"] = str(self.port)
        self.mod = load("proxy")
        self.srv = self.mod.QuietServer(("127.0.0.1", self.port), self.mod.Tap)

    def __enter__(self):
        self._t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()

    def conn(self, timeout=20):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def records(self):
        path = os.environ["LLAMATAP_REC"]
        if not os.path.exists(path):
            return []
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def wait_records(self, n, timeout=15):
        """Block until `n` records exist.

        The handler runs on its own thread and calls record() AFTER writing
        the response, so reading the file the instant the client is done is a
        race. This waits on the condition with a generous ceiling; it is not a
        timing assertion — a slow machine just waits longer.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            recs = self.records()
            if len(recs) >= n:
                return recs
            time.sleep(0.02)
        return self.records()

    def wait_inflight_empty(self, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.inflight() == []:
                return True
            time.sleep(0.02)
        return self.inflight() == []

    def inflight(self):
        path = os.environ["LLAMATAP_INFLIGHT"]
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                return json.load(f) or []
        except Exception:
            return []


BODY = json.dumps({"model": "m", "stream": True,
                   "messages": [{"role": "user", "content": "hi"}]})


class Streaming(TapTest):

    def test_first_event_reaches_the_client_before_the_rest_is_sent(self):
        """F2 — the whole point of a streaming tap."""
        gate = threading.Event()
        events = sse(*[f"tok{i} " for i in range(30)])
        routes = {"/v1/chat/completions": {
            "status": 200,
            "headers": {"Content-Type": "text/event-stream"},
            "chunks": events, "chunked": True, "gate": gate}}
        with MockUpstream(routes) as up, ProxyHarness(up.url) as tap:
            c = tap.conn()
            c.request("POST", "/v1/chat/completions", BODY,
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            # Read only as much as the first event. If the proxy is buffering
            # to 4096 bytes this blocks until the gate opens and the whole
            # response is sent, so the assertion below never gets to run with
            # the gate still shut.
            got = {}

            def read_first():
                try:
                    got["data"] = r.read1(4096)
                except Exception as e:      # pragma: no cover - diagnostic
                    got["error"] = e

            t = threading.Thread(target=read_first, daemon=True)
            t.start()
            t.join(5)
            still_withheld = not gate.is_set()
            arrived = got.get("data") or b""
            gate.set()
            try:
                r.read()
            except Exception:
                pass
            c.close()

        self.assertTrue(still_withheld,
                        "test bug: the gate opened before the assertion")
        self.assertTrue(arrived,
                        "nothing reached the client while the upstream still "
                        "had events withheld — the proxy is buffering the stream")
        self.assertIn(b"tok0", arrived,
                      f"the first event did not arrive first: {arrived[:120]!r}")

    def test_the_whole_stream_still_arrives_intact(self):
        """Forwarding sooner must not drop or reorder anything."""
        events = sse(*[f"t{i}" for i in range(50)])
        routes = {"/v1/chat/completions": {
            "status": 200, "headers": {"Content-Type": "text/event-stream"},
            "chunks": events, "chunked": True}}
        with MockUpstream(routes) as up, ProxyHarness(up.url) as tap:
            c = tap.conn()
            c.request("POST", "/v1/chat/completions", BODY,
                      {"Content-Type": "application/json"})
            body = c.getresponse().read().decode()
            c.close()
        for i in range(50):
            self.assertIn(f"t{i}", body, f"event t{i} was lost")
        self.assertIn("[DONE]", body)


class Lifecycle(TapTest):

    def test_unreachable_upstream_is_recorded_and_cleared(self):
        """F7 — the model-host-is-asleep case: the console used to show a
        request spinning forever and nothing was written down."""
        dead = f"http://127.0.0.1:{free_port()}"      # nothing listening
        with ProxyHarness(dead) as tap:
            c = tap.conn()
            c.request("POST", "/v1/chat/completions", BODY,
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            status = r.status
            r.read()
            c.close()
            recs = tap.wait_records(1)
            empty = tap.wait_inflight_empty()

        self.assertGreaterEqual(status, 500,
                                "an unreachable upstream should be a gateway error")
        self.assertEqual(len(recs), 1,
                         "the failed exchange was not recorded")
        self.assertEqual(recs[0]["req"], json.loads(BODY),
                         "the failed exchange lost its request")
        self.assertTrue(empty,
                        "the in-flight entry leaked; the console would show "
                        "this request running forever")

    def test_upstream_error_status_is_recorded(self):
        routes = {"/v1/chat/completions": {
            "status": 500, "headers": {"Content-Type": "application/json"},
            "chunks": [b'{"error":"boom"}']}}
        with MockUpstream(routes) as up, ProxyHarness(up.url) as tap:
            c = tap.conn()
            c.request("POST", "/v1/chat/completions", BODY,
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            self.assertEqual(r.status, 500)
            r.read()
            c.close()
            self.assertEqual(len(tap.wait_records(1)), 1)
            self.assertTrue(tap.wait_inflight_empty())

    def test_client_hangup_mid_stream_clears_inflight(self):
        gate = threading.Event()
        events = sse(*[f"x{i} " for i in range(40)])
        routes = {"/v1/chat/completions": {
            "status": 200, "headers": {"Content-Type": "text/event-stream"},
            "chunks": events, "chunked": True, "gate": gate}}
        with MockUpstream(routes) as up, ProxyHarness(up.url) as tap:
            c = tap.conn()
            c.request("POST", "/v1/chat/completions", BODY,
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            r.read1(64)
            c.close()                      # walk away mid-stream
            gate.set()
            # The handler must finish and tidy up on its own.
            self.assertTrue(tap.wait_inflight_empty(),
                            "a client hangup left the request in flight")


class RecordingFidelityOverHttp(TapTest):

    def test_recorded_request_is_what_the_client_sent(self):
        """F18 — with a shim firing, the recording must still show the
        harness's bytes, and must say which shim ran."""
        body = json.dumps({"model": "m", "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "answer in French"},
            {"role": "user", "content": "2+2?"}]})
        routes = {"/v1/chat/completions": CHAT_OK}
        with MockUpstream(routes) as up, ProxyHarness(up.url) as tap:
            c = tap.conn()
            c.request("POST", "/v1/chat/completions", body,
                      {"Content-Type": "application/json"})
            c.getresponse().read()
            c.close()
            recs = tap.wait_records(1)
            sent_upstream = json.loads(up.requests[0][3])

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["req"], json.loads(body),
                         "the recording holds the proxy's rewrite, not the "
                         "harness's request")
        self.assertEqual(recs[0]["meta"].get("shims"), ["hoist_system"],
                         "a rewrite happened but the recording does not say so")
        # and the shim did its job on the way out
        roles = [m["role"] for m in sent_upstream["messages"]]
        self.assertEqual([i for i, r in enumerate(roles) if r == "system" and i > 0], [])


class Framing(TapTest):

    def test_bodyless_get_is_forwarded_and_recorded(self):
        """A model-discovery request is ordinary traffic."""
        routes = {"/v1/models": {
            "status": 200, "headers": {"Content-Type": "application/json"},
            "chunks": [b'{"data":[{"id":"m"}]}']}}
        with MockUpstream(routes) as up, ProxyHarness(up.url) as tap:
            c = tap.conn()
            c.request("GET", "/v1/models")
            r = c.getresponse()
            self.assertEqual(r.status, 200)
            self.assertIn(b'"data"', r.read())
            c.close()
            recs = tap.wait_records(1)
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0]["req"])
        # and the reader must survive it — that is F1, checked here end to end
        T = load("thread")
        T.conversation(recs)

    def test_chunked_request_is_not_forwarded_with_empty_body(self):
        """F8 — forwarding Transfer-Encoding with no body is the one outcome
        that is never acceptable. Either carry the body or refuse."""
        routes = {"/v1/chat/completions": CHAT_OK}
        with MockUpstream(routes) as up, ProxyHarness(up.url) as tap:
            c = tap.conn()
            c.putrequest("POST", "/v1/chat/completions")
            c.putheader("Content-Type", "application/json")
            c.putheader("Transfer-Encoding", "chunked")
            c.endheaders()
            raw = BODY.encode()
            c.send(b"%x\r\n%s\r\n0\r\n\r\n" % (len(raw), raw))
            r = c.getresponse()
            status = r.status
            r.read()
            c.close()
            upstream_calls = list(up.requests)

        if status < 400:
            self.assertTrue(upstream_calls, "nothing reached upstream at all")
            self.assertEqual(json.loads(upstream_calls[0][3]), json.loads(BODY),
                             "upstream received a request with no body")
        else:
            self.assertEqual(upstream_calls, [],
                             "the request was refused downstream but still "
                             "forwarded upstream")


if __name__ == "__main__":
    unittest.main()
