"""Reading a recording back: request normalisation, response parsing, props.

Regressions covered: F1 (a bodyless request makes the whole recording
unopenable), F13 (SSE responses parse), F15 (props works against a direct
llama-server, and a failure is not cached forever).
"""

import json
import os
import unittest

from _support import MockUpstream, TapTest, load, sse, write_records


class NormaliseArity(TapTest):
    """F1 — normalise() must return (preamble, turns) on EVERY path.

    It used to return a bare list from its early guard and a 2-tuple at the
    end, so `preamble, turns = normalise(req)` raised ValueError for any
    request that was not a dict. A single `GET /v1/models` is recorded with
    `req: null`, which made the entire recording unopenable in the console.
    """

    def setUp(self):
        super().setUp()
        self.T = load("thread")

    def test_returns_a_pair_for_every_input_shape(self):
        for value in (None, [], "a string", 0, {}, {"messages": []}):
            with self.subTest(value=value):
                out = self.T.normalise(value)
                self.assertIsInstance(out, tuple, f"{value!r} did not give a tuple")
                self.assertEqual(len(out), 2, f"{value!r} gave {len(out)} values")
                preamble, turns = out          # must not raise
                self.assertIsInstance(preamble, str)
                self.assertIsInstance(turns, list)

    def test_a_bodyless_request_does_not_break_the_recording(self):
        recs = [
            {"t": "2026-10-02T10:00:00", "path": "/v1/models",
             "status": 200, "req": None, "resp": '{"data":[]}', "meta": {}},
            {"t": "2026-10-02T10:00:01", "path": "/v1/chat/completions",
             "status": 200,
             "req": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
             "resp": '{"choices":[{"message":{"content":"hello"}}]}', "meta": {}},
        ]
        msgs = self.T.conversation(recs)
        self.assertTrue(any(m.get("text") == "hi" for m in msgs),
                        "the real exchange was lost")

    def test_the_bodyless_exchange_is_still_visible(self):
        """It must not crash, and it must not silently vanish either: a
        recording that says 2 exchanges should not render as 1 with no
        explanation."""
        recs = [{"t": "2026-10-02T10:00:00", "path": "/v1/models",
                 "status": 200, "req": None, "resp": '{"data":[]}', "meta": {}}]
        msgs = self.T.conversation(recs)
        self.assertTrue(msgs, "a non-conversation exchange produced nothing at all")


class ResponseParsing(TapTest):
    """F13 — the response parser must handle SSE, which is what harnesses send.

    Every record from a streaming harness is SSE-framed. A json.loads-only
    parser sees nothing in all of them.
    """

    def setUp(self):
        super().setUp()
        self.T = load("thread")

    def test_plain_json_response(self):
        body = json.dumps({"choices": [{"message": {"content": "hello there"},
                                        "finish_reason": "stop"}]})
        turns = self.T.from_response(body)
        self.assertTrue(any("hello there" in (t.get("text") or "") for t in turns))

    def test_sse_response(self):
        body = b"".join(sse("hel", "lo ", "there")).decode()
        turns = self.T.from_response(body)
        text = " ".join(t.get("text") or "" for t in turns)
        self.assertIn("hello there", text.replace("  ", " "))

    def test_sse_with_reasoning_and_tool_calls(self):
        events = [
            b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
            b'{"name":"bash","arguments":"{\\"cmd\\":"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
            b'{"arguments":"\\"ls\\"}"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        turns = self.T.from_response(b"".join(events).decode())
        kinds = [t.get("kind") for t in turns]
        self.assertIn("thinking", kinds, f"reasoning not surfaced: {turns}")
        self.assertTrue(any(t.get("tool") == "bash" for t in turns),
                        f"tool call not surfaced: {turns}")

    def test_empty_and_garbage_are_safe(self):
        for body in ("", None, "not json at all", "data: nonsense\n\n"):
            with self.subTest(body=body):
                self.assertIsInstance(self.T.from_response(body), list)


class PropsLookup(TapTest):
    """F15 — props must work against a direct llama-server, not only llama-swap.

    The default upstream is now a direct llama-server on 127.0.0.1:8080, which
    serves /props and has no /upstream/<model>/props at all. Looking only for
    the llama-swap route meant a fresh install showed no model details and
    cached that empty result permanently.
    """

    PROPS = {
        "model_ftype": "IQ4_XS - 4.25 bpw",
        "build_info": "b11040-test",
        "total_slots": 4,
        "default_generation_settings": {"n_ctx": 32768,
                                        "params": {"temperature": 0.7}},
    }

    def test_direct_llama_server_props(self):
        body = json.dumps(self.PROPS).encode()
        routes = {"/props": {"status": 200,
                             "headers": {"Content-Type": "application/json"},
                             "chunks": [body]}}
        with MockUpstream(routes) as up:
            os.environ["LLAMATAP_UPSTREAM"] = up.url
            T = load("thread")
            got = T.props("some-model")
        self.assertEqual(got.get("build_info"), "b11040-test",
                         "a direct llama-server's /props was not found")

    def test_llama_swap_props_still_works(self):
        body = json.dumps(self.PROPS).encode()
        routes = {"/upstream/some-model/props":
                  {"status": 200, "headers": {"Content-Type": "application/json"},
                   "chunks": [body]}}
        with MockUpstream(routes) as up:
            os.environ["LLAMATAP_UPSTREAM"] = up.url
            T = load("thread")
            got = T.props("some-model")
        self.assertEqual(got.get("build_info"), "b11040-test",
                         "the llama-swap route regressed")

    def test_a_failed_lookup_is_not_cached_forever(self):
        """A server that is briefly unreachable must not permanently hide the
        model details for the rest of the session."""
        os.environ["LLAMATAP_UPSTREAM"] = "http://127.0.0.1:1"
        T = load("thread")
        self.assertEqual(T.props("m"), {})
        body = json.dumps(self.PROPS).encode()
        with MockUpstream({"/props": {"status": 200, "chunks": [body]}}) as up:
            T.PROPS_CACHE and None        # cache survives; must not be poisoned
            os.environ["LLAMATAP_UPSTREAM"] = up.url
            got = T.props("m", upstream=up.url)
        self.assertEqual(got.get("build_info"), "b11040-test",
                         "an earlier failure was cached and never retried")


if __name__ == "__main__":
    unittest.main()
