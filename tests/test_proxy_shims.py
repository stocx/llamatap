"""The two body rewrites, and the rule that a tap must not lie about them.

Regressions covered:

F5  `_hoist_system` ran on OpenAI chat requests and moved any non-leading
    system message into a top-level `system` field, which an OpenAI-compatible
    server ignores. The instruction never reached the model.

F6  `_merge_developer` iterated a string-valued `content` character by
    character, extracted nothing, and still dropped the item.

F18 The rewritten body was what got recorded, so llamatap reported its own
    edit as the harness's request — in a tool whose entire claim is "see what
    the harness sent".

The shims exist for a real reason: Qwen3.5's chat template raises "System
message must be at the beginning." So the test does not demand "never rewrite".
It demands the rewrite be LOSSLESS and the recording be HONEST.
"""

import json
import os
import unittest

from _support import TapTest, load


def shim_host(mod, path="/v1/chat/completions"):
    """Bind the two rewrite methods to a stand-in with a request path."""
    tap = next(v for v in vars(mod).values()
               if isinstance(v, type) and hasattr(v, "_hoist_system"))

    class Dummy:
        pass

    d = Dummy()
    d.path = path
    d._hoist_system = tap._hoist_system.__get__(d, Dummy)
    d._merge_developer = tap._merge_developer.__get__(d, Dummy)
    return d


def all_text(obj):
    """Every string anywhere in a JSON structure, for "did anything vanish"."""
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(all_text(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(all_text(v))
    return out


OPENAI_BODY = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "IMPORTANT: always answer in French."},
        {"role": "user", "content": "what is 2+2?"},
    ],
}


class HoistSystem(TapTest):

    def setUp(self):
        super().setUp()
        self.P = load("proxy")

    def test_openai_request_keeps_every_instruction(self):
        """F5 — the core of it. Whatever the shim does to satisfy a template,
        the words must still be somewhere an OpenAI server will read."""
        host = shim_host(self.P, "/v1/chat/completions")
        out = json.loads(host._hoist_system(json.dumps(OPENAI_BODY).encode()))
        self.assertIn("IMPORTANT: always answer in French.",
                      "\n".join(all_text(out.get("messages"))),
                      "the instruction is no longer in `messages`, which is the "
                      "only place an OpenAI-compatible server looks")

    def test_openai_request_gets_no_anthropic_system_field(self):
        """A top-level `system` field is Anthropic's protocol. On a chat
        completion it is dead weight at best and a silent drop at worst."""
        host = shim_host(self.P, "/v1/chat/completions")
        out = json.loads(host._hoist_system(json.dumps(OPENAI_BODY).encode()))
        self.assertNotIn("system", out,
                         "a top-level `system` field was added to an OpenAI request")

    def test_openai_request_satisfies_the_template_constraint(self):
        """The reason the shim exists: no system message after index 0."""
        host = shim_host(self.P, "/v1/chat/completions")
        out = json.loads(host._hoist_system(json.dumps(OPENAI_BODY).encode()))
        later = [i for i, m in enumerate(out["messages"])
                 if m.get("role") == "system" and i > 0]
        self.assertEqual(later, [], "a non-leading system message remains")

    def test_an_already_valid_openai_request_is_untouched(self):
        body = {"model": "m", "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}]}
        raw = json.dumps(body).encode()
        host = shim_host(self.P, "/v1/chat/completions")
        self.assertEqual(host._hoist_system(raw), raw,
                         "a valid request was rewritten anyway")

    def test_anthropic_request_still_hoists(self):
        """On /v1/messages the top-level `system` field IS the protocol."""
        body = {"model": "m", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be terse"}]}
        host = shim_host(self.P, "/v1/messages")
        out = json.loads(host._hoist_system(json.dumps(body).encode()))
        self.assertIn("be terse", "\n".join(all_text(out.get("system"))),
                      "the Anthropic hoist regressed")

    def test_garbage_and_empty_bodies_pass_through(self):
        host = shim_host(self.P)
        for raw in (b"", b"not json", b"[]", b'{"no":"messages"}'):
            with self.subTest(raw=raw):
                self.assertEqual(host._hoist_system(raw), raw)


class MergeDeveloper(TapTest):

    def setUp(self):
        super().setUp()
        self.P = load("proxy")

    def test_string_valued_developer_item_is_preserved(self):
        """F6 — reproduced exactly: a plain-string developer instruction was
        deleted outright once any other item supplied a text block."""
        body = {"instructions": "base", "input": [
            {"role": "developer", "content": "NEVER delete files."},
            {"role": "developer",
             "content": [{"type": "input_text", "text": "Use tabs."}]},
            {"role": "user", "content": [{"type": "input_text", "text": "go"}]},
        ]}
        host = shim_host(self.P, "/v1/responses")
        out = json.loads(host._merge_developer(json.dumps(body).encode()))
        self.assertIn("NEVER delete files.", out["instructions"],
                      "a string-valued developer instruction was dropped")
        self.assertIn("Use tabs.", out["instructions"])
        self.assertIn("base", out["instructions"])

    def test_nothing_is_dropped_without_being_merged(self):
        """Whatever leaves `input` must turn up in `instructions`."""
        body = {"instructions": "", "input": [
            {"role": "system", "content": "sys as string"},
            {"role": "developer", "content": "dev as string"},
            {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
        ]}
        host = shim_host(self.P, "/v1/responses")
        out = json.loads(host._merge_developer(json.dumps(body).encode()))
        kept = "\n".join(all_text(out["input"]))
        merged = out["instructions"]
        for phrase in ("sys as string", "dev as string"):
            self.assertTrue(phrase in merged or phrase in kept,
                            f"{phrase!r} vanished entirely")

    def test_openai_body_is_untouched(self):
        raw = json.dumps(OPENAI_BODY).encode()
        host = shim_host(self.P, "/v1/chat/completions")
        self.assertEqual(host._merge_developer(raw), raw)


class RecordingFidelity(TapTest):
    """F18 — the recording must hold what the client sent, not what we sent on.

    Checked here at the unit level; the HTTP path is checked in
    test_proxy_http.py.
    """

    def test_record_stores_the_bytes_it_is_given(self):
        P = load("proxy")
        original = json.dumps(OPENAI_BODY).encode()
        P.record("/v1/chat/completions", 200, original, b'{"ok":true}',
                 {"shims": ["hoist_system"]})
        with open(os.environ["LLAMATAP_REC"]) as f:
            rec = json.loads(f.readline())
        self.assertEqual(rec["req"], json.loads(original))
        self.assertEqual(rec["meta"].get("shims"), ["hoist_system"],
                         "meta must name any shim that fired, so a rewritten "
                         "request stays auditable")


if __name__ == "__main__":
    unittest.main()
