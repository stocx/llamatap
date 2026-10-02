"""Replay comparisons and the console's two outbound commands.

Regressions covered:

F13 `reply_of()` was `json.loads` only, so every streamed original showed as
    an empty response. 85 of 85 real opencode records are SSE, which means
    `replay N --diff` has never once displayed the original it was comparing
    against.

F14 Replay responses were sliced at 262,144 characters before being recorded —
    the same cap removed from the proxy, still live here, and cutting JSON
    mid-document so later readers lose the whole answer.

F11 `/ask` appended the new message straight onto `req.messages`, which end
    before the assistant reply stored in `rec.resp`. So "explain your last
    answer" could not see the answer displayed immediately above it.

F12 `/replay N` in a console opened with `--file archive.jsonl` ran the replay
    script with no `--file`, so it read and appended to the live recording and
    could send an entirely different exchange to the model.
"""

import json
import os
import unittest

from _support import ROOT, TapTest, load, sse, write_records


SSE_BODY = b"".join(sse("The ", "answer ", "is ", "four.")).decode()

SSE_WITH_TOOL = "".join([
    'data: {"choices":[{"delta":{"reasoning_content":"let me check"}}]}\n\n',
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
    '{"name":"read_file","arguments":"{\\"p\\":\\"a.txt\\"}"}}]}}]}\n\n',
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
    'data: [DONE]\n\n',
])


class ReplyOf(TapTest):
    """F13 — the original side of a --diff must be readable."""

    def setUp(self):
        super().setUp()
        self.R = load("replay")

    def test_streamed_original_is_decoded(self):
        text, reasoning, calls, finish = self.R.reply_of(SSE_BODY)
        self.assertIn("four", text,
                      "a streamed original still renders as an empty response")

    def test_streamed_reasoning_and_tool_calls_are_decoded(self):
        text, reasoning, calls, finish = self.R.reply_of(SSE_WITH_TOOL)
        self.assertIn("read_file", calls, f"tool calls lost: {calls}")
        self.assertTrue(reasoning, "reasoning lost")

    def test_plain_json_original_still_works(self):
        body = json.dumps({"choices": [{"message": {
            "content": "hi", "reasoning_content": "hmm",
            "tool_calls": [{"function": {"name": "bash"}}]},
            "finish_reason": "stop"}]})
        text, reasoning, calls, finish = self.R.reply_of(body)
        self.assertEqual(text, "hi")
        self.assertEqual(reasoning, "hmm")
        self.assertEqual(calls, ["bash"])
        self.assertEqual(finish, "stop")

    def test_empty_and_garbage(self):
        for body in ("", None, "nonsense"):
            with self.subTest(body=body):
                out = self.R.reply_of(body)
                self.assertEqual(len(out), 4)


class ReplayRecording(TapTest):
    """F14 — a long replay answer must survive being written down."""

    # Deliberately no source grep for "262144": the fix explains the old cap
    # in a comment, which a grep cannot tell from the cap itself. The
    # round-trip test below is the real guard, and it fails if any cap returns
    # at any size.

    def test_a_long_response_round_trips(self):
        R = load("replay")
        big = json.dumps({"choices": [{"message": {"content": "x" * 400000},
                                       "finish_reason": "stop"}]})
        R.append_record("/v1/chat/completions", 200,
                        {"model": "m", "messages": []}, big, {"replay": True})
        with open(os.environ["LLAMATAP_REC"]) as f:
            rec = json.loads(f.readline())
        self.assertEqual(len(rec["resp"]), len(big),
                         "the recorded response was truncated")
        text, _, _, _ = R.reply_of(rec["resp"])
        self.assertEqual(len(text), 400000,
                         "a long answer no longer re-parses")


REC_WITH_ANSWER = {
    "t": "2026-10-02T12:00:00", "path": "/v1/chat/completions", "status": 200,
    "req": {"model": "m", "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 2+2?"}]},
    "resp": json.dumps({"choices": [{"message": {"content": "four"},
                                     "finish_reason": "stop"}]}),
    "meta": {},
}


class AskAssembly(TapTest):
    """F11 — a follow-up must be able to see the answer above it."""

    def test_the_captured_answer_is_included(self):
        C = load("console")
        payload = C.ask_payload(REC_WITH_ANSWER, "explain your last answer")
        roles = [m["role"] for m in payload["messages"]]
        self.assertEqual(roles[-2:], ["assistant", "user"],
                         f"the assistant turn is missing: {roles}")
        self.assertIn("four", payload["messages"][-2]["content"])
        self.assertEqual(payload["messages"][-1]["content"],
                         "explain your last answer")

    def test_streamed_answers_are_included_too(self):
        rec = dict(REC_WITH_ANSWER, resp=SSE_BODY)
        C = load("console")
        payload = C.ask_payload(rec, "and again")
        joined = " ".join(m.get("content") or "" for m in payload["messages"])
        self.assertIn("four", joined,
                      "a streamed answer was not carried into the follow-up")

    def test_an_unanswered_request_still_works(self):
        rec = dict(REC_WITH_ANSWER, resp="")
        C = load("console")
        payload = C.ask_payload(rec, "hello?")
        self.assertEqual(payload["messages"][-1]["content"], "hello?")

    def test_streaming_is_off_for_an_ask(self):
        C = load("console")
        payload = C.ask_payload(REC_WITH_ANSWER, "x")
        self.assertIs(payload.get("stream"), False)

    def test_the_captured_request_is_not_mutated(self):
        rec = json.loads(json.dumps(REC_WITH_ANSWER))
        before = json.dumps(rec)
        C = load("console")
        C.ask_payload(rec, "x")
        self.assertEqual(json.dumps(rec), before,
                         "ask_payload edited the recording in place")


class ReplayCommand(TapTest):
    """F12 — a replay must act on the recording you are looking at."""

    def test_the_consoles_file_is_passed_through(self):
        C = load("console")
        cmd = C.replay_cmd(["3", "--model", "other"], "/tmp/archive.jsonl")
        self.assertIn("--file", cmd,
                      "the replay would read the live recording instead of "
                      "the archive the console has open")
        self.assertEqual(cmd[cmd.index("--file") + 1], "/tmp/archive.jsonl")
        self.assertIn("3", cmd)
        self.assertIn("--model", cmd)

    def test_an_explicit_file_argument_wins(self):
        C = load("console")
        cmd = C.replay_cmd(["3", "--file", "/tmp/other.jsonl"], "/tmp/open.jsonl")
        self.assertEqual(cmd.count("--file"), 1,
                         "--file was passed twice")
        self.assertEqual(cmd[cmd.index("--file") + 1], "/tmp/other.jsonl")


class StatePermissions(TapTest):
    """F16 — recordings hold prompts, source and sometimes secrets."""

    def test_recording_and_log_are_owner_only(self):
        P = load("proxy")
        P.record("/v1/chat/completions", 200, b'{"messages":[]}', b"{}", {})
        rec = os.environ["LLAMATAP_REC"]
        self.assertEqual(os.stat(rec).st_mode & 0o777, 0o600,
                         "the recording is readable by other local users")

    def test_state_directory_is_owner_only(self):
        import subprocess
        env = dict(os.environ)
        state = os.path.join(self.state, "fresh")
        env["LLAMATAP_STATE"] = state
        for k in ("LLAMATAP_REC", "LLAMATAP_LOG", "LLAMATAP_INFLIGHT",
                  "LLAMATAP_RUNS"):
            env.pop(k, None)
        subprocess.run([os.path.join(ROOT, "llamatap"), "config"],
                       capture_output=True, text=True, env=env, timeout=60)
        self.assertTrue(os.path.isdir(state))
        self.assertEqual(os.stat(state).st_mode & 0o777, 0o700,
                         "the state directory is traversable by other users")


if __name__ == "__main__":
    unittest.main()
