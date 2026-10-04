"""Folding the model's thinking to one line.

A thinking block is often the largest part of a turn — 1,370 tokens and 20
seconds in one measured case — and reading the conversation around it means
scrolling past all of it. `/thinking` collapses each one to a single clickable
row.

The rule that matters: **folding hides the text, never the cost.** A collapsed
row still states its tokens and its seconds, because a thread that looks
cheaper than it was is a worse instrument than a verbose one.
"""

import json
import os
import unittest

from _support import TapTest, load


THINK = "Let me work through this.\n" * 40


def rec_with_thinking(i=0, think=None, timings=True):
    return {
        "t": f"2026-10-02T12:00:{i:02d}",
        "path": "/v1/chat/completions", "status": 200,
        "req": {"model": "qwen", "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "what is 2+2?"}]},
        "resp": json.dumps({
            "choices": [{"message": {"content": f"four ({i})",
                                     "reasoning_content":
                                         think or f"{THINK}step {i}\n"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 300},
            "timings": {"prompt_n": 50, "predicted_n": 300,
                        "prompt_ms": 100.0, "predicted_ms": 4000.0,
                        "prompt_per_second": 500.0,
                        "predicted_per_second": 75.0}}),
        "meta": {"usage": {"prompt_tokens": 50, "completion_tokens": 300},
                 "timings": {"prompt_n": 50, "predicted_n": 300,
                             "prompt_ms": 100.0, "predicted_ms": 4000.0,
                             "prompt_per_second": 500.0,
                             "predicted_per_second": 75.0,
                             "predicted_per_token_ms": 13.3},
                 "elapsed_ms": 4200}}


class ThinkingFold(TapTest):

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, fold=False, cols=140, rows=40):
        C = self.C

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen()
        # Distinct content per record: identical messages are correctly
        # deduplicated as repeats, which would hide the second thought.
        c.recs = [rec_with_thinking(0), rec_with_thinking(1)]
        c.name = None
        c.extra = []
        c.follow = True
        c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": fold, "stats": "right"}
        c.open = set()
        c.touched = set()
        c.hover = None
        c.hotspots = []
        c.starts = []
        c.menu = []
        c.detail = None
        c.node_bodies = {}; c.show = set(C.PARTIES); c.chips = []
        c.inflight = []
        c.msg = ""
        c.msg_until = 0
        c.tick = 0
        c._flags = ""
        c._flags_started = True
        c._flags_err = ""
        c._flags_src = "running"
        c._flags_age = None
        c._rec_name = {}
        c.rebuild()
        return c

    def text_of(self, c):
        return "\n".join(
            "".join(ch for _, ch, _ in spans) for spans in c.lines)

    def test_unfolded_shows_the_thinking_text(self):
        self.assertIn("Let me work through this", self.text_of(self.console()))

    def test_thinking_is_a_node_before_any_command_is_run(self):
        """It must be collapsible straight away, like every other node — not
        only after switching a global mode."""
        c = self.console(fold=False)
        nodes = [n for (_l, _a, _b, n) in c.hotspots if n[1] == "thinking"]
        self.assertEqual(len(nodes), 2,
                         "thinking has no +/- until /thinking is used")

    def test_it_starts_expanded_with_a_minus(self):
        body = self.text_of(self.console(fold=False))
        row = next(l for l in body.split("\n") if "Thought" in l)
        self.assertIn("−", row, f"expected an open node marker: {row!r}")

    def test_one_thought_can_be_closed_on_its_own(self):
        c = self.console(fold=False)
        nodes = [n for (_l, _a, _b, n) in c.hotspots if n[1] == "thinking"]
        before = len(c.lines)
        c.toggle(nodes[0])
        self.assertLess(len(c.lines), before, "closing one changed nothing")
        rows = [l for l in self.text_of(c).split("\n") if "Thought" in l]
        self.assertIn("+", rows[0], "the clicked thought did not close")
        self.assertIn("−", rows[1], "closing one closed the others too")

    def test_a_hand_closed_thought_survives_a_rebuild(self):
        c = self.console(fold=False)
        nodes = [n for (_l, _a, _b, n) in c.hotspots if n[1] == "thinking"]
        c.toggle(nodes[0])
        c.rebuild()
        rows = [l for l in self.text_of(c).split("\n") if "Thought" in l]
        self.assertIn("+", rows[0],
                      "the default reopened a thought the user had closed")

    def test_the_command_overrides_hand_choices(self):
        """`/thinking close` means all of them, including ones opened by hand."""
        c = self.console(fold=True)
        nodes = [n for (_l, _a, _b, n) in c.hotspots if n[1] == "thinking"]
        c.toggle(nodes[0])                      # open one by hand
        c.run_cmd("thinking close")
        for row in [l for l in self.text_of(c).split("\n") if "Thought" in l]:
            self.assertIn("+", row, "a hand-opened thought stayed open")

    def test_folded_hides_the_text(self):
        body = self.text_of(self.console(fold=True))
        self.assertNotIn("Let me work through this", body,
                         "the thinking text is still on screen when folded")

    def test_folded_is_shorter(self):
        self.assertLess(len(self.console(fold=True).lines),
                        len(self.console().lines),
                        "folding did not save any rows")

    def test_thinking_has_a_speaker_row(self):
        """It was the one block on screen with no name above it, so a thought
        read as belonging to whoever spoke last."""
        body = self.text_of(self.console(fold=True))
        self.assertIn("→ thinking", body,
                      "no speaker row above the thought")

    def test_folded_still_states_the_cost(self):
        """The point of the collapsed row — the cost is now on the SPEAKER
        row, beside the name, the way a tool result carries its stats."""
        body = self.text_of(self.console(fold=True))
        self.assertIn("Thought", body, "no folded thinking row at all")
        row = next(l for l in body.split("\n") if "→ thinking" in l)
        self.assertIn("tokens", row, f"no token count beside the name: {row!r}")
        self.assertRegex(row, r"\d+\.\d+s|\d+s",
                         f"no duration beside the name: {row!r}")

    def test_the_cost_is_outside_the_block(self):
        """`▌ − Thought: 31 tokens · 0.4s` put the meta inside the bar, where
        every other block keeps only its text."""
        body = self.text_of(self.console(fold=False))
        row = next(l for l in body.split("\n") if "Thought" in l)
        self.assertNotIn("tokens", row,
                         f"the cost is still inside the block: {row!r}")

    def test_an_unknown_duration_is_stated_not_omitted(self):
        """A row with only a token count looks like a thought that was free."""
        C = self.C
        rec = rec_with_thinking(0)
        rec["meta"]["timings"].pop("predicted_per_token_ms", None)

        class Screen:
            def getmaxyx(self):
                return (40, 140)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [rec]; c.name = None; c.extra = []
        c.follow = True; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": True, "stats": "right"}
        c.open = set(); c.touched = set()
        c.hover = None; c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        row = next((l for l in self.text_of(c).split("\n")
                    if "→ thinking" in l), "")
        self.assertTrue("s" in row.split("tokens")[-1] or "unknown" in row,
                        f"neither a duration nor a reason for its absence: {row!r}")

    def test_each_one_is_individually_expandable(self):
        c = self.console(fold=True)
        nodes = [nid for (_l, _a, _b, nid) in c.hotspots if nid[1] == "thinking"]
        self.assertEqual(len(nodes), 2,
                         f"expected one node per thinking block, got {nodes}")
        self.assertEqual(len(set(nodes)), 2, "the nodes are not distinct")

        c.toggle(nodes[0])      # what a click does; c.open alone is not it
        body = self.text_of(c)
        self.assertIn("Let me work through this", body,
                      "opening one node did not reveal its text")
        # and the other is still shut
        self.assertEqual(
            sum(1 for l in body.split("\n") if l.strip().startswith("+ Thought")
                or "+ Thought" in l), 1,
            "the second thought did not stay folded")

    def test_open_all_then_close_all(self):
        c = self.console(fold=True)
        c.run_cmd("thinking open")
        self.assertIn("Let me work through this", self.text_of(c))
        self.assertFalse(c.opts["fold_thinking"])
        c.run_cmd("thinking close")
        self.assertNotIn("Let me work through this", self.text_of(c))
        self.assertIn("Thought", self.text_of(c), "the folded rows disappeared")

    def test_the_bare_command_toggles(self):
        c = self.console(fold=False)
        c.run_cmd("thinking")
        self.assertTrue(c.opts["fold_thinking"])
        self.assertNotIn("Let me work through this", self.text_of(c))
        c.run_cmd("thinking")
        self.assertFalse(c.opts["fold_thinking"])
        self.assertIn("Let me work through this", self.text_of(c))

    def test_the_command_clears_per_node_decisions(self):
        c = self.console(fold=False)
        nodes = [n for (_l, _a, _b, n) in c.hotspots if n[1] == "thinking"]
        c.toggle(nodes[0])
        self.assertTrue(any(n[1] == "thinking" for n in c.touched))
        c.run_cmd("thinking")
        self.assertFalse(any(n[1] == "thinking" for n in c.touched),
                         "a global default left per-node choices behind")

    def test_the_command_is_listed(self):
        names = [n for n, _, _ in self.C.COMMANDS]
        self.assertIn("thinking", names)

    def test_the_key_is_documented(self):
        keys = "\n".join(self.C.HELP)
        self.assertIn("t ", keys, "the t key is not in the help")

    def test_status_bar_says_which_mode(self):
        for fold, word in ((True, "FOLDED"), (False, "full")):
            with self.subTest(fold=fold):
                c = self.console(fold=fold)
                c.scr.getmaxyx()
                self.assertTrue(hasattr(c, "opts"))
                self.assertEqual(c.opts["fold_thinking"], fold)

    def test_folding_leaves_everything_else_alone(self):
        """Replies, tool calls and the user's own words must still show."""
        body = self.text_of(self.console(fold=True))
        self.assertIn("what is 2+2?", body)
        self.assertIn("four", body)

    def test_the_rail_still_fits_when_folded(self):
        """Folding changes the conversation, never the rail."""
        for cols in (90, 118, 160):
            with self.subTest(cols=cols):
                c = self.console(fold=True, cols=cols)
                w = c.rail_w()
                for spans, _ in c.dash(w):
                    t = ("".join(x for x, _ in spans)
                         if isinstance(spans, list) else spans)
                    self.assertLessEqual(self.C.dwidth(t), w,
                                         f"rail row too wide: {t!r}")


if __name__ == "__main__":
    unittest.main()


class ThinkingSpacing(ThinkingFold):
    """A thought is a block like any other and gets the same gap after it.

    The blank line between blocks is emitted at the end of the message loop,
    which the thinking branch skipped with `continue` — so a thought was the
    one block that ran straight into whatever spoke next.
    """

    def rows(self, c):
        return ["".join(ch for _, ch, _ in spans).rstrip() for spans in c.lines]

    def test_a_blank_line_follows_a_thought(self):
        for fold in (True, False):
            with self.subTest(folded=fold):
                rows = self.rows(self.console(fold=fold))
                hits = [i for i, l in enumerate(rows) if "Thought" in l]
                self.assertTrue(hits, "no thinking block rendered")
                last = hits[-1]
                tail = next((i for i in range(last, len(rows))
                             if not rows[i].strip()), None)
                self.assertIsNotNone(tail, "a thought ends the buffer with no gap")
                between = [l for l in rows[last + 1:tail] if l.strip()]
                self.assertTrue(
                    all("▌" in l for l in between),
                    f"a non-block row ran straight into the thought: {between[:2]}")

    def test_the_gap_matches_every_other_block(self):
        """One blank line, not two — a thought must not look specially
        spaced either."""
        rows = self.rows(self.console(fold=True))
        for i, l in enumerate(rows[:-2]):
            if "Thought" in l:
                self.assertTrue(
                    rows[i + 1].strip() == "" or "▌" in rows[i + 1],
                    f"unexpected row after a thought: {rows[i + 1]!r}")
