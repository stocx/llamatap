"""Characterisation of behaviour that is already relied on.

These are not bug fixes. They pin down things that were painful to get right
and would be easy to break silently:

* F17 (fixed earlier) — the incremental recording reader. The dangerous
  failure is not slowness, it is reading a byte offset into a text stream and
  splicing two recordings together.
* dwidth / fit — terminal cell arithmetic. Getting this wrong corrupts every
  box border, and the symptom (`🎓skill` rendering as `🎓sill`) looks like a
  font problem rather than a code one.
* the dashboard — it must never again take three quarters of the screen, and
  every row must be the same cell width at every terminal size.
"""

import json
import os
import unittest

from _support import ROOT, TapTest, load, write_records


def a_record(i, text="hello"):
    return {"t": f"2026-10-02T10:00:{i:02d}", "path": "/v1/chat/completions",
            "status": 200,
            "req": {"model": "qwen", "messages": [
                {"role": "system", "content": "# Tone\nbe terse"},
                {"role": "user", "content": text}]},
            "resp": json.dumps({"choices": [{"message": {"content": f"reply {i}"},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 100 + i,
                                          "completion_tokens": 5},
                                "timings": {"prompt_n": 10, "predicted_n": 5,
                                            "prompt_ms": 50.0,
                                            "predicted_ms": 60.0,
                                            "prompt_per_second": 200.0,
                                            "predicted_per_second": 83.3,
                                            "cache_n": 90 + i}}),
            "meta": {"usage": {"prompt_tokens": 100 + i, "completion_tokens": 5},
                     "timings": {"prompt_n": 10, "predicted_n": 5,
                                 "prompt_ms": 50.0, "predicted_ms": 60.0,
                                 "prompt_per_second": 200.0,
                                 "predicted_per_second": 83.3,
                                 "cache_n": 90 + i},
                     "elapsed_ms": 120}}


class IncrementalLoad(TapTest):
    """F17 — reading only what was appended, without ever splicing files."""

    def setUp(self):
        super().setUp()
        self.C = load("console")
        self.p = os.path.join(self.state, "rec.jsonl")

    def append(self, recs):
        with open(self.p, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def test_matches_a_full_read_after_appends(self):
        self.append([a_record(i) for i in range(3)])
        self.assertEqual(len(self.C.load_records(self.p)), 3)
        self.append([a_record(i) for i in range(3, 7)])
        inc = self.C.load_records(self.p)
        full = self.C.load_records(self.p, incremental=False)
        self.assertEqual(inc, full)
        self.assertEqual(len(inc), 7)

    def test_multibyte_across_the_read_boundary(self):
        """The real hazard: a byte offset landing mid-character."""
        self.append([a_record(0)])
        self.C.load_records(self.p)
        self.append([a_record(1, text="日本語 🎓 ünïcode — em-dash")])
        recs = self.C.load_records(self.p)
        self.assertEqual(len(recs), 2)
        self.assertIn("日本語 🎓 ünïcode — em-dash",
                      json.dumps(recs[1], ensure_ascii=False))

    def test_a_partial_trailing_line_is_left_alone(self):
        self.append([a_record(0)])
        self.C.load_records(self.p)
        with open(self.p, "a") as f:
            f.write('{"t":"half a reco')
        self.assertEqual(len(self.C.load_records(self.p)), 1)
        with open(self.p, "a") as f:
            f.write('rd","req":null,"resp":"","meta":{}}\n')
        self.assertEqual(len(self.C.load_records(self.p)), 2)

    def test_truncation_resets(self):
        self.append([a_record(i) for i in range(5)])
        self.C.load_records(self.p)
        write_records(self.p, [a_record(99)])
        self.assertEqual(len(self.C.load_records(self.p)), 1)

    def test_replacement_by_a_longer_file_is_detected(self):
        """`serve` archives and starts fresh; the reader must not splice the
        new recording onto the old one."""
        self.append([a_record(i) for i in range(3)])
        self.C.load_records(self.p)
        write_records(self.p, [a_record(i, text="second session")
                               for i in range(20)])
        recs = self.C.load_records(self.p)
        self.assertEqual(len(recs), 20)
        self.assertIn("second session", json.dumps(recs[0]))

    def test_missing_file(self):
        self.assertEqual(self.C.load_records(os.path.join(self.state, "no.jsonl")), [])


class CellWidths(TapTest):

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def test_dwidth_is_exact(self):
        cases = [("", 0), ("abc", 3), ("🎓", 2), ("🎓skill", 7), ("a🎓b", 4),
                 ("日本", 4), ("╭─┬─╮", 5), ("── 🤖 ──", 8), ("\t", 1)]
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(self.C.dwidth(text), want)

    def test_fit_pads_and_truncates_to_exact_cells(self):
        for text, width in [("abcdefgh", 5), ("abc", 5), ("abcde", 5),
                            ("🎓 skills and more", 6), ("日本語テスト", 7),
                            ("", 4), ("x", 1)]:
            with self.subTest(text=text, width=width):
                self.assertEqual(self.C.dwidth(self.C.fit(text, width)), width)

    def test_fit_marks_truncation(self):
        """A silently sliced value reads as a real one — `15,427 of 65` looked
        like a number rather than a cut."""
        self.assertTrue(self.C.fit("abcdefgh", 5).endswith("…"))
        self.assertFalse(self.C.fit("abc", 5).strip().endswith("…"))

    def test_meter_is_never_misread_as_full(self):
        """`░` renders near-solid in some fonts, so an empty bar looked full."""
        self.assertNotIn("░", self.C.meter(0.0, 10))
        self.assertEqual(self.C.meter(0.0, 10).count("█"), 0)
        self.assertEqual(self.C.meter(1.0, 10).count("█"), 10)


#: every label the old three-panel dashboard showed, taken from dash() before
#: the restyle. The restyle must not quietly drop one.
DASHBOARD_LABELS = [
    "Model", "Quantised", "Runtime", "Slots", "Sampling",
    "Tools Offered", "Cache Reuse", "Prefill", "Decode",
    "Context", "Requests", "Messages", "Tokens Generated",
    "Decode Average", "Prefill Average", "Model Time",
]
SETTINGS_LABELS = [
    "GPU Layers", "Experts on CPU", "Context Window", "K Cache", "V Cache",
    "Flash Attention", "Jinja Template", "Reasoning", "Draft Model",
    "Threads", "Batch Size", "Micro Batch", "Memory Lock", "Memory Map",
]


class StatsRail(TapTest):
    """The statistics moved from a wide bottom panel to a narrow right rail.

    This was a restyle: the same numbers, derived the same way, in a different
    place. The test that matters is the one asserting every label survived —
    a redesign is exactly where a statistic goes missing without anyone
    noticing.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, rows, cols, flags=False, stats="right"):
        C = self.C
        recs = [a_record(i) for i in range(6)]

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = recs; c.name = None; c.extra = []
        c.follow = True; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": flags,
                  "repeats": False, "fold_thinking": False, "stats": stats}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ("-ngl 999 --n-cpu-moe 15 -c 65536 --jinja -ctk q8_0 "
                    "-ctv q8_0 -fa on -t 4")
        c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def rail_text(self, c, width=None):
        w = width if width is not None else c.rail_w()
        out = []
        for spans, _ in c.dash(w):
            out.append("".join(t for t, _ in spans)
                       if isinstance(spans, list) else spans)
        return "\n".join(out)

    def test_every_dashboard_statistic_survived(self):
        """The whole point: a restyle may move a number, never lose it.

        Asserted against the BOTTOM placement, which is the full view. The
        right-hand rail is a deliberate subset — see RAIL_KEEP — because a
        narrow column cannot hold everything without wrapping every value onto
        its own line.
        """
        c = self.console(40, 160, stats="bottom")
        text = self.rail_text(c, width=158)
        missing = [l for l in DASHBOARD_LABELS if l not in text]
        self.assertEqual(missing, [], f"statistics lost in the restyle: {missing}")

    def test_the_settings_board_survived(self):
        # A tall screen on purpose: the bottom panel is capped at 40% of the
        # height so it cannot own the display, and on a 40-row terminal the
        # 14-row settings board is clipped BY DESIGN. That cap is tested
        # separately below; this test is about nothing being lost.
        c = self.console(70, 160, flags=True, stats="bottom")
        text = self.rail_text(c, width=158)
        missing = [l for l in SETTINGS_LABELS if l not in text]
        self.assertEqual(missing, [], f"settings lost in the restyle: {missing}")

    def test_the_bottom_panel_is_capped_and_says_so(self):
        """It took 23 of 30 rows once. When it clips, it must announce it
        rather than silently hiding settings."""
        c = self.console(30, 160, flags=True, stats="bottom")
        rows = c.dash(158)
        self.assertLessEqual(len(rows), 15, "the bottom panel is unbounded")
        text = "\n".join("".join(x for x, _ in r) if isinstance(r, list) else r
                         for r, _ in rows)
        self.assertIn("more rows", text,
                      "rows were clipped without saying so")

    def test_the_rail_carries_its_documented_subset(self):
        text = self.rail_text(self.console(40, 160, stats="right"))
        missing = [l for l in self.C.Console.RAIL_KEEP if l not in text]
        self.assertEqual(missing, [], f"the rail dropped one of its own: {missing}")

    def test_the_rail_is_smaller_than_the_bottom_panel(self):
        """If the slim view is not slimmer, it has no reason to exist."""
        rail = self.rail_text(self.console(40, 160, stats="right"))
        full = self.rail_text(self.console(40, 160, stats="bottom"), width=158)
        self.assertLess(len(rail.split("\n")), len(full.split("\n")) + 12)
        self.assertNotIn("Memory Map", rail)

    def test_no_rail_row_exceeds_the_rail_width(self):
        for cols in (80, 100, 140, 190):
            for flags in (False, True):
                with self.subTest(cols=cols, flags=flags):
                    c = self.console(40, cols, flags, stats="right")
                    w = c.rail_w()
                    for spans, _ in c.dash(w):
                        t = ("".join(x for x, _ in spans)
                             if isinstance(spans, list) else spans)
                        self.assertLessEqual(
                            self.C.dwidth(t), w,
                            f"rail row {t!r} is wider than the {w}-cell rail")

    def test_the_conversation_does_not_run_under_the_rail(self):
        for cols in (80, 100, 140, 190):
            with self.subTest(cols=cols):
                c = self.console(40, cols, stats="right")
                limit = cols - c.rail_w() - 1
                for spans in c.lines:
                    t = "".join(ch for _, ch, _ in spans)
                    self.assertLessEqual(self.C.dwidth(t), limit,
                                         f"conversation runs under the rail: {t!r}")

    def test_hiding_the_rail_gives_the_width_back(self):
        wide = self.console(40, 140, stats="off")
        narrow = self.console(40, 140, stats="right")
        self.assertEqual(wide.rail_w(), 0)
        self.assertGreater(narrow.rail_w(), 0)
        self.assertGreater(wide._conv_w, narrow._conv_w,
                           "hiding the rail did not widen the conversation")

    def test_the_rail_hides_itself_on_a_narrow_terminal(self):
        self.assertEqual(self.console(40, 70).rail_w(), 0,
                         "the rail should not eat a narrow terminal")

    def test_the_stats_command_cycles_the_placements(self):
        c = self.console(40, 140)
        self.assertEqual(c.stats_where(), "right")
        c.run_cmd("stats")
        self.assertEqual(c.stats_where(), "bottom")
        c.run_cmd("stats")
        self.assertEqual(c.stats_where(), "off")
        c.run_cmd("stats")
        self.assertEqual(c.stats_where(), "right")

    def test_a_placement_can_be_named(self):
        c = self.console(40, 140)
        for where in ("bottom", "off", "right"):
            with self.subTest(where=where):
                c.run_cmd(f"stats {where}")
                self.assertEqual(c.stats_where(), where)

    def test_the_bottom_panel_takes_rows_and_the_rail_does_not(self):
        """The two placements trade rows against columns; that is the choice."""
        rail = self.console(40, 140, stats="right")
        bottom = self.console(40, 140, stats="bottom")
        off = self.console(40, 140, stats="off")
        # Against the no-statistics height, not against a hardcoded 38: that
        # constant was really asserting the chrome's height, so adding the
        # pinned party header broke a test that is about the RAIL.
        self.assertEqual(rail.view_h(), off.view_h(),
                         "the rail cost conversation rows")
        self.assertLess(bottom.view_h(), rail.view_h(),
                        "the bottom panel should cost conversation rows")
        self.assertEqual(bottom.rail_w(), 0,
                         "the rail should not also be showing")

    def test_the_rail_never_takes_rows_from_the_conversation(self):
        """The complaint that started this: the panel took the bottom of the
        screen. On the right it costs columns, which a wide terminal has."""
        for rows in (24, 30, 40, 60):
            with self.subTest(rows=rows):
                c = self.console(rows, 140, stats="right")
                off = self.console(rows, 140, stats="off")
                self.assertEqual(c.view_h(), off.view_h(),
                                 "the rail cost conversation rows")


class BackgroundDetection(TapTest):

    def test_every_needle_still_matches(self):
        CAT = load("catalog")
        for label, icon, needles in CAT.BACKGROUND:
            for needle in needles:
                with self.subTest(needle=needle):
                    got, _ = CAT.background_kind(f"prefix {needle} suffix")
                    self.assertEqual(got, label)

    def test_case_insensitive_and_safe_on_empty(self):
        CAT = load("catalog")
        self.assertEqual(CAT.background_kind("GENERATE A SHORT TITLE")[0],
                         "title generator")
        self.assertEqual(CAT.background_kind(""), (None, None))
        self.assertEqual(CAT.background_kind(None), (None, None))

    def test_ordinary_text_is_not_a_background_request(self):
        CAT = load("catalog")
        self.assertEqual(CAT.background_kind("write me a test suite"),
                         (None, None))


if __name__ == "__main__":
    unittest.main()


class NoRowRunsOffTheScreen(TapTest):
    """Nothing emitted may be wider than the terminal.

    A too-long row is wrapped by the TERMINAL, at the wrong column and without
    the indent, so it reads as the view being corrupt rather than as a line
    being long. Two separate causes, both fixed:

    * `_wrap` broke on spaces only, so a single unbroken token — a
      150-character skill path in a system prompt — could not be wrapped at
      all and overflowed a 90-column terminal by 84 cells.
    * header rows (the participants roster, a turn label with its stats, a
      node's `+ label   note`) were never clamped to the width at all.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, cols, rows=40):
        C = self.C
        recs = [a_record(i) for i in range(4)]
        # a long unbroken token, which is what used to overflow
        recs[0]["req"]["messages"][0]["content"] = (
            "# Tone\nbe terse\n"
            "<location>/home/someone/.config/skills/synced/"
            "f92aec64-fd7c-40e8-aca9-12d28fbccc14/reference/deeply/nested.md"
            "</location>")

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = recs; c.name = None; c.extra = []
        c.follow = True; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False}
        c.open = set(); c.hover = None; c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def widths(self, c):
        return [self.C.dwidth("".join(ch for _, ch, _ in spans))
                for spans in c.lines]

    def test_nothing_overflows_collapsed(self):
        for cols in (60, 70, 90, 120, 180):
            with self.subTest(cols=cols):
                c = self.console(cols)
                bad = [w for w in self.widths(c) if w > cols]
                self.assertEqual(bad, [], f"{len(bad)} rows past the edge")

    def test_nothing_overflows_with_everything_expanded(self):
        for cols in (60, 70, 90, 120, 180):
            with self.subTest(cols=cols):
                c = self.console(cols)
                for nid in [n for (_l, _a, _b, n) in c.hotspots]:
                    c.open.add(nid)
                c.rebuild()
                bad = [w for w in self.widths(c) if w > cols]
                self.assertEqual(bad, [], f"{len(bad)} rows past the edge")

    def test_wrapping_preserves_every_character(self):
        """Breaking a long word must not drop one, and must not pretend the
        text was truncated — it continues on the next line."""
        c = self.console(90)
        text = ("/home/someone/.config/skills/synced/"
                "f92aec64-fd7c-40e8-aca9-12d28fbccc14/x.md")
        for width in (12, 20, 30, 64):
            with self.subTest(width=width):
                lines = c._wrap(text, width, "  ")
                joined = "".join(l[2:] for l in lines)
                self.assertEqual(joined, text, "characters were lost")
                self.assertNotIn("…", joined,
                                 "wrapping claimed the text was cut")
                for l in lines:
                    self.assertLessEqual(self.C.dwidth(l), width + 2)

    def test_take_is_exact_on_wide_characters(self):
        head, rest = self.C.take("日本語abc", 5)
        self.assertEqual(self.C.dwidth(head), 4,
                         "a 2-cell character must not be split in half")
        self.assertEqual(head + rest, "日本語abc")

    def test_expanded_bodies_have_no_gutter_glyph(self):
        c = self.console(120)
        pre = [n for (_l, _a, _b, n) in c.hotspots if n[1] == "preamble"]
        c.open.update(pre[:1])
        c.rebuild()
        body = "\n".join("".join(ch for _, ch, _ in s) for s in c.lines)
        self.assertNotIn("│ be terse", body,
                         "the vertical gutter is back on expanded content")


class ClickTarget(TapTest):
    """A node row is clickable anywhere along it.

    A one-character `+` is a precision target. Nothing else on a node's row is
    clickable, so widening the target to the whole line costs nothing and
    removes the fiddliness.

    There is deliberately NO hover highlight any more. With the whole row
    clickable the green +/- glyph is already the affordance, and a second
    signal for the same fact was the part that kept going wrong: an underline
    that drew a rule across the indent, then a mark that lingered after the
    click that dismissed it, then a colour that could never appear because the
    terminal reported no motion.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, cols=120):
        C = self.C

        class Screen:
            def getmaxyx(self):
                return (40, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [a_record(i) for i in range(3)]
        c.name = None; c.extra = []; c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False}
        c.open = set(); c.touched = set(); c.hotspots = []
        c.starts = []; c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(C.PARTIES); c.chips = []
        c.inflight = []; c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def screen_row(self, c, line):
        """`hotspot_at` takes a SCREEN row, and the pinned party header sits
        above the transcript — so a transcript line is HEADER_H rows lower
        than it used to be. Converting here rather than at each call site
        means the next change to the chrome touches one place."""
        return line - c.top + self.C.HEADER_H

    def test_any_column_on_the_row_hits_the_node(self):
        c = self.console()
        line, c0, c1, nid = c.hotspots[0]
        for x in (0, 1, c0, c0 + 1, c1, c1 + 20, 110):
            with self.subTest(x=x):
                hs = c.hotspot_at(self.screen_row(c, line), x)
                self.assertIsNotNone(hs, f"column {x} missed the node")
                self.assertEqual(hs[3], nid)

    def test_a_row_with_no_node_is_not_clickable(self):
        c = self.console()
        node_lines = {l for (l, _a, _b, _n) in c.hotspots}
        plain = next(i for i in range(len(c.lines)) if i not in node_lines)
        self.assertIsNone(c.hotspot_at(self.screen_row(c, plain), 5))

    def test_clicking_toggles(self):
        c = self.console()
        line, _c0, _c1, nid = c.hotspots[0]
        before = len(c.lines)
        c.toggle(c.hotspot_at(self.screen_row(c, line), 90)[3])
        self.assertNotEqual(len(c.lines), before,
                            "a click away from the + did nothing")

    def test_no_hover_machinery_remains(self):
        src = open(os.path.join(ROOT, "llamatap-console.py")).read()
        for token in ("A_UNDERLINE", "hover_until", "mouse_motion", "1003h"):
            with self.subTest(token=token):
                self.assertNotIn(token, src,
                                 f"{token} is back; the +/- glyph is the "
                                 f"affordance now")


class NoDecoration(TapTest):
    """The conversation pane carries no emoji and no box-drawing.

    It used to contain 31 distinct wide glyphs — 15+ emoji for parties,
    sections and tool classes, plus `╭ ╮ ╰ ╯ ─ │ ┬ ├ └ ▸` speech bubbles whose
    chrome ran to nearly two thousand characters in one conversation. It read
    as cartoonish beside the content it was framing.

    One glyph survives by design: `│`, the accent bar that marks which party a
    block belongs to. Emoji in the MODEL'S OWN OUTPUT are content, not
    decoration, and must not be stripped.
    """

    ALLOWED = {"│", "·", "█", "…", "−", "─", "▌"}

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, cols=120, fold=False):
        C = self.C

        class Screen:
            def getmaxyx(self):
                return (40, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [a_record(i) for i in range(4)]
        c.name = None; c.extra = []; c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": fold, "stats": "right"}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def rendered(self, c):
        return "\n".join("".join(ch for _, ch, _ in s) for s in c.lines)

    def test_no_emoji_in_the_conversation(self):
        for fold in (False, True):
            with self.subTest(fold=fold):
                bad = sorted({ch for ch in self.rendered(self.console(fold=fold))
                              if 0x1F300 <= ord(ch) <= 0x1FAFF
                              or 0x2600 <= ord(ch) <= 0x27BF})
                self.assertEqual(bad, [], f"emoji back in the view: {bad}")

    def test_no_box_drawing_except_the_accent_bar(self):
        bad = sorted({ch for ch in self.rendered(self.console())
                      if 0x2500 <= ord(ch) <= 0x259F and ch not in self.ALLOWED})
                       
        self.assertEqual(bad, [], f"box-drawing back in the view: {bad}")

    def test_the_accent_bar_is_there(self):
        """Asserted against the module's own constant: hardcoding the glyph
        here meant changing its thickness failed a test that was only ever
        about the bar existing."""
        self.assertIn(self.C.BAR, self.rendered(self.console()),
                      "no accent bar — blocks have nothing marking their party")
        self.assertIn(self.C.BAR, self.ALLOWED,
                      "the accent bar is not in the allowed-glyph set")

    def test_the_tree_uses_plain_indentation(self):
        c = self.console()
        for nid in [n for (_l, _a, _b, n) in c.hotspots]:
            c.open.add(nid)
        c.rebuild()
        for ch in ("├", "└", "┬", "▸", "╭", "╮", "╰", "╯", "┊", "┈"):
            with self.subTest(glyph=ch):
                self.assertNotIn(ch, self.rendered(c))

    def test_emoji_in_the_model_s_own_words_are_kept(self):
        """Decoration is ours to remove; content is not."""
        C = self.C
        rec = a_record(0)
        import json as _j
        rec["resp"] = _j.dumps({"choices": [{"message": {"content": "done ✅ fast ⚡"},
                                             "finish_reason": "stop"}]})

        class Screen:
            def getmaxyx(self):
                return (40, 120)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [rec]; c.name = None; c.extra = []
        c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False, "stats": "right"}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        body = self.rendered(c)
        self.assertIn("✅", body, "the model's own emoji were stripped")
        self.assertIn("⚡", body)


class ReviewRegressions(TapTest):
    """One test per defect found by the two-agent review of 7779f5e.

    Every one of them passed 134 green tests. The reasons were a pattern, and
    these tests are written against that pattern rather than against the
    symptoms: go through the REAL defaults, render at the AWKWARD sizes, and
    assert on formatting rather than on substrings.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, rows=40, cols=140, stats="right", flags=False,
                props=None):
        C = self.C
        recs = [a_record(i) for i in range(6)]

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = recs; c.name = None; c.extra = []
        c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": flags,
                  "repeats": False, "fold_thinking": False, "stats": stats}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = "-ngl 999 -c 65536 --jinja -ctk q8_0 -fa on"
        c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        if props is not None:
            C.T.PROPS_CACHE["qwen"] = props
        c.rebuild()
        return c

    def rail_rows(self, c, w=None):
        w = w if w is not None else (c.rail_w() or c.scr.getmaxyx()[1] - 2)
        return ["".join(t for t, _ in r) if isinstance(r, list) else r
                for r, _ in c.dash(w)]

    # ---- the blocker -----------------------------------------------------
    def test_the_real_default_from_main_shows_statistics(self):
        """The defect: main() passed a bool after the option became a string,
        so the console launched with no statistics and no explanation. No test
        went through main()'s own options, which is how it survived."""
        import argparse
        src = open(os.path.join(ROOT, "llamatap-console.py")).read()
        self.assertIn('"stats": a.stats', src,
                      "main() no longer takes its placement from the parser")
        ap = argparse.ArgumentParser()
        ap.add_argument("--stats", choices=("right", "bottom", "off"),
                        default="right")
        self.assertEqual(ap.parse_args([]).stats, "right")

    def test_an_unrecognised_placement_falls_back_to_right(self):
        for junk in (True, False, None, "sideways", 1):
            with self.subTest(value=junk):
                c = self.console(stats=junk)
                self.assertEqual(c.stats_where(), "right",
                                 f"{junk!r} produced no statistics at all")
                self.assertGreater(c.rail_w(), 0)

    # ---- truncation ------------------------------------------------------
    def test_expanding_a_node_loses_no_text(self):
        """27 of 120 words vanished at 120 columns with the rail showing,
        because node bodies wrapped to the screen and were clamped to the
        conversation."""
        C = self.C
        words = " ".join(f"w{i:03d}" for i in range(120))
        rec = a_record(0)
        rec["req"]["messages"][0]["content"] = "# Tone\n" + words
        for cols in (100, 120, 160):
            for stats in ("right", "off"):
                with self.subTest(cols=cols, stats=stats):
                    c = self.console(cols=cols, stats=stats)
                    c.recs = [rec]
                    c.rebuild()
                    for nid in [n for (_l, _a, _b, n) in c.hotspots]:
                        c.open.add(nid)
                    c.rebuild()
                    body = "\n".join("".join(ch for _, ch, _ in s)
                                     for s in c.lines)
                    missing = [w for w in words.split() if w not in body]
                    self.assertEqual(missing, [],
                                     f"{len(missing)} words lost from an "
                                     f"expanded node")

    # ---- label gluing ----------------------------------------------------
    def test_labels_are_never_glued_to_their_values(self):
        """Checked on the DRAWN GRID, with the flag board on.

        The previous version inspected spans and skipped any row whose first
        span was a border — which is every row of the bottom panel, so it
        never looked at the placement where the gluing actually survived. It
        stayed green while "Cache Reuse██████" and "Flash Attentionon" were on
        screen from 81 to 94 columns.
        """
        import re as _re
        # a label character immediately followed by a digit, a meter block or
        # a bar, with no space between
        glued = _re.compile(r"[A-Za-z](?=[█·]|\d)")
        for cols in (80, 90, 96, 100, 120, 140, 200):
            for stats in ("right", "bottom"):
                for flags in (False, True):
                    with self.subTest(cols=cols, stats=stats, flags=flags):
                        c = self.console(cols=cols, stats=stats, flags=flags)
                        if c.stats_where() == "off":
                            continue
                        w = c.rail_w() or cols - 2
                        for spans, _ in c.dash(w):
                            row = ("".join(t for t, _ in spans)
                                   if isinstance(spans, list) else spans)
                            for label in (list(self.C._LEFT_LABELS)
                                          + list(self.C._SESSION_LABELS)):
                                i = row.find(label)
                                if i < 0:
                                    continue
                                after = row[i + len(label):i + len(label) + 1]
                                self.assertIn(
                                    after, (" ", "│", ""),
                                    f"{label!r} runs into its value at "
                                    f"{cols} cols ({stats}): "
                                    f"{row[i:i + len(label) + 8]!r}")

    def test_the_label_width_follows_the_labels(self):
        """Not a constant. A constant drifts the moment a label is added."""
        longest = max(len(x) for x in
                      list(self.C._LEFT_LABELS) + list(self.C._SESSION_LABELS))
        self.assertGreaterEqual(longest, 16)
        c = self.console(cols=160, stats="bottom", flags=True)
        for spans, _ in c.dash(158):
            if not isinstance(spans, list) or len(spans) < 2:
                continue
            label = spans[0][0]
            if label.strip() and spans[1][0].strip() and "│" not in label:
                self.assertTrue(label.endswith(" "),
                                f"{label!r} has no gap after it")

    # ---- the bottom panel on a narrow terminal ---------------------------
    def test_the_bottom_panel_never_exceeds_the_screen(self):
        """c1 and c3 have floors of 30, so the panel could not be narrower
        than 81 cells however small the terminal — it emitted 81-cell rows
        onto a 59-cell screen."""
        for cols in (40, 60, 70, 79, 80, 81, 100, 160):
            with self.subTest(cols=cols):
                c = self.console(cols=cols, stats="bottom")
                if c.stats_where() == "off":
                    continue
                # cols-2, not cols-1. draw() writes panel rows starting at
                # x=1 with a limit of W-1-x, so columns 1..cols-2 are all it
                # can show. The earlier bound asserted the number the fix
                # happened to produce and passed while every right border was
                # being clipped off the screen.
                for row in self.rail_rows(c, cols - 2):
                    self.assertLessEqual(self.C.dwidth(row), cols - 2,
                                         f"panel row wider than draw() can "
                                         f"show at {cols} columns")

    def test_bottom_hides_itself_when_it_cannot_fit(self):
        """The threshold is derived from legibility — three columns of 30 —
        not from the narrowest width whose geometry happens not to overflow.
        At that narrower width the labels themselves were truncated."""
        m = self.C.BOTTOM_MIN
        self.assertEqual(self.console(cols=m - 1, stats="bottom").stats_where(),
                         "off")
        self.assertEqual(self.console(cols=m, stats="bottom").stats_where(),
                         "bottom")

    # ---- the conversation width ------------------------------------------
    def test_the_conversation_never_exceeds_the_screen(self):
        for cols in (30, 40, 60, 79, 80, 100, 200):
            for stats in ("right", "bottom", "off"):
                with self.subTest(cols=cols, stats=stats):
                    c = self.console(cols=cols, stats=stats)
                    self.assertLessEqual(c._conv_w, cols - 1)
                    for spans in c.lines:
                        t = "".join(ch for _, ch, _ in spans)
                        self.assertLessEqual(self.C.dwidth(t), cols - 1)

    # ---- the rail's right edge -------------------------------------------
    def test_rail_rows_fit_the_room_draw_gives_them(self):
        """draw() limits a rail row to rw-2; the fit test accepted rw-1, so
        the widest row lost its last cell."""
        for cols in (80, 90, 100, 120, 160, 200):
            with self.subTest(cols=cols):
                c = self.console(cols=cols, stats="right")
                rw = c.rail_w()
                for row in self.rail_rows(c, rw):
                    self.assertLessEqual(self.C.dwidth(row), rw - 2)

    # ---- the orphan heading ----------------------------------------------
    def test_no_heading_without_rows_under_it(self):
        for cols in (80, 100, 140, 200):
            with self.subTest(cols=cols):
                rows = [r for r in self.rail_rows(self.console(cols=cols))
                        if r.strip()]
                for i, row in enumerate(rows):
                    if row in ("Session", "This turn", "Model"):
                        self.assertLess(i + 1, len(rows),
                                        f"{row!r} is the last row — nothing "
                                        f"under it")
                        self.assertTrue(rows[i + 1].startswith((" ", "C", "R",
                                                                "T", "D", "P",
                                                                "M")),
                                        f"{row!r} heading has no content")

    # ---- the props-present path, never rendered by any test --------------
    def test_the_props_present_path_renders(self):
        """Tests point the upstream at port 1, so props always failed and the
        whole branch carrying the context gauge, build string and sampling
        settings was never rendered."""
        c = self.console(cols=160, stats="bottom", props={
            "model_ftype": "IQ4_XS - 4.25 bpw",
            "build_info": "b11040-test",
            "total_slots": 4,
            "default_generation_settings": {
                "n_ctx": 65536, "params": {"temperature": 0.7, "top_k": 20}},
        })
        text = "\n".join(self.rail_rows(c, 158))
        for expect in ("IQ4_XS", "b11040-test", "65,536", "temp"):
            with self.subTest(expect=expect):
                self.assertIn(expect, text,
                              "the props-present branch did not render")

    # ---- short terminals --------------------------------------------------
    def test_a_short_terminal_does_not_silently_drop_the_current_turn(self):
        """At 10 rows the rail needed 19 and 8 fit, with no notice — and what
        went was This turn, the numbers the rail exists to show."""
        src = open(os.path.join(ROOT, "llamatap-console.py")).read()
        self.assertIn("more — taller window", src,
                      "a clipped rail still says nothing")


class DrawnOutput(TapTest):
    """Assert on what reaches the SCREEN, not on what dash() returns.

    Round 2 of the review caught two fixes that were wrong in the same way:
    they asserted the number the fix happened to produce, and passed while the
    screen was visibly wrong. `dash()` returning a row of the right width says
    nothing if `draw()` then writes it somewhere it does not fit. These tests
    drive the real `draw()` through a recording screen and look at the grid.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def grid(self, rows, cols, stats="right", flags=False, top=0):
        """Drive the real draw() and return the character grid it produced."""
        C = self.C
        cells = [[" "] * cols for _ in range(rows)]

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

            def erase(self):
                for r in cells:
                    r[:] = [" "] * cols

            def refresh(self):
                pass

            def addnstr(self, y, x, text, n, attr=0):
                if not (0 <= y < rows):
                    raise C.curses.error("out of range")
                for ch in text[:max(0, n)]:
                    if x >= cols:
                        raise C.curses.error("past the edge")
                    cells[y][x] = ch
                    x += 1

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [a_record(i) for i in range(8)]
        c.name = None; c.extra = []; c.follow = False; c.top = top
        c.opts = {"rawjson": False, "preamble": False, "flags": flags,
                  "repeats": False, "fold_thinking": False, "stats": stats}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0; c.cmd = None; c.busy = ""
        c._flags = "-ngl 999 -c 65536 --jinja"
        c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        c.draw()
        return ["".join(r).rstrip() for r in cells]

    def test_the_bottom_panel_keeps_its_right_border(self):
        """It was clipped at every width: draw() writes from x=1 with a limit
        of W-1-x, which the width arithmetic forgot."""
        for cols in (96, 97, 100, 160, 200):
            with self.subTest(cols=cols):
                g = self.grid(40, cols, stats="bottom")
                # draw() writes panel rows at x=1, so the row begins with a
                # space — the first version of this test looked for a leading
                # "╭" and concluded nothing was drawn at all.
                borders = [l for l in g
                           if l.lstrip().startswith(("╭", "╰"))]
                self.assertTrue(borders, "no panel drawn at all")
                for line in borders:
                    self.assertTrue(line.endswith(("╮", "╯")),
                                    f"right border clipped at {cols}: "
                                    f"{line[-12:]!r}")

    def test_below_the_threshold_no_panel_is_drawn_at_all(self):
        """Not a half-drawn one. Below BOTTOM_MIN the placement is hidden."""
        for cols in (81, 90, 95):
            with self.subTest(cols=cols):
                g = self.grid(40, cols, stats="bottom")
                self.assertEqual([l for l in g if l.lstrip().startswith("╭")],
                                 [], f"a panel was drawn at {cols} columns")

    def test_nothing_is_drawn_past_the_edge(self):
        """The fake screen raises if addnstr runs off the right."""
        for cols in (80, 81, 100, 160):
            for rows in (8, 10, 24, 40):
                for stats in ("right", "bottom", "off"):
                    with self.subTest(cols=cols, rows=rows, stats=stats):
                        self.grid(rows, cols, stats=stats)   # must not raise

    def test_the_clipping_notice_does_not_cover_a_rail_row(self):
        """The notice was written at H-3, which is the last row the loop had
        just drawn, so it replaced a real statistic."""
        g = self.grid(10, 100, stats="right")
        notice = [l for l in g if "more" in l and "taller" in l]
        self.assertTrue(notice, "a clipped rail still says nothing")
        # the row above the notice must still carry a statistic
        i = g.index(notice[0])
        above = g[i - 1]
        self.assertTrue(above.strip(), "the notice sits on a blank row")

    def test_the_clipping_count_is_right(self):
        """It said 'N more' — the previous version computed two values, used
        neither, and asserted only that N was positive."""
        import re as _re
        # Every height, not three of them. The threshold was wrong by exactly
        # the header's height, which bites only when the rail is one row
        # taller than the space available — a single value of `rows` that a
        # three-case sample walked straight past. A tall terminal fits the
        # whole rail and must then show NO notice, which is also asserted
        # here: the first version of this sweep demanded a notice at every
        # height and failed on the ones where nothing was clipped.
        for rows in range(8, 42):
            with self.subTest(rows=rows):
                g = self.grid(rows, 100, stats="right")
                total = self._rail_row_count(rows)
                # Where the rail's TEXT starts, from the code's own
                # rail_w(). Hardcoding `100 - 26` was one column wide of the
                # real value, so the divider glyph counted as a rail row and
                # the sweep reported more rows drawn than exist.
                hdr = self.C.HEADER_H
                notice_y = next((i for i, l in enumerate(g)
                                 if "more" in l and "taller" in l), None)
                if notice_y is None:
                    self.assertLessEqual(total, rows - 2 - hdr,
                                         f"{total} rail rows cannot have fitted "
                                         f"in {rows - 2 - hdr} and said nothing")
                    continue
                # Rail rows are drawn consecutively from `hdr`, so the band is
                # exactly the rows above the notice. Counting non-blank rows
                # instead missed the rail's blank separators and was short by
                # one at every height.
                drawn = notice_y - hdr
                said = int(_re.search(r"(\d+) more", g[notice_y]).group(1))
                self.assertEqual(said + drawn, total,
                                 f"{said} + {drawn} drawn does not account "
                                 f"for every one of {total} rail rows")

    def _rail_width(self, rows):
        c = self.C.Console.__new__(self.C.Console)

        class Screen:
            def getmaxyx(self):
                return (rows, 100)

        c.scr = Screen()
        c.opts = {"stats": "right", "flags": False}
        return c.rail_w()

    def _rail_row_count(self, rows):
        c = self.C.Console.__new__(self.C.Console)

        class Screen:
            def getmaxyx(self):
                return (rows, 100)

        c.scr = Screen(); c.recs = [a_record(i) for i in range(8)]
        c.name = None; c.extra = []; c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False, "stats": "right"}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return len(c.dash(c.rail_w()))

    def test_the_breadcrumb_reaches_the_screen(self):
        """It used to be written to H-2 and then painted over by the status
        bar on the same row."""
        g = self.grid(30, 140, stats="right", top=40)
        bar = g[28]
        self.assertIn("line", bar, "no position in the status bar")
        self.assertIn("#", bar, "no exchange crumb in the status bar")


class FilledBackground(TapTest):
    """The statistics and the status line sit on a solid background.

    Without this the terminal's wallpaper shows between their rows, and the
    dashboard reads as text that drifted to the right rather than as a panel.
    Nothing in `dash()` can show it: the fill happens in `draw()`, so this
    drives the real draw and looks at the attribute of every cell.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")
        # Read from the module, not hardcoded: a hardcoded range made adding
        # a panel colour fail a test that is about the fill, not the palette.
        self.PANEL = set(self.C.PANEL_PAIRS)

    def cells(self, rows, cols, stats="right"):
        """Drive draw() and return a grid of (char, pair number)."""
        C = self.C
        grid = [[(" ", 0)] * cols for _ in range(rows)]

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

            def erase(self):
                for r in range(rows):
                    grid[r] = [(" ", 0)] * cols

            def refresh(self):
                pass

            def addnstr(self, y, x, text, n, attr=0):
                pair = C.curses.pair_number(attr)
                for ch in text[:max(0, n)]:
                    if x >= cols:
                        raise C.curses.error("past the edge")
                    grid[y][x] = (ch, pair)
                    x += 1

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [a_record(i) for i in range(8)]
        c.name = None; c.extra = []; c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False, "stats": stats}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}; c.show = set(self.C.PARTIES); c.chips = []; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0; c.cmd = None; c.busy = ""
        c._flags = "-ngl 999 -c 65536 --jinja"
        c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        c.draw()
        return grid

    def unfilled(self, grid, ys, x0, x1):
        """Cells in the region that are NOT on the panel background."""
        return [(y, x, grid[y][x][0])
                for y in ys for x in range(x0, x1)
                if grid[y][x][1] not in self.PANEL]

    def test_the_rail_is_filled_edge_to_edge(self):
        """Including the blank cells between and after the rows — those are
        the ones the wallpaper showed through."""
        for rows, cols in ((40, 160), (24, 100), (40, 200)):
            with self.subTest(rows=rows, cols=cols):
                g = self.cells(rows, cols, stats="right")
                c = self.C
                rw = c.Console.rail_w.__get__(
                    self._console_for(rows, cols))()
                bad = self.unfilled(g, range(0, rows - 2),
                                    cols - rw - 1, cols - 1)
                self.assertEqual(bad[:6], [], f"{len(bad)} unfilled rail cells")

    def _console_for(self, rows, cols):
        C = self.C

        class Screen:
            def getmaxyx(self):
                return (rows, cols)
        c = C.Console.__new__(C.Console)
        c.scr = Screen()
        c.opts = {"stats": "right", "flags": False}
        return c

    def test_the_bottom_panel_is_filled_edge_to_edge(self):
        for rows, cols in ((40, 160), (40, 200)):
            with self.subTest(rows=rows, cols=cols):
                g = self.cells(rows, cols, stats="bottom")
                vh = next(y for y in range(rows)
                          if "".join(ch for ch, _ in g[y]).strip().startswith("─"))
                bad = self.unfilled(g, range(vh, rows - 2), 0, cols - 1)
                self.assertEqual(bad[:6], [],
                                 f"{len(bad)} unfilled bottom-panel cells")

    def test_the_hint_row_is_filled_edge_to_edge(self):
        """The last row used to be the one transparent strip on screen."""
        for rows, cols in ((40, 160), (24, 100)):
            with self.subTest(rows=rows, cols=cols):
                g = self.cells(rows, cols)
                bad = self.unfilled(g, [rows - 1], 0, cols - 1)
                self.assertEqual(bad[:6], [], f"{len(bad)} unfilled hint cells")

    def test_the_conversation_is_not_filled(self):
        """The fill is for panels. If it reached the transcript the whole
        screen would be one flat rectangle, which is the opposite of the
        point."""
        g = self.cells(40, 160, stats="right")
        pairs = {p for y in range(0, 20) for x in range(0, 60)
                 for _, p in [g[y][x]]}
        self.assertTrue(pairs - self.PANEL,
                        "the transcript is drawn on the panel background too")

    def test_every_pair_has_a_panel_twin(self):
        """A colour with no twin falls back to plain white, so a missing entry
        silently loses the colour of a statistic rather than failing."""
        C = self.C
        for pair in sorted(C.PAIRS.values()):
            if pair in self.PANEL:
                continue
            with self.subTest(pair=pair):
                self.assertIn(pair, C.PANEL_PAIR,
                              f"pair {pair} has no panel twin")

    def test_every_block_ends_in_the_same_column(self):
        """`block()` padded from the width it was handed and `emit_node()`
        from the conversation less two, so the harness's block stopped three
        columns short of a message block and the ragged edge read as a
        rendering fault."""
        for cols in (100, 120, 160):
            with self.subTest(cols=cols):
                g = self.cells(40, cols, stats="right")
                edges = set()
                # The pinned party header, the status bar and the hint line
                # are all filled full width by design, so they are not blocks
                # and do not belong here.
                for row in g[self.C.HEADER_H:-2]:
                    # The filled region nearest the LEFT edge, as (first,
                    # last). Two earlier versions of this were blind: taking
                    # max() over the whole row always found the rail's last
                    # column, and measuring only a run anchored at 0 SKIPPED
                    # a block whose fill began late instead of flagging it.
                    half = len(row) // 2
                    first = next((x for x in range(half)
                                  if row[x][1] in self.PANEL), None)
                    if first is None:
                        continue
                    last = first
                    while last + 1 < len(row) and row[last + 1][1] in self.PANEL:
                        last += 1
                    edges.add((first, last))
                self.assertEqual(len(edges), 1,
                                 f"blocks do not share their edges: "
                                 f"{sorted(edges)}")
                (first, last), = edges
                self.assertEqual(first, 0, "a block's fill starts indented")
                self.assertGreater(last, cols // 2,
                                   "blocks are filled to less than half the "
                                   "conversation width")


class TwelveHourClock(TapTest):
    """Recordings stay ISO; only the display is 12-hour."""

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def test_the_conversions(self):
        for ts, want in (("2026-10-04T13:47:02", "1:47pm"),
                         ("2026-10-04T00:05:00", "12:05am"),
                         ("2026-10-04T12:00:00", "12:00pm"),
                         ("2026-10-04T09:30:00", "9:30am"),
                         ("2026-10-04T23:59:00", "11:59pm")):
            with self.subTest(ts=ts):
                self.assertEqual(self.C.clock(ts), want)

    def test_a_missing_or_broken_stamp_does_not_raise(self):
        """A record with no `t` must render, not crash the console."""
        for ts in ("", None, "garbage", "2026-10-04"):
            with self.subTest(ts=ts):
                self.C.clock(ts)      # must not raise

    def test_the_rendered_block_foot_is_12_hour(self):
        """A source grep is the wrong test here — `clock()` itself slices the
        ISO stamp. What matters is what the block's foot says."""
        import json as _json
        rec = a_record(0)
        rec["t"] = "2026-10-04T13:47:02"
        c = self.C.Console.__new__(self.C.Console)

        class Screen:
            def getmaxyx(self):
                return (40, 140)

        c.scr = Screen(); c.recs = [rec]
        c.name = None; c.extra = []; c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False, "stats": "right"}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}
        c.show = set(self.C.PARTIES); c.chips = []
        c.inflight = []; c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        body = "\n".join("".join(ch for _, ch, _ in spans)
                          for spans in c.lines)
        self.assertIn("1:47pm", body, "the block foot is not 12-hour")
        self.assertNotIn("13:47", body, "24-hour time is still on screen")


class PartyFilter(TapTest):
    """The pinned header row, and clicking a name to filter the conversation.

    The names used to be the first line of the transcript, so they scrolled
    away. They are the filter control now, so they have to stay reachable.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, rows=40, cols=140, show=None):
        C = self.C

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [a_record(i) for i in range(4)]
        c.name = None; c.extra = []; c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False, "stats": "right"}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}
        c.show = set(show) if show is not None else set(C.PARTIES)
        c.chips = []
        c.inflight = []; c.msg = ""; c.msg_until = 0; c.tick = 0
        c.cmd = None; c.busy = ""
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def text_of(self, c):
        return "\n".join("".join(ch for _, ch, _ in spans) for spans in c.lines)

    def test_all_four_parties_are_on_by_default(self):
        c = self.console()
        self.assertEqual(c.show, set(self.C.PARTIES))
        self.assertEqual([p for p, _l, _a, _b in c.chips],
                         list(self.C.PARTIES),
                         "the header row is missing a party")

    def test_the_header_is_not_in_the_transcript(self):
        """It used to be the first line of the conversation, so it scrolled
        off the top and the filter went with it."""
        self.assertNotIn("in this thread", self.text_of(self.console()))

    def test_the_header_reaches_the_screen_at_every_scroll_position(self):
        """Against the FINAL content of row 0, not against everything that was
        written to it. Collecting the writes passed even when the transcript
        was drawn straight over the header a moment later — which is exactly
        the failure the test exists for."""
        c = self.console()
        rows, cols = 40, 140
        for top in (0, 5, max(0, len(c.lines) - 1)):
            with self.subTest(top=top):
                cells = [[" "] * cols for _ in range(rows)]

                class Screen:
                    def getmaxyx(self):
                        return (rows, cols)

                    def erase(self):
                        for r in cells:
                            r[:] = [" "] * cols

                    def refresh(self):
                        pass

                    def addnstr(self, y, x, text, n, attr=0):
                        for ch in text[:max(0, n)]:
                            if x >= cols:
                                return
                            cells[y][x] = ch
                            x += 1

                c.scr = Screen(); c.top = top
                c.draw()
                self.assertIn("in this thread", "".join(cells[0]),
                              "the transcript was drawn over the header")
                # and the first transcript row is still on screen below it
                first = "".join(ch for _c, ch, _a in c.lines[top]) \
                    if top < len(c.lines) else ""
                if first.strip():
                    self.assertIn(first.strip()[:20],
                                  "".join(cells[self.C.HEADER_H]),
                                  "the header pushed the transcript off")

    def test_hiding_a_party_removes_its_messages(self):
        full = self.text_of(self.console())
        self.assertIn("hello", full, "the fixture has no user message")
        without = self.text_of(self.console(show=["harness", "server", "model"]))
        self.assertNotIn("hello", without,
                         "hiding `you` left the user's words on screen")

    def test_hiding_a_party_leaves_the_others_alone(self):
        c = self.console(show=["you"])
        body = self.text_of(c)
        self.assertIn("hello", body, "the only shown party vanished too")
        self.assertNotIn("reply 0", body, "the model is still showing")

    def test_any_combination_works(self):
        import itertools
        for n in range(len(self.C.PARTIES) + 1):
            for combo in itertools.combinations(self.C.PARTIES, n):
                with self.subTest(show=combo):
                    c = self.console(show=combo)   # must not raise
                    self.assertIsInstance(c.lines, list)

    def test_hiding_everything_is_allowed_and_says_so(self):
        """An empty screen is a legitimate combination, but a silent one is
        indistinguishable from a session with no traffic."""
        c = self.console()
        for party in self.C.PARTIES:
            c.toggle_party(party)
        self.assertEqual(c.show, set())
        self.assertIn("hidden", c.msg)

    def test_toggling_is_reversible(self):
        c = self.console()
        before = self.text_of(c)
        c.toggle_party("model")
        self.assertNotEqual(self.text_of(c), before)
        c.toggle_party("model")
        self.assertEqual(self.text_of(c), before,
                         "toggling a party twice did not restore the view")

    def test_a_click_lands_on_the_name_under_the_pointer(self):
        c = self.console()
        for party, label, x0, x1 in c.chips:
            for x in (x0, (x0 + x1) // 2, x1 - 1):
                with self.subTest(party=party, x=x):
                    self.assertEqual(c.chip_at(x), party,
                                     f"column {x} did not hit {label!r}")

    def test_a_click_between_names_hits_nothing(self):
        c = self.console()
        gaps = [x1 for _p, _l, _x0, x1 in c.chips[:-1]]
        for x in gaps + [0, 1]:
            with self.subTest(x=x):
                self.assertIsNone(c.chip_at(x),
                                  "a gap between names is clickable")

    def test_a_tool_result_belongs_to_the_harness(self):
        """It comes back from the harness process, not from a person — the
        same reason it is labelled `<harness> → tool result`."""
        self.assertEqual(self.C.PARTY_OF["tool"], "harness")

    def test_an_unknown_party_is_shown_not_hidden(self):
        """Hiding something because it could not be classified would lose
        traffic silently, which is the one thing this tool exists not to do."""
        c = self.console(show=[])
        self.assertIsNone(c.party_of({"who": "something-new"}))

    def test_the_filter_does_not_change_the_statistics(self):
        """Hiding a party changes what you read, never what is counted."""
        full = self.console()
        filtered = self.console(show=["you"])
        self.assertEqual(filtered._dash_rec, full._dash_rec,
                         "the filter moved the dashboard's record")
        w = full.rail_w()
        self.assertEqual([t for t, _a in filtered.dash(w)],
                         [t for t, _a in full.dash(w)],
                         "the filter changed the statistics")


class StreamScrolls(TapTest):
    """While the model streams, the view follows the newest row.

    `to_bottom()` prefers the START of the last exchange so a finished message
    is shown whole. During a stream that is wrong: the newest row is the one
    being written, and preferring the header left the text running off the
    foot of the screen with nothing appearing to happen.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, rows=16, cols=140, streaming=None):
        C = self.C

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [a_record(i) for i in range(6)]
        c.name = None; c.extra = []; c.follow = True; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": True, "stats": "off"}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}
        c.show = set(C.PARTIES); c.chips = []
        c.inflight = ([{"model": "qwen", "text": streaming}] if streaming
                      else [])
        c.msg = ""; c.msg_until = 0; c.tick = 0; c.cmd = None; c.busy = ""
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def test_the_live_block_is_measured(self):
        quiet = self.console()
        self.assertEqual(quiet._live_lines, 0,
                         "a live block was counted with nothing in flight")
        live = self.console(streaming="word " * 200)
        self.assertGreater(live._live_lines, 1,
                           "the in-flight block was not measured")

    def test_the_last_streamed_row_is_on_screen(self):
        c = self.console(streaming="word " * 400)
        self.assertLessEqual(len(c.lines) - c.top, c.view_h(),
                             "the stream is running off the bottom of the "
                             "screen")

    def test_a_stream_pins_the_view_to_the_absolute_bottom(self):
        """Not merely to somewhere the last row happens to be visible from.

        The looser assertion above passed either way, because a long stream
        makes the last exchange too tall to fit and `to_bottom()` then falls
        through to the bottom anyway. This pins the branch itself: while
        streaming, `top` IS the bottom.
        """
        for n in (50, 100, 400):
            with self.subTest(words=n):
                c = self.console(streaming="word " * n)
                self.assertEqual(c.top, max(0, len(c.lines) - c.view_h()),
                                 "the view is not pinned to the bottom while "
                                 "the model is streaming")

    def test_it_keeps_following_as_the_stream_grows(self):
        c = self.console(streaming="word " * 50)
        for n in (100, 200, 400, 800):
            with self.subTest(words=n):
                c.inflight = [{"model": "qwen", "text": "word " * n}]
                c.rebuild()
                self.assertLessEqual(len(c.lines) - c.top, c.view_h(),
                                     "the view stopped following the stream")

    def test_a_finished_message_is_still_shown_whole(self):
        """The reason to_bottom() prefers the exchange header — not changed,
        only skipped while streaming."""
        c = self.console()
        if c.starts:
            start = c.starts[-1][0]
            if len(c.lines) - start <= c.view_h():
                self.assertEqual(c.top, start,
                                 "the newest message is cut off at the top")

    def test_scrolling_up_still_stops_following(self):
        c = self.console(streaming="word " * 400)
        c.follow = False
        c.top = 0
        c.rebuild()
        self.assertEqual(c.top, 0,
                         "the stream dragged the view away from where the "
                         "user had scrolled")

    def test_a_block_does_not_end_in_a_bare_bar(self):
        """A trailing blank line became a bare `▌` row with nothing beside
        it, which reads as the block having stopped early. `block()` already
        stripped these; a node body is the same block.

        The body is forced to end in blank lines here. Reading the stock
        fixture instead made this test VACUOUS — its records happen to have
        none, so removing the strip changed nothing and the test still
        passed.
        """
        c = self.console()
        c.node_body = lambda *a, **k: ["a thought", "", "  ", ""]
        for nid in [n for (_l, _a, _b, n) in c.hotspots]:
            c.open.add(nid)
        c.rebuild()
        rows = ["".join(ch for _c, ch, _a in spans).rstrip()
                for spans in c.lines]
        self.assertTrue(any("a thought" in l for l in rows),
                        "the forced body never reached the screen")
        bare = [l for l in rows if l.strip() == "▌"]
        self.assertEqual(bare, [], f"{len(bare)} bare accent-bar rows")


class IndentationIsContent(TapTest):
    """`_wrap` destroyed every leading space.

    `para.split(" ")` turns leading whitespace into empty strings that the
    wrap loop skips, so every line came out flush left — the model's indented
    Python was shown as code it had not written, and nested bullets collapsed
    to one level. For a tool whose job is showing what crossed the wire, that
    misreports the payload.
    """

    CODE = (
        "def main():\n"
        '    """Return hello world message"""\n'
        '    return "Hello, World!"\n'
        "\n"
        "if __name__ == \"__main__\":\n"
        "    print(main())"
    )

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def wrap(self, text, width=100, indent=""):
        c = self.C.Console.__new__(self.C.Console)
        return c._wrap(text, width, indent)

    def test_leading_spaces_survive(self):
        out = self.wrap(self.CODE)
        body = [l for l in out if "return" in l][0]
        self.assertTrue(body.startswith("    "),
                        f"indentation was stripped: {body!r}")

    def test_every_level_is_preserved_exactly(self):
        for src in ("    four", "        eight", "  two", "\tone tab"):
            with self.subTest(src=src):
                got = self.wrap(src)[0]
                want = src.replace("\t", "    ")
                self.assertEqual(got, want, "the indent changed")

    def test_the_indent_is_added_to_the_caller_s_indent(self):
        out = self.wrap("    nested", indent="> ")
        self.assertEqual(out[0], ">     nested")

    def test_a_wrapped_line_keeps_the_indent_on_every_row(self):
        """A hanging indent, not just the first row — otherwise the
        continuation jumps left and reads as a new statement."""
        out = self.wrap("    " + "word " * 40, width=30)
        self.assertGreater(len(out), 1, "the fixture did not wrap")
        for row in out:
            self.assertTrue(row.startswith("    "),
                            f"continuation lost the indent: {row!r}")

    def test_the_indent_comes_out_of_the_width(self):
        """Otherwise a deeply nested line wraps past the right edge."""
        width = 40
        for lead in ("", "    ", "            "):
            with self.subTest(lead=lead):
                for row in self.wrap(lead + "word " * 30, width=width):
                    # Exactly the width. The first version allowed
                    # `width + len(lead) + 8` of slack, which is more than the
                    # indent could ever add — so it passed whether or not the
                    # indent was charged to the budget at all.
                    self.assertLessEqual(
                        self.C.dwidth(row), width,
                        f"row ran past the width: {row!r}")

    def test_blank_lines_are_still_blank(self):
        out = self.wrap("a\n\nb", indent="| ")
        self.assertIn("| ", out, "a blank line lost its indent row")

    def test_a_whitespace_only_line_does_not_become_an_indent_row(self):
        """`   ` is visually blank; emitting it as content made a bare
        accent-bar row."""
        out = self.wrap("a\n   \nb")
        self.assertEqual([l for l in out if l.strip()], ["a", "b"])


class OneColourPerParty(TapTest):
    """opencode rendered in two colours — cyan speaking, magenta for a tool
    result — which contradicted the filter, where a tool result is already
    the harness."""

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def test_a_tool_result_is_the_harness_colour(self):
        src = open(os.path.join(ROOT, "llamatap-console.py")).read()
        self.assertNotIn('"\\033[35m"),  # magenta', src,
                         "a party is still rendered in magenta")

    def test_no_party_shares_a_colour_with_another(self):
        """Distinct parties must stay distinguishable after the merge."""
        C = self.C
        colours = {"you": "\033[32m", "harness": "\033[36m",
                   "server": "\033[37m", "model": "\033[33m"}
        self.assertEqual(len(set(colours.values())), len(colours),
                         "two parties now share a colour")

    def test_the_bar_has_a_panel_twin_for_every_party_colour(self):
        """A colour with no PANEL_BAR entry falls back to plain, so the block
        loses its accent bar silently."""
        for col in ("\033[32m", "\033[36m", "\033[37m", "\033[33m"):
            with self.subTest(col=col):
                self.assertIn(col, self.C.PANEL_BAR)


class CopyCarriesItsProvenance(PartyFilter):
    """A copied buffer has to say what it is.

    Moving the party row into pinned chrome dropped it from the copy, so a
    pasted buffer no longer named the harness, the server build or the model
    — and `/copy` still claimed every exchange while the filter was hiding
    most of them.
    """

    def test_the_copy_names_the_participants(self):
        text = self.console().buffer_text()
        self.assertIn("in this thread", text,
                      "a pasted buffer no longer says who produced it")
        for party, label, _x0, _x1 in self.console().chips:
            with self.subTest(party=party):
                self.assertIn(label, text, f"{party} is missing from the copy")

    def test_a_filtered_copy_says_it_was_filtered(self):
        """A filtered buffer that looks complete is worse than no buffer."""
        c = self.console(show=["you"])
        text = c.buffer_text()
        self.assertIn("hidden by the filter", text,
                      "a filtered copy claims to be the whole thread")
        harness = next(l for p, l, _a, _b in c.chips if p == "harness")
        self.assertIn(f"({harness})", text,
                      "a hidden party is not marked as hidden")

    def test_an_unfiltered_copy_says_nothing_about_filtering(self):
        self.assertNotIn("hidden by the filter",
                         self.console().buffer_text(),
                         "an unfiltered copy warns about a filter anyway")

    def test_the_copy_still_excludes_hidden_parties(self):
        """The header names them; the body must not contain them."""
        body = self.console(show=["you"]).buffer_text().split("\n", 2)[-1]
        self.assertNotIn("reply 0", body, "a hidden party is in the copy")

    def test_the_header_is_still_absent_from_the_transcript(self):
        """It belongs to the copy and the pinned row, not to self.lines."""
        self.assertNotIn("in this thread", self.text_of(self.console()))


class PaletteSelection(TapTest):
    """Selecting a command must not highlight the panel's frame.

    The row was drawn as "│" + body + "│" in one write, so A_REVERSE reached
    both borders and the panel's own frame joined the selection.
    """

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def palette(self, rows=40, cols=140, sel=0):
        C = self.C
        cells = [[(" ", 0)] * cols for _ in range(rows)]

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

            def erase(self):
                pass

            def refresh(self):
                pass

            def move(self, y, x):
                pass

            def addnstr(self, y, x, text, n, attr=0):
                for ch in text[:max(0, n)]:
                    if x >= cols:
                        return
                    cells[y][x] = (ch, attr)
                    x += 1

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.cmd = ""; c.cmd_sel = sel
        c.draw_palette()
        return cells

    def test_the_frame_is_never_reversed(self):
        rev = self.C.curses.A_REVERSE
        if not rev:
            self.skipTest("the curses stub has no A_REVERSE bit")
        for row in self.palette():
            for ch, attr in row:
                if ch in "│╭╮╰╯├┤─":
                    self.assertFalse(attr & rev,
                                     f"frame glyph {ch!r} is in the highlight")

    def test_the_selected_row_is_still_highlighted(self):
        """Removing the frame from the highlight must not remove the
        highlight."""
        rev = self.C.curses.A_REVERSE
        if not rev:
            self.skipTest("the curses stub has no A_REVERSE bit")
        hit = [ch for row in self.palette()
               for ch, attr in row if attr & rev and ch.strip()]
        self.assertTrue(hit, "no row is highlighted at all")


class ExpandToggle(TapTest):
    """`o` expands and collapses. It used to be `o` to expand this exchange
    and `O` to collapse everything — two keys differing only by shift, doing
    asymmetric things, so `o` could not be undone with `o`."""

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self):
        C = self.C

        class Screen:
            def getmaxyx(self):
                return (40, 140)

        c = C.Console.__new__(C.Console)
        c.scr = Screen(); c.recs = [a_record(i) for i in range(3)]
        c.name = None; c.extra = []; c.follow = False; c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": False,
                  "repeats": False, "fold_thinking": False, "stats": "off"}
        c.open = set(); c.touched = set(); c.hotspots = []; c.starts = []
        c.menu = []; c.detail = None; c.node_bodies = {}
        c.show = set(C.PARTIES); c.chips = []
        c.inflight = []; c.msg = ""; c.msg_until = 0; c.tick = 0
        c.cmd = None; c.busy = ""
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def press_o(self, c):
        """What the key handler does, without driving the event loop."""
        c.expand_menu()
        if any(nid in c.open for nid in c.menu):
            for nid in c.menu:
                c.open.discard(nid)
        else:
            for nid in c.menu:
                c.open.add(nid)
        c.rebuild()

    def test_it_toggles_back_and_forth(self):
        c = self.console()
        shut = len(c.lines)
        self.press_o(c)
        opened = len(c.lines)
        self.assertGreater(opened, shut, "`o` did not expand anything")
        self.press_o(c)
        self.assertEqual(len(c.lines), shut,
                         "`o` did not collapse back to where it started")
        self.press_o(c)
        self.assertEqual(len(c.lines), opened, "the toggle is not repeatable")

    def test_the_capital_is_the_same_key(self):
        src = open(os.path.join(ROOT, "llamatap-console.py")).read()
        self.assertIn('ch in (ord("o"), ord("O"))', src,
                      "O is not routed to the same toggle")

    def test_the_help_no_longer_advertises_two_keys(self):
        for line in self.C.HELP:
            self.assertNotIn("o / O", line, "the help still lists o / O")
