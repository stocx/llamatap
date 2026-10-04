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
        c.menu = []; c.detail = None; c.node_bodies = {}; c.inflight = []
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
        self.assertEqual(rail.view_h(), 38)
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
                self.assertEqual(c.view_h(), rows - 2)


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
        c.menu = []; c.detail = None; c.node_bodies = {}; c.inflight = []
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
        c.starts = []; c.menu = []; c.detail = None; c.node_bodies = {}
        c.inflight = []; c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def test_any_column_on_the_row_hits_the_node(self):
        c = self.console()
        line, c0, c1, nid = c.hotspots[0]
        for x in (0, 1, c0, c0 + 1, c1, c1 + 20, 110):
            with self.subTest(x=x):
                hs = c.hotspot_at(line - c.top, x)
                self.assertIsNotNone(hs, f"column {x} missed the node")
                self.assertEqual(hs[3], nid)

    def test_a_row_with_no_node_is_not_clickable(self):
        c = self.console()
        node_lines = {l for (l, _a, _b, _n) in c.hotspots}
        plain = next(i for i in range(len(c.lines)) if i not in node_lines)
        self.assertIsNone(c.hotspot_at(plain - c.top, 5))

    def test_clicking_toggles(self):
        c = self.console()
        line, _c0, _c1, nid = c.hotspots[0]
        before = len(c.lines)
        c.toggle(c.hotspot_at(line - c.top, 90)[3])
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

    ALLOWED = {"│", "·", "█", "…", "−", "─"}

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
        c.menu = []; c.detail = None; c.node_bodies = {}; c.inflight = []
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
                      if 0x2500 <= ord(ch) <= 0x257F and ch not in self.ALLOWED})
                       
        self.assertEqual(bad, [], f"box-drawing back in the view: {bad}")

    def test_the_accent_bar_is_there(self):
        self.assertIn("│", self.rendered(self.console()),
                      "no accent bar — blocks have nothing marking their party")

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
        c.menu = []; c.detail = None; c.node_bodies = {}; c.inflight = []
        c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        body = self.rendered(c)
        self.assertIn("✅", body, "the model's own emoji were stripped")
        self.assertIn("⚡", body)
