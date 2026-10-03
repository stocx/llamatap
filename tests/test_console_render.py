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


class Dashboard(TapTest):

    def setUp(self):
        super().setUp()
        self.C = load("console")

    def console(self, rows, cols, flags=False):
        C = self.C
        recs = [a_record(i) for i in range(6)]

        class Screen:
            def getmaxyx(self):
                return (rows, cols)

        c = C.Console.__new__(C.Console)
        c.scr = Screen()
        c.recs = recs
        c.name = None
        c.extra = []
        c.follow = True
        c.top = 0
        c.opts = {"rawjson": False, "preamble": False, "flags": flags,
                  "repeats": False}
        c.open = set()
        c.hover = None
        c.hotspots = []
        c.starts = []
        c.menu = []
        c.detail = None
        c.node_bodies = {}
        c.inflight = []
        c.msg = ""
        c.msg_until = 0
        c.tick = 0
        c._flags = "-ngl 999 -c 65536 --jinja -ctk q8_0 -fa on"
        c._flags_started = True
        c._flags_err = ""
        c._flags_src = "running"
        c._flags_age = None
        c._rec_name = {}
        c.rebuild()
        return c

    def test_every_row_is_the_same_cell_width(self):
        for cols in (90, 98, 118, 160, 200):
            for flags in (False, True):
                with self.subTest(cols=cols, flags=flags):
                    c = self.console(40, cols, flags)
                    widths = set()
                    for spans, _ in c.dash(cols):
                        text = ("".join(t for t, _ in spans)
                                if isinstance(spans, list) else spans)
                        widths.add(self.C.dwidth(text))
                    self.assertEqual(len(widths), 1,
                                     f"ragged border: widths {widths}")

    def test_it_never_takes_most_of_the_screen(self):
        """It took 23 of 30 rows once, which made the tool unusable."""
        for rows in (24, 30, 40, 60):
            for flags in (False, True):
                with self.subTest(rows=rows, flags=flags):
                    c = self.console(rows, 140, flags)
                    used = len(c.dash(140))
                    self.assertLessEqual(used, int(rows * 0.5),
                                         f"dashboard took {used} of {rows} rows")
                    self.assertGreaterEqual(c.view_h(), 5,
                                            "no room left for the conversation")

    def test_the_settings_board_obeys_its_toggle(self):
        off = len(self.console(40, 140, flags=False).dash(140))
        on = len(self.console(40, 140, flags=True).dash(140))
        self.assertGreater(on, off,
                           "/flags did not change the dashboard at all")

    def test_the_launch_command_shows_without_the_board(self):
        """One row, always: it is the most informative line in the panel and
        must not be hidden by the 14-row settings board's toggle."""
        c = self.console(40, 140, flags=False)
        text = "\n".join(
            ("".join(t for t, _ in spans) if isinstance(spans, list) else spans)
            for spans, _ in c.dash(140))
        self.assertIn("-ngl 999", text)


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


class HoverHighlight(TapTest):
    """The mouse highlight must mark the target, not the row.

    Applying A_UNDERLINE to the whole row underlined the leading indent too,
    and underlined spaces are a solid horizontal rule running out to the left
    of the label. It read as a divider drawn on purpose — and since the
    highlight expires after 1.5s, as one that came and went at random.
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
        c.open = set(); c.hover = None; c.hover_until = 0; c.hotspots = []
        c.starts = []; c.menu = []; c.detail = None; c.node_bodies = {}
        c.inflight = []; c.msg = ""; c.msg_until = 0; c.tick = 0
        c._flags = ""; c._flags_started = True; c._flags_err = ""
        c._flags_src = "running"; c._flags_age = None; c._rec_name = {}
        c.rebuild()
        return c

    def test_hover_reports_the_clickable_columns(self):
        import time
        c = self.console()
        line, c0, c1, _nid = c.hotspots[0]
        c.hover = (line - c.top, c0 + 1)
        c.hover_until = time.time() + 10
        self.assertEqual(c.hover_line(), (line, c0, c1))

    def test_nothing_is_ever_underlined(self):
        """A colour change was what was asked for. An underline covers the
        leading indent too, and underlined spaces are a solid rule."""
        src = open(os.path.join(ROOT, "llamatap-console.py")).read()
        self.assertNotIn("A_UNDERLINE", src,
                         "the hover underline is back")

    def test_hover_recolours_rather_than_adds_an_attribute(self):
        import time
        c = self.console()
        line, c0, c1, _nid = c.hotspots[0]
        c.hover = (line - c.top, c0 + 1)
        c.hover_until = time.time() + 10
        hl_line, hl_c0, hl_c1 = c.hover_line()
        plus = next(sp for sp in c.lines[line]
                    if sp[1].strip() in ("+", "−"))
        col, chunk, _attr = plus
        self.assertTrue(col + self.C.dwidth(chunk) > hl_c0 and col <= hl_c1,
                        "the + is not inside the highlighted range")

    def test_clicking_clears_the_highlight_immediately(self):
        """It used to be re-armed by the click event itself, so the mark sat
        there for 1.5 s after the click, doing nothing."""
        src = open(os.path.join(ROOT, "llamatap-console.py")).read()
        click = src[src.index("BUTTON1_CLICKED"):]
        click = click[:click.index("elif bstate")]
        self.assertIn("self.hover, self.hover_until = None, 0.0", click,
                      "a click leaves the highlight armed")

    def test_the_indent_is_not_highlighted(self):
        """The whole point: no mark under the leading blanks."""
        import time
        c = self.console()
        line, c0, c1, _nid = c.hotspots[0]
        c.hover = (line - c.top, c0 + 1)
        c.hover_until = time.time() + 10
        hl_line, hl_c0, hl_c1 = c.hover_line()
        lead_col, lead_text, _a = c.lines[line][0]
        self.assertTrue(lead_text.strip() == "",
                        "expected the first span to be the indent")
        highlighted = (lead_col + self.C.dwidth(lead_text) > hl_c0
                       and lead_col <= hl_c1)
        self.assertFalse(highlighted,
                         "the indent is underlined, which draws a rule")

    def test_no_hover_means_no_highlight(self):
        c = self.console()
        self.assertIsNone(c.hover_line())

    def test_a_stale_hover_expires(self):
        import time
        c = self.console()
        line, c0, _c1, _nid = c.hotspots[0]
        c.hover = (line - c.top, c0 + 1)
        c.hover_until = time.time() - 1
        self.assertIsNone(c.hover_line(),
                          "a stale highlight would look permanent")

    def test_hovering_nothing_clickable_highlights_nothing(self):
        import time
        c = self.console()
        c.hover = (0, 0)
        c.hover_until = time.time() + 10
        self.assertIsNone(c.hover_line())
