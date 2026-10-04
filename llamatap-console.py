#!/usr/bin/env python3
"""Split-pane live console: conversation scrolls, a bordered dashboard pins.

    conversation, scrollable          ↑ ↓ PgUp PgDn Home End
    ────────────────────────────
    ╭─ model ──────────────────╮      pinned, real time
    │ context / cache / speed  │
    ╰──────────────────────────╯
     status bar
    / command line                    /clear /replay /raw /help …

Why curses when the plain renderer avoids it: pinning is the requirement, and
nothing can be pinned in a scrolling transcript. The cost is losing the
terminal's own scrollback, so scrolling is implemented here.

The CONVERSATION renderer is not duplicated — llamatap-thread.py's `render` is
called with stdout redirected into a buffer and its ANSI translated into curses
attributes, so every front end shows the same thread. The DASHBOARD is drawn
here instead of being reused, because a box-drawn panel is console chrome and
makes no sense in a pipeable text dump.
"""
import curses, importlib.util, io, json, os, re, shlex, subprocess, sys, threading, time, unicodedata
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("llamatap_thread", "llamatap-thread.py")
CAT = _load("llamatap_catalog", "llamatap-catalog.py")
# One archive implementation, shared with `serve` and the dispatcher. There
# used to be three, which is how they came to disagree about whether `clear`
# archives at all.
ARCHIVE = _load("llamatap_archive", "llamatap-archive.py")


def ask_payload(rec, text):
    """The request body for `/ask`: the captured exchange, continued.

    The captured `req.messages` end BEFORE the assistant reply, which lives in
    `rec.resp`. Appending the new user message straight onto them skipped that
    reply, so "explain your last answer" could not see the answer printed
    immediately above it, and each repeated /ask dropped another one.

    Reconstructs the completed exchange first. A captured turn that ended in a
    tool call has no resolved result — nothing here can run a tool — so the
    model sees the call it made and no output, which is the honest picture.
    """
    req = json.loads(json.dumps(rec.get("req") or {}))   # never edit the record
    msgs = list(req.get("messages") or [])
    turns = T.from_response(rec.get("resp"))
    said = "\n".join(t.get("text") or "" for t in turns
                      if t.get("who") == "model" and t.get("kind") == "")
    calls = [t for t in turns if t.get("kind") == "tool call"]
    if said.strip() or calls:
        assistant = {"role": "assistant", "content": said.strip()}
        if calls:
            assistant["tool_calls"] = [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": c.get("tool") or "",
                              "arguments": c.get("text") or "{}"}}
                for i, c in enumerate(calls)]
        msgs.append(assistant)
    msgs.append({"role": "user", "content": text})
    req["messages"] = msgs
    req["stream"] = False
    return req


def replay_cmd(args, path):
    """The replay command line, carrying the recording the console has open.

    Without the `--file`, `/replay N` in a console opened on an archive read
    and appended to the LIVE recording instead — and so could send a
    completely different exchange to the model.
    """
    cmd = [os.path.join(HERE, "llamatap-replay.py")] + list(args)
    if path and "--file" not in cmd:
        cmd += ["--file", path]
    return cmd


def cat_markers():
    return CAT.CONTENT_MARKERS
# Default locations. Nothing here may name a particular machine or a
# particular person's home directory: this has to work for whoever clones it.
# The dispatcher exports all of these, so these defaults only apply when a
# .py file is run directly.
_STATE = os.environ.get("LLAMATAP_STATE") or os.path.join(
    os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"),
    "llamatap")
# The model server. 127.0.0.1:8080 is llama-server's own default port and
# assumes it runs on this machine, which is the common fresh install.
_UPSTREAM_DEFAULT = "http://127.0.0.1:8080"
REC = os.environ.get("LLAMATAP_REC", os.path.join(_STATE, "llamatap-records.jsonl"))
LOG = os.environ.get("LLAMATAP_LOG", os.path.join(_STATE, "llamatap.log"))
INFLIGHT = os.environ.get("LLAMATAP_INFLIGHT",
                          os.path.join(_STATE, "llamatap-inflight.json"))
# Braille spinners look great in screenshots and render as mush in plenty of
# terminal fonts — it came out as ".:" here. ASCII always works; override with
# LLAMATAP_SPINNER if your font handles something nicer.
SPIN = os.environ.get("LLAMATAP_SPINNER", "⠏⠛⠹⢸⣰⣤⣆⡇")
# A `+` in its own colour is the affordance: it says "there is more here".
C_PLUS, C_DIM, C_BODY, C_OFF = "\033[32m", "\033[36m", "\033[37m", "\033[0m"
RUNS = os.environ.get("LLAMATAP_RUNS") or os.path.join(_STATE, "runs")

ANSI = re.compile(r"\033\[([0-9;]*)m")


_CELLS = {}


def dwidth(text):
    """Display width in terminal cells.

    Emoji and CJK occupy TWO cells but count as one character in len(), so
    using len() to track column positions makes every span after an emoji land
    one cell early — the terminal then overwrites it. That is why `🎓skill`
    rendered as `🎓sill` and why the dashboard border came out ragged.

    Per-character widths are memoised. This is called on every span of every
    line on every rebuild — 59,000 calls and 716,000 unicodedata lookups on a
    14 MB recording, which was 660 ms of a 500 ms frame budget. The set of
    distinct characters in a terminal is tiny and never changes.
    """
    if text.isascii():
        # The overwhelmingly common case. Every ASCII character is one cell,
        # so len() is exact — and this skips 900,000 dict lookups per rebuild.
        return len(text)
    w = 0
    for ch in text:
        cw = _CELLS.get(ch)
        if cw is None:
            o = ord(ch)
            if unicodedata.combining(ch):
                cw = 0
            elif unicodedata.east_asian_width(ch) in ("W", "F") or \
                    0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or \
                    0x1F000 <= o <= 0x1F2FF:
                cw = 2
            else:
                cw = 1
            _CELLS[ch] = cw
        w += cw
    return w
PAIRS = {31: 1, 32: 2, 33: 3, 34: 4, 35: 5, 36: 6, 37: 7, 90: 8, 94: 9,
         # Panel pairs: every foreground above, again on a filled background,
         # so a panel is a solid rectangle instead of letting the terminal's
         # wallpaper show through between and around its text. There has to be
         # one per colour: a statistic drawn in its own colour on the default
         # background punches a hole in the panel it sits on.
         100: 10, 101: 11, 102: 12, 103: 13, 104: 14, 105: 15,
         106: 16, 107: 17, 108: 18, 109: 19}

#: Every pair that carries the filled background. Derived from PAIRS so it
#: cannot drift from it: codes at 100 and above are the panel codes.
PANEL_PAIRS = frozenset(pair for code, pair in PAIRS.items() if code >= 100)

#: The panel foreground, by pair number, for `boot()` to register.
PANEL_FG = ((10, "WHITE"), (11, "GREEN"), (12, "CYAN"), (13, "YELLOW"),
            (14, "MAGENTA"), (15, "WHITE"), (16, "RED"), (17, "BLUE"),
            (18, "WHITE"), (19, "BLUE"))

#: Panel twin of each ordinary colour pair, for a region drawn on the filled
#: background. Two pairs cannot be OR'd together — the pair number lives in
#: the attribute bits — so a colour has to be translated, not combined.
PANEL_PAIR = {0: 10, 1: 16, 2: 11, 3: 13, 4: 17, 5: 14, 6: 12, 7: 10,
              8: 18, 9: 19}

#: Panel twin of each ANSI escape the renderers emit. The same translation as
#: PANEL_PAIR, one layer earlier: `on_panel()` rewrites a row's own escapes,
#: so text inside a block keeps the block's background.
PANEL_CODE = {"\033[31m": "\033[106m", "\033[32m": "\033[101m",
              "\033[33m": "\033[103m", "\033[34m": "\033[107m",
              "\033[35m": "\033[104m", "\033[36m": "\033[102m",
              "\033[37m": "\033[100m", "\033[90m": "\033[108m",
              "\033[94m": "\033[109m"}

#: The four parties the header row filters by, in the order they appear. A
#: tool result maps to the harness: it comes back from the harness process,
#: not from a person, which is the same reason it is labelled
#: "<harness> → tool result" rather than attributed to the user.
PARTIES = ("you", "harness", "server", "model")

#: Rows the pinned header occupies. Named so the three places that have to
#: agree — view_h(), the draw offset and hotspot_at() — cannot drift.
HEADER_H = 1
PARTY_OF = {"you": "you", "harness": "harness", "server": "server",
            "model": "model", "tool": "harness"}

#: The accent bar. A half-block reads as a solid stripe where `│` was a hairline.
BAR = "▌"

#: Body text on the panel.
C_PANEL = "\033[100m"
#: A party's accent bar, drawn on the panel rather than on the wallpaper.
PANEL_BAR = {"\033[32m": "\033[101m",   # you
             "\033[36m": "\033[102m",   # harness
             "\033[33m": "\033[103m",   # model
             "\033[35m": "\033[104m",   # tool
             "\033[37m": "\033[105m"}   # server


def clock(ts):
    """The wall clock from an ISO timestamp, as 12-hour time.

    The recordings keep ISO — sortable and unambiguous — and only the display
    is 12-hour, which is what the user reads.
    """
    hhmm = (ts or "")[-8:-3]
    try:
        h, mm = hhmm.split(":")
        h = int(h)
    except ValueError:
        return hhmm
    return f"{(h % 12) or 12}:{mm}{'am' if h < 12 else 'pm'}"


def on_panel(text, width=0):
    """One row of a panel: every escape swapped for its panel twin, padded.

    Without the swap, a statistic drawn in its own colour resets the
    background to the terminal's default and punches a hole in the block it
    sits inside — which is what `+ system prompt  23,031 chars` did on every
    node row. Without the padding the row stops where its text does and the
    wallpaper resumes mid-block.
    """
    for plain, panel in PANEL_CODE.items():
        text = text.replace(plain, panel)
    # A reset inside the row would end the fill as well as the colour.
    text = text.replace(C_OFF, C_PANEL)
    pad = max(0, width - dwidth(ANSI.sub("", text)))
    return f"{text}{' ' * pad}{C_OFF}"

#: Every statistics label, so a column is sized from the labels that exist
#: rather than from a constant that drifts when one is added. Flag-board
#: labels come from T.FLAG_BOARD at render time and are folded in beside these.
#: Narrowest terminal at which the bottom panel is worth drawing: three
#: columns of 30 cells, two dividers and two borders. Chosen for legibility —
#: the arithmetic alone survives to about 77, where the labels are truncated.
BOTTOM_MIN = 30 * 3 + 2 + 4

_LEFT_LABELS = ("Model", "Quantised", "Runtime", "Slots", "Sampling",
                "Tools Offered", "Cache Reuse", "Prefill", "Decode",
                "Generated", "Statistics", "Note")
_SESSION_LABELS = ("Context", "Requests", "Messages", "Tokens Generated",
                   "Decode Average", "Prefill Average", "Model Time")

HELP = [
    # Keys only. The slash commands come from COMMANDS, which /help prints
    # above these, so listing them here too went stale the moment a command
    # was renamed — this list used to advertise /icons, removed weeks ago.
    "j k  or  ↑ ↓     scroll one line",
    "ctrl-d / ctrl-u  scroll half a screen",
    "ctrl-f / ctrl-b  scroll a screen (also Space, PgDn, PgUp)",
    "gg               jump to the start of the conversation",
    "G                jump to the newest message and follow new ones",
    "click a node row expand or collapse it — anywhere on the line",
    "x                number the expandable nodes, then press 1-9 to toggle",
    "o                expand or collapse everything in this exchange",
    "r                tool-call arguments: raw JSON off the wire vs readable",
    "p                open or close the system prompt",
    "f                show or hide the launch settings in Model Details",
    "t                fold the model's thinking to one line, or unfold it",
    "s                statistics: right (slim), bottom (full), off",
    "y                copy the whole scroll buffer to the clipboard",
    "/                command palette, filters as you type",
    "Esc              cancel — it does NOT quit",
    "q                quit",
]


# The command palette. One table so the list, the matching and the help text
# can never disagree with what the dispatcher actually accepts.
COMMANDS = [
    ("copy",     "[raw]",            "copy the scroll buffer to the clipboard"),
    ("save",     "[path]",           "write the buffer to a file"),
    ("pager",    "",                 "page the buffer in less, where selection works"),
    ("clear",    "",                 "archive this recording, start a fresh one"),
    ("ask",      "<message>",        "ask the model directly, in the harness's captured context"),
    ("replay",   "N [--model M]",    "re-send exchange N, optionally to another model"),
    ("read",     "N",                "jump to exchange N"),
    ("list",     "",                 "print the index into the conversation"),
    ("raw",      "",                 "tool-call arguments: raw JSON off the wire vs readable"),
    ("preamble", "",                 "open or close the system prompt node"),
    ("flags",    "",                 "show or hide the launch settings in Model Details"),
    ("repeats",  "[open|close]",     "show re-sent messages as collapsed lines; open/close them all"),
    ("thinking", "[open|close]",     "fold the model's thinking to one line; open/close them all"),
    ("stats",    "[right|bottom|off]", "statistics: slim on the right, full across the bottom, or hidden"),
    ("name",     "<label>",          "override the detected harness name"),
    ("help",     "",                 "list these commands"),
    ("quit",     "",                 "exit llamatap console"),
]


def match_commands(typed):
    """Commands matching what has been typed so far.

    Prefix matches first — typing `re` should put `read` and `replay` at the
    top — then substring matches on name or description so `arch` still finds
    `clear`.
    """
    t = (typed or "").strip().split(" ")[0].lower()
    if not t:
        return list(COMMANDS)
    pre = [c for c in COMMANDS if c[0].startswith(t)]
    sub = [c for c in COMMANDS if c not in pre and (t in c[0] or t in c[2].lower())]
    return pre + sub

def panelise(attr):
    """The same attribute, drawn on the filled panel background.

    A colour pair number lives inside the attribute bits, so two pairs cannot
    be combined — the ordinary pair has to be swapped for its panel twin.
    """
    mask = getattr(curses, "A_COLOR", 0)
    try:
        n = curses.pair_number(attr) if mask else 0
    except Exception:
        n = 0
    return (attr & ~mask) | curses.color_pair(PANEL_PAIR.get(n, 10))


def clamp_spans(spans, width):
    """Drop whatever lies past `width` CELLS.

    Header rows — the participants roster, a turn label with its stats, a
    node's `+ label   note` — were emitted at whatever width their content
    happened to be. On a 70-column terminal 79 rows ran past the edge, and the
    terminal wrapped them itself, at the wrong column and without the indent,
    which read as the expansion being corrupt rather than as a line being too
    long.
    """
    if width <= 0:
        return spans
    out = []
    for pos, chunk, attr in spans:
        if pos >= width:
            break
        if pos + dwidth(chunk) <= width:
            out.append((pos, chunk, attr))
            continue
        head, _rest = take(chunk, width - pos)
        if head:
            out.append((pos, head, attr))
        break
    return out


def ansi_lines(text):
    out = []
    for raw in text.split("\n"):
        spans, attr, col, pos, last = [], 0, 0, 0, 0
        for m in ANSI.finditer(raw):
            chunk = raw[last:m.start()]
            if chunk:
                spans.append((pos, chunk, attr | (curses.color_pair(col) if col else 0)))
                pos += dwidth(chunk)
            for c in ([int(c) for c in m.group(1).split(";") if c.isdigit()] or [0]):
                if c == 0:
                    attr, col = 0, 0
                elif c == 1:
                    attr |= curses.A_BOLD
                elif c in PAIRS:
                    col = PAIRS[c]
            last = m.end()
        chunk = raw[last:]
        if chunk:
            spans.append((pos, chunk, attr | (curses.color_pair(col) if col else 0)))
        out.append(spans)
    return out


# Incremental tail state, per path: (size_already_read, records_so_far).
# The recording is append-only, so re-reading and re-parsing all of it twice a
# second was pure waste — 91 ms per poll on a 14 MB recording, on top of the
# rebuild, which together saturated the event loop and made the live view look
# frozen while opencode was working.
_TAIL = {}


_BOUNDARY = 256


def _boundary_sig(path, consumed):
    """The bytes immediately before `consumed`, as proof the file still
    contains what we already read.

    A 64-byte check on the FILE HEAD was not enough, and a test caught it
    splicing two recordings together: every record begins with a timestamp, so
    two sessions starting in the same second produce byte-identical openings.
    The question that actually matters is not "does this file start the same
    way" but "is this file still a superset of what I consumed" — so look at
    the boundary, where a rewritten file differs.
    """
    if consumed <= 0:
        return b""
    start = max(0, consumed - _BOUNDARY)
    try:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(consumed - start)
    except OSError:
        return b""


def load_records(path, incremental=True):
    """Every record in the recording, re-reading only what was appended.

    Falls back to a full read whenever the file is not a pure append of what
    was seen before — a new recording after `serve` archived the old one, or a
    `--file` switch. Shrinking or a changed prefix both mean "different file".
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        _TAIL.pop(path, None)
        return []
    seen, recs, sig = (_TAIL.get(path, (0, None, b"")) if incremental
                       else (0, None, b""))
    if recs is None or size < seen or sig != _boundary_sig(path, seen):
        # Shrunk, or the content at the boundary changed: a different file at
        # the same path. `serve` archives the recording and starts a fresh
        # one, so this happens in normal use and must never splice two
        # recordings into one conversation.
        seen, recs = 0, []
    if size == seen:
        return list(recs)
    out = list(recs)
    try:
        # Binary, because the offset is a byte count: seeking a text stream to
        # an arbitrary byte offset can land mid-character and misdecode.
        with open(path, "rb") as f:
            f.seek(seen)
            data = f.read()
            # A trailing partial line means the proxy is mid-write; leave it
            # for the next poll rather than parsing half a record.
            cut = data.rfind(b"\n")
            if cut < 0:
                return out
            whole = data[:cut + 1]
            consumed = seen + len(whole)
            for raw in whole.splitlines():
                raw = raw.strip()
                if raw:
                    try:
                        out.append(json.loads(raw.decode("utf8", "replace")))
                    except Exception:
                        pass
    except FileNotFoundError:
        _TAIL.pop(path, None)
        return []
    _TAIL[path] = (consumed, out, _boundary_sig(path, consumed))
    return list(out)


def render_to_text(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            fn(*a, **kw)
        except Exception as e:
            print(f"[render error: {e}]")
    return buf.getvalue()


STRIP = re.compile(r"\033\[[0-9;]*m")


def clipboard(text):
    """Copy to the system clipboard.

    A full-screen curses app owns the screen, so dragging the mouse selects
    whatever is painted in that rectangle — the dashboard included — not the
    logical scroll buffer. Selection is therefore not a usable way to get text
    out of this window, and an explicit copy is not a convenience but the only
    route. wl-copy first: this is a Wayland session.
    """
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return cmd[0]
        except Exception:
            continue
    return None


def read_inflight():
    """Requests currently in flight, as published by the proxy."""
    try:
        with open(INFLIGHT) as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def take(text, width):
    """(prefix that fits in `width` cells, remainder).

    Not `fit`: that appends an ellipsis, which is right for a truncated value
    and wrong for a wrapped one — the text continues on the next line, so
    marking it as cut would be a lie. It also miscounted, because the ellipsis
    is a character and the remainder was sliced by character length.
    """
    w = 0
    for i, ch in enumerate(text):
        cw = dwidth(ch)
        if w + cw > width:
            return text[:i], text[i:]
        w += cw
    return text, ""


def fit(text, width):
    """Truncate and pad to exactly `width` terminal CELLS.

    Slicing with [:n] and padding with :<n counts characters, and an emoji is
    one character but two cells — which is why the launch-flag rows ran past
    the dashboard's right border.
    """
    out, w, cut = [], 0, False
    for ch in text or "":
        cw = dwidth(ch)
        if w + cw > width:
            cut = True
            break
        out.append(ch)
        w += cw
    if cut and width >= 3:
        # Say that something was dropped. A silently sliced value reads as a
        # real one — "15,427 of 65" looked like a number, not a truncation.
        while out and w + 1 > width:
            w -= dwidth(out.pop())
        return "".join(out) + "…" + " " * max(0, width - w - 1)
    return "".join(out) + " " * max(0, width - w)


def meter(frac, n):
    """`░` renders as a near-solid block in several terminal fonts, which makes
    an EMPTY bar look full. `·` cannot be misread."""
    frac = max(0.0, min(1.0, frac or 0))
    full = int(frac * n)
    return "█" * full + "·" * (n - full)


class Console:
    def __init__(self, scr, path, name, opts):
        self.scr, self.path, self.name, self.opts = scr, path, name, opts
        self.recs, self.lines, self.extra = [], [], []
        self.top, self.follow, self.msg, self.cmd = 0, True, "", None
        self.busy = ""
        self.inflight, self.tick = [], 0
        self.msg_until = 0.0
        self.hotspots, self.detail, self.menu, self.starts = [], None, [], []
        self._dash_rec, self._dash_bg = None, False
        self.open = set()       # expanded nodes, by (exchange, kind, key)
        self.touched = set()    # nodes the user has opened or closed by hand;
                                # a default never overrides an explicit choice
        self.node_bodies = {}   # precomputed bodies for synthetic nodes
        self.cmd_sel = 0        # highlighted row in the command palette
                                # a rendering artifact rather than a cursor
        #: Parties whose messages are shown. All four by default; clicking a
        #: name in the header row toggles it, so any combination is possible.
        self.show = set(PARTIES)
        #: (party, label, x0, x1) for each clickable name in the header row,
        #: built by rebuild() because only it knows the harness and model
        #: names, and read by draw().
        self.chips = []

    # ------------------------------------------------------------- conversation
    # ------------------------------------------------------------ the tree
    def _fetch_flags(self):
        """Read llama-server's launch flags once, off the draw path.

        props does NOT report them — no -ngl, -c, -ctk, -fa, no draft model —
        and those decide speed and whether a long context fits. Sourced by
        T.launch_info(), which prefers llama-swap's /running (the command line
        of the process actually running, no ssh needed) over a disk cache over
        the config file. Runs once, in a thread; the dashboard shows it when it
        arrives.
        """
        if getattr(self, "_flags_started", False) or not self.recs:
            return
        model = None
        for r in reversed(self.recs):
            model = (r.get("req") or {}).get("model")
            if model:
                break
        if not model:
            return
        self._flags_started = True

        def work():
            try:
                cmd, src, age = T.launch_info(model)
            except Exception:
                cmd, src, age = None, "none", None
            self._flags = cmd or ""
            self._flags_src, self._flags_age = src, age
            if not cmd:
                # Say why rather than omitting the row: a blank reads as "this
                # server was launched with no flags", which is never true.
                self._flags_err = ("no launch command — model is not loaded "
                                   "and no cached copy")
        threading.Thread(target=work, daemon=True).start()

    def _wrap(self, text, width, indent):
        """Wrap to `width` CELLS, breaking inside a word when it is too long.

        Breaking only on spaces meant a single unbroken token could not be
        wrapped at all and ran straight off the screen — a 150-character
        skill path in a system prompt overflowed an 90-column terminal by 84
        cells. The terminal then wrapped it itself, at the wrong place and
        without the indent, which looked like the expansion was corrupt.
        """
        width = max(8, width)
        out = []
        for para in (text or "").split("\n"):
            if not para.strip():
                out.append(indent)
                continue
            # Leading whitespace is CONTENT. `para.split(" ")` turns it into
            # empty strings that the loop below skips, so every line came out
            # flush left: the model's indented Python was rendered as code it
            # had not written, and a nested bullet list collapsed to one
            # level. For a tool whose job is showing what crossed the wire,
            # that is the worst kind of defect — it misreports the payload.
            lead = para[:len(para) - len(para.lstrip())].replace("\t", "    ")
            para = para.lstrip()
            ind = indent + lead
            # The indent comes out of the line's budget, or a deeply nested
            # line wraps past the right edge. Floored, so pathological
            # nesting still renders something.
            width_ = max(8, width - dwidth(lead))
            cur = ""
            for word in para.split(" "):
                # A word wider than the whole line can never fit beside
                # anything; break it into width-sized pieces.
                while dwidth(word) > width_:
                    if cur:
                        out.append(ind + cur)
                        cur = ""
                    head, word = take(word, width_)
                    if not head:
                        break
                    out.append(ind + head)
                if not word:
                    continue
                if cur and dwidth(cur) + 1 + dwidth(word) > width_:
                    out.append(ind + cur)
                    cur = word
                elif not cur:
                    cur = word
                else:
                    cur = cur + " " + word
            if cur or not out or out[-1] != ind:
                out.append(ind + cur)
        return out

    def block(self, body, col, indent, width, text_col=None, foot=None):
        """Message text behind a coloured accent bar.

        Replaces a drawn bubble — `╭ ╮ ╰ ╯ ─ │` plus a `╰─▸` tail — whose
        chrome ran to nearly two thousand box characters in one conversation
        and read as cartoonish beside the content. A single `│` in the
        speaker's colour says the same thing: this text belongs to them.

        `foot` is a dim line inside the bar, used for the timestamp, which
        used to occupy a column on the far left of every single row.
        """
        text_col = text_col if text_col is not None else C_BODY
        pad = " " * indent
        # Trailing blank lines in the source text became bare `│` rows with
        # nothing beside them, which read as the block having stopped early.
        lines = self._wrap((body or "").strip("\n"), max(8, width - 2), "")
        while lines and not lines[-1].strip():
            lines.pop()
        barc = PANEL_BAR.get(col, C_PANEL)
        for l in lines + ([foot] if foot else []):
            self.emit(on_panel(f"{barc}{pad}{BAR}{C_PANEL} {l}",
                               self.conv_w()))

    def node_body(self, kind, key, rec, h):
        """The text behind one expandable node."""
        pre = getattr(self, "node_bodies", {}).get((kind, key))
        if pre is not None:
            return pre
        if kind == "preamble":
            return h["preamble"].split("\n")
        if kind == "tool":
            for t in (rec.get("req") or {}).get("tools") or []:
                fn = (t.get("function") or t) if isinstance(t, dict) else {}
                if fn.get("name") == key:
                    return json.dumps(t, indent=2).split("\n")
            return ["(not in this request)"]
        if kind == "marker":
            pre = h["preamble"]
            for lab, icon, needles in CAT.CONTENT_MARKERS:
                if lab != key:
                    continue
                for nd in needles:
                    j = pre.find(nd)
                    if j == -1:
                        continue
                    start = pre.rfind("\n", 0, max(0, j - 200)) + 1
                    end = pre.find("\n", min(len(pre), j + 3000))
                    end = len(pre) if end == -1 else end
                    body = pre[start:end].split("\n")
                    if end < len(pre):
                        body.append(f"… excerpt of {end-start:,} chars; "
                                    f"expand the system prompt for all of it")
                    return body
            return ["(not located)"]
        return []

    def build_chips(self, harness, server, model):
        """(party, label, x0, x1) for each clickable name in the header row.

        Column positions are computed once here rather than in `draw()`: the
        click target and the drawn text then cannot disagree, which is the way
        a hand-placed label and a hand-placed hotspot always eventually do.
        """
        out, x = [], 2 + dwidth("in this thread:") + 2
        for party, label in (("you", "you"), ("harness", harness),
                             ("server", server), ("model", model)):
            out.append((party, label, x, x + dwidth(label)))
            x += dwidth(label) + 3
        return out

    def party_of(self, m):
        """Which header-row filter a message belongs to.

        An unrecognised party returns None and is always shown: hiding
        something because it could not be classified would lose traffic
        silently, which is the one thing this tool exists not to do.
        """
        return PARTY_OF.get(m.get("who"))

    def toggle_party(self, party):
        if party in self.show:
            self.show.discard(party)
        else:
            self.show.add(party)
        if not self.show:
            self.note("every party hidden — click a name to show it again")
        else:
            self.note(" showing: " + ", ".join(p for p in PARTIES
                                               if p in self.show) + " ")
        self.rebuild()

    def chip_at(self, x):
        for party, _label, x0, x1 in self.chips:
            if x0 <= x < x1:
                return party
        return None

    def conv_w(self):
        """The width every row of the conversation is clamped to.

        Also the right edge every filled panel row pads to: `block()` sized
        its fill from the width it was handed and `emit_node()` from the
        conversation less two, so the harness's block and a message block
        ended in different columns and the ragged edge between them read as a
        rendering fault.
        """
        return getattr(self, "_conv_w", None) or self.scr.getmaxyx()[1]

    def emit(self, text):
        for spans in ansi_lines(text):
            self.lines.append(clamp_spans(spans, self.conv_w()))

    def emit_node(self, rec_i, rec, h, kind, key, label, note, depth, last,
                  base=12):
        """One +/- row, then its body if open. Registers the clickable span.

        `base` is the column the parent message's NAME starts at — children
        line up under the speaker, not under the timestamp.
        """
        nid = (rec_i, kind, key)
        open_ = nid in self.open
        sign = "−" if open_ else "+"
        pad = "   " * (depth - 1)
        # Plain indentation, no box-drawing. `├`/`└` drew a tree beside rows
        # that were already indented under their parent, which is the same
        # information twice and the kind of chrome that read as cartoonish.
        branch = "  " if depth > 1 else ""
        bar = getattr(self, "_bar", "")
        # A block built from +nodes gets the same filled panel as one built
        # from prose — otherwise the harness's own block was the only thing on
        # screen with the wallpaper showing through it.
        barc = PANEL_BAR.get(bar, C_PANEL) if bar else ""
        # The fill starts at column 0, as `block()` does. Colouring after the
        # indent left the two leading columns bare, so a node row's panel
        # began two cells right of a message row's.
        lead = f"{barc}{' ' * base}{BAR}{C_PANEL} " if bar else f"{' ' * base}"
        head = f"{lead}{pad}{branch}"
        text = f"{C_PLUS}{sign}{C_PANEL if bar else C_OFF} {label}"
        line_no = len(self.lines)
        row = head + text + (f"   {C_DIM}{note}{C_OFF}" if note else "")
        # A node row inside a block is a panel row: it runs the full width of
        # the block, and its own colours keep the block's background.
        self.emit(on_panel(row, self.conv_w()) if bar else row)
        c0 = dwidth(head)
        self.hotspots.append((line_no, c0, c0 + dwidth(f"{sign} {label}") + 1, nid))
        if open_:
            # Plain indentation, no gutter glyph. A `│ ` ran down the left of
            # every expanded line; it marked nothing the indent did not
            # already show, cost two columns on each of hundreds of lines of
            # prose, and made an opened node look busier than the text it was
            # showing.
            # The conversation's width, not the screen's. Wrapping to the
            # screen and then letting emit() clamp to _conv_w cut the tail off
            # every line — 27 of 120 words lost from an expanded prompt at 120
            # columns with the rail showing.
            avail = self.conv_w()
            # Bounded by the conversation as well as floored: a floor of 20
            # still outran an 18-cell conversation on a 20-column terminal,
            # and half the expanded text was cut.
            # The name gutter and the accent bar come out of the width too;
            # wrapping to `base` alone cut the tail off every expanded line
            # the moment the gutter was introduced.
            w = max(8, min(avail - 2, avail - base - 6 - 3 * depth))
            ind = " " * base + ((barc + BAR + C_PANEL + " ") if bar else "") \
                + "   " * depth + "  "
            body_lines = list(self.node_body(kind, key, rec, h))
            # Trailing blank lines in the source text became bare `▌` rows
            # with nothing beside them, which reads as the block having
            # stopped early. `block()` already strips these; a node body is
            # the same block and needs the same guard.
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            for ln in body_lines:
                for piece in self._wrap(ln, w, ind) or [ind]:
                    self.emit(on_panel(f"{C_BODY}{piece}", avail)
                              if bar else f"{C_BODY}{piece}{C_OFF}")
        return nid

    def rebuild(self, *a, **kw):
        """Timed wrapper. The recording grows all session, so rebuild cost
        grows with it; the live-stream throttle scales off this measurement
        instead of a constant that was true when it was written."""
        t0 = time.time()
        try:
            return self._rebuild(*a, **kw)
        finally:
            self._rb_s = time.time() - t0

    def _rebuild(self):
        """One group chat: four parties, every message once, in order.

        Exchanges are flattened by T.conversation(); this only lays them out.
        A per-exchange header repeated the participants and the setup for every
        turn, which read as a stack of separate transcripts rather than one
        conversation between four parties.
        """
        self.lines, self.hotspots, self.starts = [], [], []
        self.node_bodies = {}
        # Wrap to the space LEFT OF THE RAIL, not the whole screen, or every
        # line runs under it.
        # Bounded by the screen as well as floored, or a terminal under ~42
        # columns gets rows wider than it is, which draw() then clips.
        cols = self.scr.getmaxyx()[1]
        self._conv_w = max(8, min(cols - 1, cols - self.rail_w() - 2))
        self._dash_rec, self._dash_bg = None, False
        self._rec_name = {}
        W = self._conv_w
        self._fetch_flags()
        msgs = T.conversation(self.recs, self.name, self.opts.get("repeats", False))
        P, parties = CAT.PARTY, {}
        for mm in msgs:
            if mm["who"] == "harness":
                parties["harness"] = mm.get("name", "harness")
        model = ""
        for rec in self.recs:
            model = (rec.get("req") or {}).get("model") or model
        if not model:
            # nothing recorded yet — the in-flight request knows the model
            for job in (self.inflight or []):
                model = job.get("model") or model
        build = ""
        try:
            build = (T.props(model) or {}).get("build_info", "")
        except Exception:
            pass

        # The four parties, as a PINNED header row rather than the first two
        # lines of the transcript: it used to scroll away, and the names are
        # now the filter control, so they have to stay reachable.
        self.chips = self.build_chips(
            parties.get("harness") or "harness (not identified yet)",
            f"llama-server {build or '(version unknown)'}",
            model or "model (waiting)")
        self.emit("")

        # One colour per participant, the way a messaging app tints each
        # person's bubbles. Distinct hues, none of them bright-black.
        style = {
            "you":     (P["you"],             "\033[32m"),   # green
            "harness": (P["harness"].rstrip(), "\033[36m"),  # cyan
            "server":  (P["server"].rstrip(),  "\033[37m"),  # white
            "model":   (P["model"],            "\033[33m"),  # yellow
            # A tool result comes back from the HARNESS, so it is the
            # harness's colour. It used to be magenta, which gave opencode
            # two colours on one screen for no reason the reader could see —
            # and contradicted the filter, which already treats a tool result
            # as the harness.
            "tool":    ("",                    "\033[36m"),  # harness cyan
        }

        # Which exchange is background, and what harness each belongs to.
        # Derived from EVERY message, not from `setup` messages: a setup
        # message is only emitted when the preamble changes, so seven identical
        # requests produce one — and the dashboard stayed pinned to the first,
        # reporting its cold cache forever.
        bg_by_rec = {}
        for m in msgs:
            ri = m.get("rec", 0)
            if m.get("kind") == "setup":
                bg_by_rec[ri] = bool(m.get("background"))
            bg_by_rec.setdefault(ri, False)
            if m.get("name"):
                self._rec_name[ri] = (m["name"], bg_by_rec.get(ri, False))
        for ri in range(len(self.recs) - 1, -1, -1):
            if not bg_by_rec.get(ri, False):
                self._dash_rec = self.recs[ri]
                break
        self._dash_bg = bg_by_rec.get(len(self.recs) - 1, False)
        for ri, bg in bg_by_rec.items():
            if ri in self._rec_name:
                self._rec_name[ri] = (self._rec_name[ri][0], bg)

        for mi, m in enumerate(msgs):
            # The header-row filter. Applied here, AFTER bg_by_rec and the
            # dashboard record were derived above from every message, so
            # hiding a party changes what you read and never what is counted.
            party = self.party_of(m)
            if party is not None and party not in self.show:
                continue
            # (first line of this message, which exchange it came from) — the
            # group-chat view has many messages per exchange, so these are not
            # the same index any more.
            self.starts.append((len(self.lines), m.get("rec", 0)))
            icon, col = style.get(m["who"], ("·", ""))
            # Label by party explicitly. The old form fell through to "you" for
            # anything that was not the harness, the server or the model —
            # which meant every TOOL RESULT was attributed to the user. They
            # come back from the harness, not from a person.
            who = {"harness": m.get("name"),
                   "server":  m.get("name"),
                   "model":   model,
                   "tool":    f"{m.get('name')} → tool result",
                   "you":     "you"}.get(m["who"], m["who"])
            kind = m.get("kind") or ""
            # Indent by party, the way a chat client offsets each speaker.
            # The model is a reply to whoever just spoke, so it sits deeper.
            # Computed HERE, before anything reads it: the thinking branch
            # below used to run first and pick up a stale value left by the
            # previous message.
            # Flush left. Indenting each party was a chat-client habit; it
            # cost columns, made the left edge ragged, and the accent bar
            # already says whose block this is.
            pad = 0
            if kind == "thinking":
                # Same party, so the same colour. Thinking is distinguished by
                # the dotted outline and lighter text, not by being a different
                # speaker.
                # "→ thinking" reads the same way as "→ tool result" and
                # "→ write()": the arrow says this is the model doing
                # something, not a different speaker.
                icon, who = "", f"{model} → thinking"
                col = "\033[33m"
            elif kind == "tool call":
                _c, icon = CAT.classify_tool(m.get("tool"))
                # The MODEL asks for a tool call, so it is the model's
                # colour, for the same reason the result is the harness's.
                who, col = f"{model} → {m.get('tool')}()", "\033[33m"
            elif kind == "background":
                who = m.get("name")
            elif kind == "cut off":
                icon, who, col = "", "no reply", "\033[31m"
            elif kind == "setup":
                # No narration. What this request contains is visible in the
                # tree underneath it; describing it in prose as well is noise.
                who = m.get("name")


            if m.get("repeat"):
                n_ = m.get("chars", 0)
                key = f"{mi}"
                self.node_bodies[("repeat", key)] = (m.get("text") or "").split("\n")
                tok_ = n_ // 4
                note = f"~{tok_:,} tokens"
                tps_ = m.get("prefill_tps")
                if tps_:
                    note += f" · ~{tok_ / max(1.0, tps_):.2f}s to re-read at {tps_:,.0f} t/s"
                note += " · cached" if m.get("cached") else " · NOT cached, re-read"
                self.emit_node(m.get("rec", 0), self.recs[m.get("rec", 0)], None,
                               "repeat", key, f"↻ {who}", note, 1, True, base=12)
                continue

            if kind == "thinking":
                # ALWAYS a +/- node, like everything else that can be opened.
                # It used to become one only after /thinking, so a single
                # thought could not be collapsed without first switching a
                # global mode — which is not how any other node behaves.
                # /thinking now just sets the default for all of them.
                #
                # The cost stays visible even when the text is hidden: a
                # 1,370-token think is most of what a turn spent, and hiding
                # that without saying so would make the thread look cheaper
                # than it was.
                key = f"{mi}"
                self.node_bodies[("thinking", key)] = (m.get("text") or "").split("\n")
                tok_ = m.get("tokens") or (m.get("chars", 0) // 4)
                note = f"{tok_:,} tokens" if m.get("tokens") else f"~{tok_:,} tokens"
                if m.get("secs") is not None:
                    tilde = "~" if m.get("secs_estimated") else ""
                    note += f" · {tilde}{m['secs']:.1f}s"
                else:
                    # Say it rather than omitting it. A row showing only tokens
                    # looks like a thought that took no time.
                    note += " · duration unknown"
                nid = (m.get("rec", 0), "thinking", key)
                if nid not in self.touched:
                    # Follow the current default until the user says otherwise.
                    if self.opts.get("fold_thinking"):
                        self.open.discard(nid)
                    else:
                        self.open.add(nid)
                # A name row, like every other block has. Thinking used to be
                # the only thing on screen with no speaker above it, so a
                # thought read as belonging to whoever spoke last — and its
                # cost sat INSIDE the bar, where a tool result puts it beside
                # the name instead.
                lbl = f"  {' ' * pad}{who}"
                gap = " " * max(2, W - dwidth(lbl) - dwidth(note) - 3)
                self.emit(f"\033[33m{lbl}{C_OFF}{gap}{C_DIM}{note}{C_OFF}")
                self._bar = "\033[33m"
                self.emit_node(m.get("rec", 0), self.recs[m.get("rec", 0)], None,
                               "thinking", key, "Thought", "", 1, True,
                               base=2 + pad)
                self._bar = ""
                # The blank line between blocks is emitted at the END of the
                # loop body, which this branch skips — so a thought was the
                # one block with no gap after it.
                self.emit("")
                continue
            n = m.get("chars", 0)
            stat = f"{n:,} chars · ~{n//4:,} tokens"
            if m.get("tokens"):
                stat = f"{m['tokens']:,} tokens"
            if m.get("secs") is not None:
                tilde = "~" if m.get("secs_estimated") else ""
                stat += f" · {tilde}{m['secs']:.1f}s"
                if m.get("tps"):
                    stat += f" · {m['tps']:.0f} t/s"
            elif kind == "setup" and m.get("prefill_ms"):
                stat += (f" · {m['prefill_ms']/1000:.1f}s prefill · "
                         f"{m.get('prefill_tps',0):.0f} t/s")
            # The name starts the line. A timestamp column used to sit to its
            # left on EVERY row, costing ten characters of width to repeat the
            # same minute over and over; it now appears once, dim, at the foot
            # of the block it belongs to.
            lbl = f"  {' ' * pad}{icon}{who}".rstrip() if not icon else \
                  f"  {' ' * pad}{icon} {who}"
            gap = " " * max(2, W - dwidth(lbl) - dwidth(stat) - 3)
            self.emit(f"{col}{lbl}{C_OFF}{gap}{C_DIM}{stat}{C_OFF}")
            # The accent bar sits just in from the name, so the block reads as
            # belonging to the speaker named above it.
            body_col = 2 + pad

            if kind == "setup":
                # The bar belongs to the block, not to the body text. A setup
                # block is built from +nodes rather than prose, so it used to
                # get no bar at all — which is why the first two harness
                # blocks looked different from everything below them.
                self._bar = col
                rec = self.recs[m["rec"]]
                h = T.header_parts(rec, m.get("name"))
                pc = m.get("preamble_chars", 0)
                total = max(1, m.get("chars", 1))
                if pc:
                    self.emit_node(m["rec"], rec, h, "preamble", "system prompt",
                                   "system prompt",
                                   f"{pc:,} chars · ~{pc//4:,} tokens · "
                                   f"{100*pc//total}% of the request", 1, False,
                                   base=body_col)
                    secs_ = m.get("sections") or []
                    pre_text = h["preamble"]
                    for k, sec in enumerate(secs_):
                        key = f"{sec['label']}@{sec['offset']}"
                        self.node_bodies[("section", key)] = \
                            pre_text[sec["offset"]:sec["offset"] + sec["length"]].split("\n")
                        ic = CAT.section_icon(sec["label"])
                        # section_icon is empty now; keep the join tidy rather
                        # than special-casing emptiness inside the catalog.
                        self.emit_node(m["rec"], rec, h, "section", key,
                                       f"{ic} {sec['label'][:40]}".strip(),
                                       f"{sec['length']:,} chars · "
                                       f"{100*sec['length']//max(1,pc)}% of the prompt",
                                       2, k == len(secs_) - 1, base=body_col)
                tc = m.get("tools_chars", 0)
                if tc:
                    nid = self.emit_node(m["rec"], rec, h, "tools", "tools",
                                         f"{CAT.TOOLBOX} {len(m.get('tools') or [])} tools".strip(),
                                         f"{tc:,} chars · ~{tc//4:,} tokens · "
                                         f"{100*tc//total}% of the request", 1, True,
                                         base=body_col)
                    if nid in self.open:
                        tl = m.get("tools") or []
                        for k, (tn, desc, size) in enumerate(tl):
                            _c, ic = CAT.classify_tool(tn)
                            self.emit_node(m["rec"], rec, h, "tool", tn, f"{ic}{tn}".strip(),
                                           f"{size:,} chars · ~{size//4:,} tokens · "
                                           f"{100*size//max(1,tc)}% of the tools",
                                           2, k == len(tl) - 1, base=body_col)
            else:
                body = m.get("text") or ""
                if kind == "tool call":
                    # Bounded by the conversation as well as floored. A floor
                    # of 24 against an 18-cell conversation cut user text,
                    # tool results and replies on a narrow terminal — the same
                    # defect emit_node had, in a different place.
                    avail = max(8, min(W - body_col - 2, W - body_col - 6))
                    if self.opts.get("rawjson"):
                        # Exactly as it crossed the wire: one escaped JSON
                        # string, newlines as \n.
                        for ln in self._wrap(body, avail, " " * body_col):
                            self.emit(f"{C_BODY}{ln}{C_OFF}")
                        self.emit("")
                        continue
                    try:
                        args = json.loads(body)
                        if not isinstance(args, dict):
                            raise ValueError
                    except Exception:
                        for ln in self._wrap(body, avail, " " * body_col):
                            self.emit(f"{C_BODY}{ln}{C_OFF}")
                        self.emit("")
                        continue
                    # Readable: each argument on its own, and a multi-line
                    # value printed as the lines it actually is. A file's
                    # contents are the whole point of a write() call and are
                    # unreadable as one escaped string.
                    pad = " " * body_col
                    for k, v in args.items():
                        if isinstance(v, str) and "\n" in v:
                            self.emit(f"{col}{pad}{k}:{C_OFF}")
                            for ln in v.split("\n"):
                                for piece in (self._wrap(ln, avail - 4, "") or [""]):
                                    self.emit(f"{C_BODY}{pad}  │ {piece}{C_OFF}")
                        else:
                            txt = v if isinstance(v, str) else json.dumps(v)
                            first = True
                            for piece in self._wrap(f"{k}: {txt}", avail, ""):
                                self.emit(f"{col if first else C_BODY}{pad}{piece}{C_OFF}")
                                first = False
                    self.emit("")
                    continue
                avail = max(8, min(W - body_col - 2, W - body_col - 6))
                if kind == "cut off":
                    # Worth recording, not worth alarming about: the harness
                    # abandoned its own background request. One dim line.
                    for ln in self._wrap(body, avail, " " * body_col):
                        self.emit(f"\033[33m{ln}{C_OFF}")
                elif m["who"] in ("you", "model", "harness", "tool"):
                    self.block(body, col, body_col, avail,
                               foot=clock(m.get("t", "")))
                else:
                    for ln in self._wrap(body, avail, " " * body_col):
                        self.emit(f"{C_BODY}{ln}{C_OFF}")
            self._bar = ""
            self.emit("")

        # The reply currently arriving, drawn as a live bubble at the end of
        # the thread. Without it the newest turn is invisible until it
        # finishes, which on a long reasoning reply is twenty seconds of
        # apparently nothing.
        W2 = self.conv_w()
        # How many rows the in-flight block occupies. `to_bottom()` reads it:
        # while the model is streaming the newest row is the one being
        # written, so the view must go to the true bottom rather than to the
        # last completed exchange's header.
        live_from = len(self.lines)
        for job in (self.inflight or []):
            think, text = job.get("thinking") or "", job.get("text") or ""
            if not (think or text):
                continue
            sp = SPIN[self.tick % len(SPIN)]
            mdl = job.get("model") or model
            if think and not text:
                # "(thinking) (thinking)" — the word was in the f-string
                # twice, once from the label and once hardcoded beside it.
                self.emit(f"\033[33m      live      {mdl} → thinking "
                          f"{sp}{C_OFF}")
                self.block(think, "\033[33m", 16, max(24, W2 - 22),
                           text_col=C_BODY)
            if text:
                self.emit(f"\033[33m      live      {mdl} {sp}{C_OFF}")
                self.block(text, "\033[33m", 16, max(24, W2 - 22))
        self._live_lines = len(self.lines) - live_from

        for e in self.extra:
            self.emit(e)
        if self.follow:
            self.to_bottom()

    # ----------------------------------------------------------------- dashboard
    def live_rows(self, inner):
        """Spinner + live token count while a request is in flight.

        Records are only written when an exchange COMPLETES, so without this
        the console looks frozen for the fifty seconds a long reasoning reply
        takes. The proxy publishes progress to a small status file; this reads
        it.
        """
        if not self.inflight:
            return []
        out = []
        sp = SPIN[self.tick % len(SPIN)]
        now = time.time()
        if len(self.inflight) > 1:
            # Concurrency is the finding, not a glitch: opencode fires its
            # background call alongside your request and the two compete for
            # the same GPU. Say so rather than leaving two near-identical
            # lines to be puzzled over.
            out.append((f"  {len(self.inflight)} requests in flight at once — "
                        f"they share the GPU and slow each other down",
                        curses.color_pair(1) | curses.A_BOLD))
        for job in sorted(self.inflight, key=lambda j: -(j.get("req_bytes") or 0)):
            el = now - (job.get("t0") or now)
            state = job.get("state", "…")
            tok = job.get("tokens") or 0
            model = job.get("model") or "?"
            if state == "prefill":
                body = (f"{sp} prefill   {model} · {job.get('req_bytes',0):,} bytes sent · "
                        f"{el:.1f}s waiting for first token")
            else:
                rate = tok / max(0.1, el - (job.get("ttft") or 0))
                body = (f"{sp} decoding  {model} · {tok:,} tokens · {rate:.0f}/sec · {el:.1f}s"
                        + (f" · first token {job['ttft']:.1f}s" if job.get("ttft") else ""))
            out.append((f"  {body[:inner]:<{inner}}", curses.color_pair(3) | curses.A_BOLD))
        return out

    def dash(self, W):
        """Three bordered sections across the full width: what the model IS,
        what this exchange COST, and what the session adds up to.

        Drawn here rather than reused from the text renderer: a boxed panel is
        console chrome and makes no sense in a pipeable dump.
        """
        full = max(16, W - 2)
        if self.stats_where() == "bottom":
            # What draw() can actually show: it writes panel rows from x=1
            # with a limit of W-1-x, so the usable span is columns 1..W-2,
            # i.e. cols-2 cells. A row is `panel columns + 2` once its four
            # borders are counted, hence -4. Earlier arithmetic said -3 and
            # forgot the left offset, so every panel row lost its right
            # border at every width.
            full = self.scr.getmaxyx()[1] - 4
        live = self.live_rows(full - 2)
        if not self.recs:
            return live + [("  waiting for the first request…", curses.color_pair(6))]

        rec = self._dash_rec if self._dash_rec is not None else self.recs[-1]
        bg_last = self._dash_bg
        req = rec.get("req") or {}
        model = req.get("model") or "?"
        meta = rec.get("meta") or {}
        u, tm = meta.get("usage") or {}, meta.get("timings") or {}
        props = T.props(model) if model else {}
        g = props.get("default_generation_settings") or {}
        prm = g.get("params") or {}
        limit = g.get("n_ctx")
        pt = u.get("prompt_tokens") or tm.get("prompt_n")
        ct = u.get("completion_tokens") or tm.get("predicted_n")
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = tm.get("cache_n")

        # Column widths. Session Totals carries the context gauge — the
        # number you actually watch — so it is no longer the runt.
        if self.stats_where() == "bottom":
            # Three panels side by side. Session Totals carries the context
            # gauge, so it is not the runt.
            c1 = max(30, int(full * 0.32))
            c3 = max(30, int(full * 0.30))
            c2 = full - c1 - c3 - 2          # two dividers
            # There is no re-derivation of `full` here. An earlier version
            # reassigned it to c1+c2+c3+2 with a comment claiming the floors
            # made the two disagree — measured across every supported width,
            # they never do, so the line was dead and the comment false. What
            # actually fixed the oversized row was the -4 above.
        else:
            # One column, so every panel lays out at the rail's width.
            c1 = c2 = c3 = full

        cmd = getattr(self, "_flags", "") or ""

        # Sized from the labels that actually exist. `f"{label:<{N}}"` pads
        # and never truncates, so any N below the longest label glues it to
        # its value: "Tokens Generated4,528", "Cache Reuse██████",
        # "Flash Attentionon". An earlier attempt capped N at 16 and enforced
        # the minimum inside kv()/kv3() — which guarded only the rows those
        # helpers build, while three sites pad their own labels and kept
        # gluing from 81 to 94 columns. pad() is now the ONLY place a label
        # is padded.
        _flag_labels = tuple(lbl for lbl, _v, _on in
                             (T.flag_board(cmd) if cmd else ()))
        L = max(dwidth(x) for x in _LEFT_LABELS + _flag_labels) + 1

        def pad(label, width):
            """The one place a statistics label is padded."""
            return f"{label:<{max(width, dwidth(label) + 1)}}"

        def kv(label, value, attr=0):
            return [(pad(label, L), curses.color_pair(6)), (str(value), attr)]

        build = props.get("build_info") or ""
        col1 = [kv("Model", model, curses.A_BOLD | curses.color_pair(3)),
                kv("Quantised", props.get("model_ftype") or "unknown"),
                kv("Runtime", f"llama.cpp {build}" if build
                   else "llama.cpp (version unavailable)"),
                kv("Slots", props.get("total_slots", "unknown")),
                ]
        sampling = ", ".join(
            f"{k.replace('temperature','temp')} "
            f"{round(prm[k], 3) if isinstance(prm.get(k), float) else prm[k]}"
            for k in ("temperature", "top_k", "top_p", "min_p") if k in prm)
        for i, piece in enumerate(self._wrap(sampling or "server defaults",
                                             max(10, c1 - L - 2), "")):
            col1.append(kv("Sampling" if i == 0 else "", piece))
        if cmd and self.opts.get("flags"):
            col1.append([("", 0)])
            # Settings board: what is set, and what is left at its default.
            # A default is information too — "jinja OFF" explains a harness
            # whose tool calls never parse.
            # One setting per line, same label column as the rest of the panel.
            # Two per line saved space but broke the alignment that makes the
            # panel scannable, and the labels no longer fit in 13 cells.
            board = T.flag_board(cmd)
            for label, val, on in board:
                col1.append([(pad(label, L), curses.color_pair(6) if on else 0),
                             (str(val), (curses.color_pair(2) | curses.A_BOLD)
                              if on else curses.color_pair(7))])
        elif getattr(self, "_flags_err", ""):
            col1.append([(self._flags_err, curses.color_pair(3))])

        col2 = []
        # Tools Offered belongs to the turn, not the session: it is what THIS
        # request put on the table. It happens to be constant today, but a
        # harness that drops tools as context fills is exactly the kind of
        # thing this instrument exists to catch.
        col2.append(kv("Tools Offered", f"{len(req.get('tools') or [])}"))
        if cached is not None and pt:
            fr = cached / pt
            bar_c = curses.color_pair(2 if fr > 0.5 else 3)
            col2.append([(pad("Cache Reuse", L), curses.color_pair(6)),
                         (meter(fr, max(6, min(12, c2 - L - 7))),
                          bar_c | curses.A_BOLD),
                         (f" {fr*100:.0f}%", bar_c)])
            col2.append(kv("", f"{cached:,} reused, {pt-cached:,} new"))
        if tm:
            col2.append(kv("Prefill",
                           f"{tm.get('prompt_n', 0):,} tokens at "
                           f"{tm.get('prompt_per_second', 0):,.0f}/sec"))
            col2.append(kv("", f"{tm.get('prompt_ms', 0)/1000:.2f} seconds"))
            col2.append(kv("Decode",
                           f"{tm.get('predicted_n', 0):,} tokens at "
                           f"{tm.get('predicted_per_second', 0):.1f}/sec",
                           curses.color_pair(3)))
            col2.append(kv("", f"{tm.get('predicted_ms', 0)/1000:.2f} seconds"))
        elif ct:
            col2.append(kv("Generated", f"{ct:,} tokens"))
        if not pt and not tm:
            why = ("response exceeded the capture cap"
                   if len(rec.get("resp") or "") > 190000 else "no token stats")
            col2.append(kv("Statistics", why, curses.color_pair(3)))
        if bg_last:
            col2.append([("", 0)])
            col2.append(kv("Note", "newest request was a background call;",
                           curses.color_pair(3)))
            col2.append(kv("", "this is the last real turn", curses.color_pair(3)))

        msgs_ = req.get("messages") or req.get("input") or []
        total_s = sum((r.get("meta") or {}).get("elapsed_ms", 0) for r in self.recs) / 1000
        # Running averages across every request in the recording, weighted by
        # tokens rather than by request — a 3-token reply should not count as
        # much as a 3,000-token one.
        dec_tok = dec_ms = pre_tok = pre_ms = 0
        for r in self.recs:
            t_ = (r.get("meta") or {}).get("timings") or {}
            dec_tok += t_.get("predicted_n") or 0
            dec_ms += t_.get("predicted_ms") or 0
            pre_tok += t_.get("prompt_n") or 0
            pre_ms += t_.get("prompt_ms") or 0
        S = max(dwidth(x) for x in _SESSION_LABELS) + 1
        def kv3(label, value, attr=0):
            return [(pad(label, S), curses.color_pair(6)), (str(value), attr)]

        # The bar has to survive a narrow column; shrink it rather than let
        # it push the percentage off the edge.
        bw = max(6, min(12, c3 - S - 7))
        col3 = []
        if pt and limit:
            fr = pt / limit
            bar_c = curses.color_pair(2 if fr < 0.7 else 3 if fr < 0.9 else 1)
            col3.append([(pad("Context", S), curses.color_pair(6)),
                         (meter(fr, bw), bar_c | curses.A_BOLD),
                         (f" {fr*100:.0f}%", bar_c)])
            col3.append(kv3("", f"{pt:,} of {limit:,} tokens"))
        elif pt:
            col3.append(kv3("Context", f"{pt:,} tokens", curses.A_BOLD))
            col3.append(kv3("", "window size unavailable", curses.color_pair(3)))
        col3 += [kv3("Requests", f"{len(self.recs):,}"),
                 kv3("Messages", f"{len(msgs_):,}")]
        if dec_tok:
            col3.append(kv3("Tokens Generated", f"{dec_tok:,}"))
        if dec_ms:
            col3.append(kv3("Decode Average",
                            f"{dec_tok / (dec_ms/1000):.1f}/sec",
                            curses.color_pair(3) | curses.A_BOLD))
        if pre_ms:
            col3.append(kv3("Prefill Average",
                            f"{pre_tok / (pre_ms/1000):,.0f}/sec",
                            curses.color_pair(6) | curses.A_BOLD))
        col3.append(kv3("Model Time", f"{total_s:.0f} seconds"))

        if self.stats_where() == "bottom":
            return self._dash_panels(live, col1, col2, col3, cmd,
                                     full, c1, c2, c3)
        return self._dash_rail(live, col1, col2, col3, cmd, W)

    #: What the slim right-hand rail carries. The bottom panel carries
    #: everything; the rail is for the few numbers you watch continuously
    #: while a turn is running, and a narrow column cannot hold more without
    #: wrapping every value onto its own line.
    RAIL_KEEP = ("Context", "Cache Reuse", "Prefill", "Decode", "Requests",
                 "Tokens Generated", "Decode Average")

    def _dash_panels(self, live, col1, col2, col3, cmd, full, c1, c2, c3):
        """Three bordered panels across the full width.

        Restored from before the restyle. The bottom of the screen is the one
        place wide enough to show Model Details, Current Request and Session
        Totals SIDE BY SIDE, and that grouping — identity, this turn, the
        whole session — is what makes the numbers scannable at a glance. A
        narrow rail cannot do it, which is why both placements exist.
        """
        def hdr(label, w):
            """Centred panel title. The three panels are distinguished by TIME
            SCALE — configuration, this turn, the whole session — so their
            titles are parallel noun phrases in Title Case."""
            t = f" {label} "
            if dwidth(t) > w:
                # Clamp, do not just stop padding. A title wider than its
                # column made the border row one cell wider than every body
                # row beneath it, and so one cell wider than the screen.
                return fit(t, w)
            pad = w - dwidth(t)
            left = pad // 2
            return "─" * left + t + "─" * (pad - left)

        rows = list(live)
        # Not "latest request" — it deliberately skips background calls, so it
        # is the last turn of YOUR conversation.
        rows.append((f"╭{hdr('Model Details', c1)}┬{hdr('Current Request', c2)}"
                     f"┬{hdr('Session Totals', c3)}╮",
                     curses.A_BOLD | curses.color_pair(6)))
        def spans_of(cell, width, base=0):
            """A column cell -> [(text, attr)] padded to exactly `width`."""
            if cell is None:
                return [(" " * width, 0)]
            if isinstance(cell, str):
                return [(fit(" " + cell, width), base)]
            out, used = [(" ", 0)], 1
            for txt, att in cell:
                room = width - used
                if room <= 0:
                    break
                t = fit(txt, min(room, dwidth(txt)))
                out.append((t, att))
                used += dwidth(t)
            if used < width:
                out.append((" " * (width - used), 0))
            return out

        div = [("│", curses.color_pair(6))]
        body = max(len(col1), len(col2), len(col3))
        # Hard cap: the conversation is what you came to read. Three columns
        # pad to the tallest, so without this one long column (the 14-row
        # settings board) decides how much of the screen is left.
        H = self.scr.getmaxyx()[0]
        room = max(4, int(H * 0.40) - 3)       # minus two borders and a hint
        W = full + 2
        clipped = body - room if body > room else 0
        if clipped:
            body = room
        for i in range(body):
            segs = list(div)
            segs += spans_of(col1[i] if i < len(col1) else None, c1)
            segs += div
            segs += spans_of(col2[i] if i < len(col2) else None, c2)
            segs += div
            segs += spans_of(col3[i] if i < len(col3) else None, c3)
            segs += div
            rows.append((segs, 0))
        if clipped:
            note = (f" {clipped} more rows — widen"
                    + (" or press f to hide settings " if self.opts.get("flags")
                       else " the window "))
            pad = max(0, full - dwidth(note))
            rows.append((f"│{note}{' ' * pad}│", curses.color_pair(3)))
        if cmd:
            short = re.sub(r"\S*llama-server\s*|-m \S+\s*|--alias \S+\s*"
                           r"|--port \S+\s*|--host \S+\s*", "", cmd)
            # A cached command may describe a previous launch, so date it
            # rather than presenting it as the running one.
            age = getattr(self, "_flags_age", None)
            if getattr(self, "_flags_src", "") == "cached" and age:
                mins = age / 60
                stale = (f"  (cached {mins:.0f} min ago)" if mins < 90
                         else f"  (cached {mins/60:.0f} h ago)")
            elif getattr(self, "_flags_src", "") == "config":
                stale = "  (from the config file, not the running process)"
            else:
                stale = ""
            rows.append((f"├{'─' * c1}┴{'─' * c2}┴{'─' * c3}┤", curses.color_pair(6)))
            bar = [("│", curses.color_pair(6))]
            rows.append((bar
                         + [(fit(" $ " + short.strip(),
                                 max(0, full - dwidth(stale))), 0),
                            (stale, curses.color_pair(3))]
                         + bar, 0))
            rows.append((f"╰{'─' * full}╯", curses.color_pair(6)))
        else:
            # Say why it is absent rather than just omitting the row: it needs
            # the ssh agent to read llama-swap's YAML, and a silent gap looks
            # like the model was launched with no flags at all.
            why = getattr(self, "_flags_err", "") or (
                "reading launch flags…" if not getattr(self, "_flags_started", False)
                else "launch command unavailable")
            rows.append((f"├{'─' * c1}┴{'─' * c2}┴{'─' * c3}┤", curses.color_pair(6)))
            rows.append((f"│{fit(' $ ' + why, full)}│", curses.color_pair(3)))
            rows.append((f"╰{'─' * full}╯", curses.color_pair(6)))
        return rows

    def _dash_rail(self, live, col1, col2, col3, cmd, W):
        """A narrow vertical rail: the same cells, filtered and stacked."""
        def heading(text):
            return ([(text, curses.A_BOLD | curses.color_pair(6))], 0)

        rows = list(live)
        # Current turn FIRST. A short window drops rows off the end, and with
        # Session leading, the numbers the rail exists to show — cache reuse,
        # prefill, decode — were the first to go.
        for title, col in (("This turn", col2), ("Session", col3),
                           ("Model", col1)):
            if not col:
                continue
            # Build the section first. A heading with nothing under it is
            # worse than no heading, and RAIL_KEEP holds nothing from the
            # Model column, so "Model" was printing as a bare word.
            section = []
            keeping = False
            for cell in col:
                if isinstance(cell, list) and cell and cell[0][0].strip():
                    keeping = cell[0][0].strip() in self.RAIL_KEEP
                if not keeping:
                    continue
                if cell is None:
                    section.append(("", 0))
                elif isinstance(cell, str):
                    section.append((cell, 0))
                elif not cell[0][0].strip():
                    # A continuation row — blank label, value only. It kept
                    # the label column's padding, which in a rail left the
                    # value stranded sixteen spaces from the edge.
                    body = "".join(t for t, _ in cell).strip()
                    for piece in self._wrap(body, max(8, W - 3), ""):
                        section.append(([("  " + piece, cell[-1][1])], 0))
                elif dwidth("".join(t for t, _ in cell)) <= W - 2:
                    # W-2, not W-1: draw() writes rail rows with a limit of
                    # rw-2, so a row accepted at rw-1 lost its last cell.
                    section.append((list(cell), 0))
                else:
                    # Too wide: label on its own line, value indented under
                    # it. The value spans stay TOGETHER — a meter and its
                    # percentage are one reading, and splitting them put
                    # "█████·" and "87%" on separate rows.
                    label = cell[0][0].rstrip()
                    rest = [(t, a) for t, a in cell[1:] if t.strip()]
                    if label:
                        section.append(([(label, cell[0][1])], 0))
                    if rest and dwidth("".join(t for t, _ in rest)) <= W - 3:
                        section.append(([("  ", 0)] + rest, 0))
                    else:
                        for t, a in rest:
                            for piece in self._wrap(t, max(8, W - 3), ""):
                                section.append(([("  " + piece, a)], 0))
            if not any(r != ("", 0) for r in section):
                continue
            if rows and rows[-1] != ("", 0):
                rows.append(("", 0))
            rows.append(heading(title))
            rows.extend(section)
        return rows

    # -------------------------------------------------------------------- layout
    def stats_where(self):
        """Where the statistics are: "right", "bottom" or "off".

        Two placements because they are good at different things. The bottom
        spans the full width, so it can hold the three panels side by side —
        Model Details / Current Request / Session Totals — which is the
        grouping that makes the numbers scannable. A right rail cannot do
        columns, so it carries only the few you watch continuously, and costs
        screen COLUMNS rather than the ten to twenty rows the bottom panel
        takes out of the conversation.
        """
        w = self.opts.get("stats", "right")
        if w not in ("right", "bottom", "off"):
            # Anything unrecognised means "right". main() once passed the
            # boolean this replaced, and the console launched showing no
            # statistics at all with nothing to say why — recoverable only
            # because the `s` cycle happened to fall through to "right".
            w = "right"
        cols = self.scr.getmaxyx()[1]
        if w == "right" and cols < 80:
            return "off"      # no columns to spare on a narrow terminal
        if w == "bottom" and cols < BOTTOM_MIN:
            # A LEGIBILITY floor, not an arithmetic one. The geometry survives
            # to about 77 columns, but by then the middle panel is ~15 cells
            # and even its LABELS are cut — "Tools Offered…", "Prefill  10 t…"
            # — which is the half worth reading. BOTTOM_MIN is the width at
            # which all three columns reach 30 cells. Below it the placement
            # is hidden and `s` says why.
            return "off"
        return w

    def rail_w(self):
        """Columns the right-hand rail occupies, or 0 when it is elsewhere."""
        if self.stats_where() != "right":
            return 0
        return max(24, min(34, self.scr.getmaxyx()[1] // 4))

    def dash_h(self):
        return len(self.dash(self.scr.getmaxyx()[1]))

    def view_h(self):
        # conversation + status + command line, and the dashboard too when it
        # is at the bottom.
        # -1 more for the pinned party header row at the top.
        h = self.scr.getmaxyx()[0] - 3
        if self.stats_where() == "bottom":
            h -= self.dash_h() + 1        # +1 for the separator rule
        return max(1, h)

    def to_bottom(self):
        """Prefer the START of the last exchange over the absolute bottom.

        Scrolling to the very last line cuts the newest message's own header
        off the top of the view, so you land mid-paragraph with no idea who is
        speaking. If the newest message fits, show it whole.
        """
        vh = self.view_h()
        bottom = max(0, len(self.lines) - vh)
        if getattr(self, "_live_lines", 0):
            # While the model is streaming, the newest line is the one being
            # written — so the view goes to the true bottom. Preferring the
            # last exchange's header instead left the stream running off the
            # foot of the screen with nothing appearing to happen.
            self.top = bottom
            return
        if getattr(self, "starts", None):
            start = self.starts[-1][0]      # (line, exchange) pairs now
            if len(self.lines) - start <= vh:
                self.top = start
                return
        self.top = bottom

    # ---------------------------------------------------------------------- draw
    def draw_palette(self):
        # NOTE: assumes the caller has already painted the frame underneath and
        # has NOT refreshed; this adds the overlay and refreshes once.
        """Centred, filtered command list. Drawn over the middle of the screen
        because a prompt pinned to the last line is easy to miss and leaves no
        room to show what the options are."""
        scr = self.scr
        H, W = scr.getmaxyx()
        matches = match_commands(self.cmd)
        self.cmd_sel = max(0, min(self.cmd_sel, max(0, len(matches) - 1)))
        bw = min(max(46, W - 20), 78)
        rows = min(len(matches), max(3, H - 12))
        bh = rows + 4
        y0 = max(0, (H - bh) // 2)
        x0 = max(0, (W - bw) // 2)

        def put(y, x, text, attr=0):
            try:
                scr.addnstr(y, x, text, max(0, min(bw, W - x)), attr)
            except curses.error:
                pass

        put(y0, x0, "╭" + "─" * (bw - 2) + "╮", curses.color_pair(6))
        typed = f"/{self.cmd}"
        put(y0 + 1, x0, "│ " + typed.ljust(bw - 4) + " │", curses.A_BOLD)
        put(y0 + 2, x0, "├" + "─" * (bw - 2) + "┤", curses.color_pair(6))
        for i in range(rows):
            m = matches[i] if i < len(matches) else None
            if m is None:
                break
            sel = (i == self.cmd_sel)
            name, args, desc = m
            left = f"/{name} {args}".strip()
            body = f" {left:<22}{desc}"
            # Three writes, not one. Drawing "│" + body + "│" as a single
            # string applied A_REVERSE to the borders too, so selecting a row
            # pulled the panel's own frame into the highlight. The frame
            # belongs to the panel, not to the selection.
            put(y0 + 3 + i, x0, "│", curses.color_pair(6))
            put(y0 + 3 + i, x0 + 1, body[:bw - 2].ljust(bw - 2),
                curses.A_REVERSE if sel else 0)
            put(y0 + 3 + i, x0 + bw - 1, "│", curses.color_pair(6))
        if not matches:
            put(y0 + 3, x0, "│" + "  no match".ljust(bw - 2) + "│", curses.color_pair(1))
        put(y0 + 3 + rows, x0, "╰" + "─" * (bw - 2) + "╯", curses.color_pair(6))
        hint = " ↑↓ choose · Tab complete · Enter run · Esc cancel "
        put(y0 + 3 + rows + 1 if y0 + 3 + rows + 1 < H else H - 1,
            x0, hint.center(bw), curses.color_pair(6))
        try:
            scr.move(y0 + 1, x0 + 2 + dwidth(typed))
        except curses.error:
            pass
        curses.curs_set(1)
        scr.refresh()

    def fill(self, y, x, w):
        """Paint `w` cells of row `y` with the panel background.

        The statistics and the status line are panels like a message block:
        without this the terminal's wallpaper shows between their rows, which
        is what made the old dashboard read as text that had drifted rather
        than as a panel.
        """
        if w <= 0:
            return
        try:
            self.scr.addnstr(y, x, " " * w, w, curses.color_pair(10))
        except curses.error:
            pass

    #: Party colour pairs for the header row, on the panel background.
    CHIP_PAIR = {"you": 11, "harness": 12, "server": 10, "model": 13}

    def draw_header(self, W):
        """The pinned party row. Each name is a filter you can click.

        Highlighted means shown. A dimmed name is a party whose messages are
        being withheld — which has to be visible at all times, or a filtered
        view is indistinguishable from a quiet session.
        """
        scr = self.scr
        self.fill(0, 0, W - 1)
        try:
            scr.addnstr(0, 2, "in this thread:", W - 3,
                        curses.color_pair(18))
        except curses.error:
            pass
        for party, label, x0, _x1 in self.chips:
            if x0 >= W - 1:
                break
            on = party in self.show
            attr = curses.color_pair(self.CHIP_PAIR.get(party, 10))
            attr |= curses.A_BOLD if on else curses.A_DIM
            if not on:
                attr = curses.color_pair(18) | curses.A_DIM
            try:
                scr.addnstr(0, x0, label, W - 1 - x0, attr)
            except curses.error:
                pass

    def draw(self):
        scr = self.scr
        scr.erase()
        H, W = scr.getmaxyx()
        vh = self.view_h()

        self.draw_header(W)

        for r in range(vh):
            i = self.top + r
            if i >= len(self.lines):
                break
            for col, chunk, attr in self.lines[i]:
                if col < W - 1:
                    try:
                        scr.addnstr(r + HEADER_H, col, chunk, W - 1 - col, attr)
                    except curses.error:
                        pass

        # separator between the transcript and the dashboard
        # Which exchange are you looking at? Once you scroll past a header the
        # turns below give no clue, and a background request's reply then reads
        # as an answer to your own question.
        crumb = ""
        ri_here = None
        for line, rec_i in (self.starts or []):
            if self.top >= line:
                ri_here = rec_i
            else:
                break
        if ri_here is not None:
            nm, bg = getattr(self, "_rec_name", {}).get(ri_here, (None, False))
            if nm:
                crumb = f" · in #{ri_here + 1} {nm}" + (" ⟨BACKGROUND⟩" if bg else "")
        # The crumb used to be written to row H-2 and then painted over by the
        # status bar on the same row, so which exchange you were looking at —
        # and whether it was a background request — was never visible. It is
        # part of the status bar now.
        self._crumb = crumb

        where = self.stats_where()
        rw = self.rail_w()
        if where == "right" and rw:
            # A divider down the full height, so the rail reads as a column
            # rather than as text that drifted to the right.
            for y in range(HEADER_H, H - 2):
                self.fill(y, W - rw - 1, rw)
                try:
                    scr.addnstr(y, W - rw - 1, "│", 1,
                                panelise(curses.color_pair(6)))
                except curses.error:
                    pass
        if where == "bottom":
            # A rule between the conversation and the panels, as before.
            for y in range(vh + HEADER_H, H - 2):
                self.fill(y, 0, W - 1)
            try:
                scr.addnstr(vh + HEADER_H, 0, "─" * (W - 1), W - 1,
                            panelise(curses.color_pair(6)))
            except curses.error:
                pass
        panel = (self.dash(rw) if where == "right" and rw
                 else self.dash(W) if where == "bottom" else [])
        # Reserve the notice row BEFORE drawing, or it lands on top of the
        # last rail row that was just drawn — which also made the count one
        # short, since the overwritten row was counted as shown.
        limit = H - 2
        # How many rail rows actually FIT, which is the limit less the rows
        # the pinned header takes — the rows start at HEADER_H, not at 0.
        # Comparing against `limit` itself under-triggered by exactly the
        # header's height, and the notice then landed on the last rail row
        # it had just drawn.
        if where == "right" and len(panel) > limit - HEADER_H:
            limit = H - 3
        shown = 0
        for r, (txt, attr) in enumerate(panel):
            y = (r + HEADER_H) if where == "right" else vh + HEADER_H + 1 + r
            if y >= limit:
                break
            shown = r + 1
            if isinstance(txt, list):
                x = (W - rw + 1) if where == "right" else 1
                for seg, a in txt:
                    if x >= W - 1:
                        break
                    try:
                        scr.addnstr(y, x, seg, W - 1 - x,
                                    panelise(a or attr))
                    except curses.error:
                        pass
                    x += dwidth(seg)
            else:
                try:
                    if where == "right":
                        scr.addnstr(y, W - rw + 1, txt, rw - 2,
                                    panelise(attr))
                    else:
                        scr.addnstr(y, 1, txt, W - 2, panelise(attr))
                except curses.error:
                    pass
        if where == "right" and shown < len(panel):
            # Say what was dropped. The bottom panel announces clipping; the
            # rail silently lost rows off the end — and because the sections
            # build Session, This turn, Model in that order, what vanished on
            # a short terminal was the CURRENT TURN, which is the reason the
            # rail exists.
            note = f" {len(panel) - shown} more — taller window "
            try:
                scr.addnstr(H - 3, W - rw + 1, fit(note, rw - 2), rw - 2,
                            panelise(curses.color_pair(3)))
            except curses.error:
                pass

        pos = ("FOLLOWING" if self.follow
               else f"line {self.top+1}/{max(1, len(self.lines))}")
        bar = (f" tool calls {'RAW' if self.opts.get('rawjson') else 'readable'} │ "
               f"repeats {'SHOWN' if self.opts.get('repeats') else 'hidden'} │ "
               f"thinking {'FOLDED' if self.opts.get('fold_thinking') else 'full'} │ "
               f"{pos}{getattr(self, '_crumb', '')} ")
        if self.busy:
            bar += f"│ {self.busy} "
        try:
            scr.addnstr(H - 2, 0, bar.ljust(W - 1), W - 1, curses.A_REVERSE)
        except curses.error:
            pass

        if self.cmd is not None:
            # Draw the palette over the finished frame. Erasing and refreshing
            # first made the whole screen blink and left it half-drawn.
            self.draw_palette()
            return
        else:
            curses.curs_set(0)
            # Ordered by what someone stuck needs first: this line is clipped
            # at the terminal width, and `q quit` used to sit at the very end
            # where it was the first thing lost on an 80-column window.
            hint = self.msg or (" / commands  ·  q quit  ·  jk gg G scroll  ·  "
                                "click a row to expand  ·  x list  ·  o all  ·  "
                                "r raw json  ·  p system prompt  ·  y copy ")
            try:
                scr.addnstr(H - 1, 0, hint.ljust(W - 1), W - 1,
                            panelise(curses.color_pair(6)))
            except curses.error:
                pass
        scr.refresh()


    # ---------------------------------------------------------------- detail
    def rec_at(self, line_idx):
        """Which exchange a buffer line belongs to."""
        if not self.recs or not self.starts:
            return None
        ri = self.starts[0][1]
        for line, rec_i in self.starts:
            if line_idx >= line:
                ri = rec_i
            else:
                break
        return self.recs[ri] if 0 <= ri < len(self.recs) else None

    def hotspot_at(self, y, x):
        """The expandable node on this ROW, if any.

        The column is ignored on purpose: a one-character `+` is a precision
        target, and nothing else on a node's row is clickable, so there is no
        ambiguity about what a click there means. Anywhere on the line works.
        """
        # -1 for the pinned header row: screen row 1 is transcript row
        # self.top. Without it every click opened the node ABOVE the one
        # under the pointer.
        li = self.top + y - HEADER_H
        for (l, c0, c1, nid) in self.hotspots:
            if l == li:
                return (l, c0, c1, nid)
        return None

    # There is deliberately no hover highlight. With the whole row clickable,
    # the green +/- glyph is already the affordance; a second signal for the
    # same fact was redundant, and it was the part that kept going wrong —
    # first an underline that drew a rule across the indent, then a mark that
    # lingered after the click that dismissed it, then a colour that could
    # never appear because the terminal was not reporting motion at all.

    def toggle(self, nid):
        if nid in self.open:
            self.open.discard(nid)
        else:
            self.open.add(nid)
        self.touched.add(nid)
        self.rebuild()

    def expand_menu(self):
        """Keyboard route: everything expandable in the exchange you are
        LOOKING at, not the newest one — scrolling back to an exchange and
        pressing x should expand that one.
        """
        if not self.recs:
            return
        rec = self.rec_at(self.top) or self.recs[-1]
        ri = self.recs.index(rec)
        req = rec.get("req") or {}
        pre, _ = T.normalise(req)
        items = [(ri, "preamble", "system prompt")]
        for lab, icon, needles in cat_markers():
            # case-SENSITIVE, matching markers_in() and detail_for(); see the
            # note in llamatap-catalog.py
            if any(nd in pre for nd in needles):
                items.append((ri, "marker", lab))
        for t in (req.get("tools") or []):
            if isinstance(t, dict):
                n = t.get("name") or (t.get("function") or {}).get("name")
                if n:
                    items.append((ri, "tool", n))
        self.menu = items
        self.msg = (" press 1-9 to expand/collapse: "
                    + "  ".join(f"{i}:{k[:12]}" for i, (_, _, k) in enumerate(items[:9], 1)))
        self.msg_until = time.time() + 15

    # ------------------------------------------------------------------ commands
    def note(self, text, secs=8):
        """Transient status, shown on the bottom line and then forgotten.

        These used to be appended to the conversation buffer, which meant an
        "archived…" confirmation sat permanently between exchanges — noise in
        a transcript you may later copy or paste. Only output you explicitly
        asked for (/help, /list) belongs in the buffer.
        """
        self.msg = f" {text} "
        self.msg_until = time.time() + secs

    def run_cmd(self, raw):
        raw = raw.strip()
        if raw.lower().startswith("ask "):
            # free text: never shell-split a sentence
            return self.cmd_ask([raw[4:]])
        try:
            parts = shlex.split(raw) if raw else []
        except ValueError:
            parts = raw.split()
        if not parts:
            return
        c, args = parts[0].lower(), parts[1:]
        if c in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        elif c == "help":
            lines = ["  commands — press / to get this as a searchable palette", ""]
            lines += [f"  /{n:<9} {a:<18} {d}" for n, a, d in COMMANDS]
            lines += ["", "  keys"] + ["  " + h for h in HELP if not h.startswith("/")]
            self.extra.append("\033[36m" + "\n".join(lines) + "\033[0m\n")
            self.rebuild()
        elif c == "clear":
            self.cmd_clear()
        elif c == "copy":
            self.cmd_copy(args)
        elif c == "save":
            self.cmd_save(args)
        elif c == "pager":
            self.cmd_pager()
        elif c == "repeats":
            arg = (args[0].lower() if args else "")
            if arg in ("open", "all", "expand"):
                self.opts["repeats"] = True
                self.rebuild()
                for (_l, _c0, _c1, nid) in self.hotspots:
                    if nid[1] == "repeat":
                        self.open.add(nid)
                self.rebuild()
                self.note("every re-sent message expanded")
            elif arg in ("close", "none", "collapse"):
                self.open = {n for n in self.open if n[1] != "repeat"}
                self.rebuild()
                self.note("re-sent messages collapsed")
            else:
                self.opts["repeats"] = not self.opts.get("repeats", False)
                if not self.opts["repeats"]:
                    self.open = {n for n in self.open if n[1] != "repeat"}
                self.rebuild()
        elif c in ("stats", "panel", "rail"):
            arg = (args[0].lower() if args else "")
            order = ("right", "bottom", "off")
            if args and arg not in order:
                return self.note(f"no such placement: {args[0]} — "
                                 f"try right, bottom or off")
            if arg in order:
                self.opts["stats"] = arg
            else:
                cur = self.opts.get("stats", "right")
                self.opts["stats"] = order[(order.index(cur) + 1) % len(order)
                                           if cur in order else 0]
            self.rebuild()
            asked, shown = self.opts["stats"], self.stats_where()
            if asked != shown and shown == "off":
                # Saying "stats across the bottom" while drawing nothing is
                # worse than drawing nothing: the user presses `s` three times
                # and two of them appear to do nothing at all.
                cols = self.scr.getmaxyx()[1]
                need = BOTTOM_MIN if asked == "bottom" else 80
                self.note(f"{asked} needs {need} columns — this window is "
                          f"{cols}, so the statistics stay hidden")
            else:
                self.note({"right": "stats on the right — the few you watch live",
                           "bottom": "stats across the bottom — everything",
                           "off": "stats hidden"}[asked])
        elif c in ("thinking", "think"):
            arg = (args[0].lower() if args else "")
            # These set the DEFAULT for every thought, so they also clear any
            # per-node decisions — otherwise "close them all" would leave the
            # ones you had opened by hand still open.
            self.touched = {n for n in self.touched if n[1] != "thinking"}
            if arg in ("open", "all", "expand"):
                self.opts["fold_thinking"] = False
                self.rebuild()
                self.note("every thought expanded")
            elif arg in ("close", "none", "collapse"):
                self.opts["fold_thinking"] = True
                self.rebuild()
                self.note("thinking folded to one line each")
            else:
                self.opts["fold_thinking"] = not self.opts.get("fold_thinking", False)
                self.rebuild()
                self.note("thinking folded — click a Thought line to open one"
                          if self.opts["fold_thinking"] else "thinking shown in full")
        elif c == "preamble":
            pre = [nid for (_l, _a, _b, nid) in self.hotspots if nid[1] == "preamble"]
            if not pre:
                return self.note("no system prompt in this recording")
            if all(n in self.open for n in pre):
                for n in pre:
                    self.open.discard(n)
                self.note("system prompt collapsed")
            else:
                for n in pre:
                    self.open.add(n)
                self.note(f"system prompt expanded ({len(pre)} request(s))")
            self.rebuild()
        elif c in ("raw", "json"):
            self.opts["rawjson"] = not self.opts.get("rawjson", False)
            self.rebuild()
            self.note("tool calls: raw JSON as sent"
                      if self.opts["rawjson"] else "tool calls: readable")
        elif c == "flags":
            self.opts["flags"] = not self.opts.get("flags", True)
            self.rebuild()
            self.note("launch settings shown" if self.opts["flags"]
                      else "launch settings hidden")
        elif c == "dedupe":
            self.opts["repeats"] = not self.opts.get("repeats", False); self.rebuild()
        elif c == "name":
            self.name = args[0] if args else None; self.rebuild()
        elif c == "list":
            self.extra.append(render_to_text(T.cmd_list, self.recs, self.scr.getmaxyx()[1] - 2))
            self.rebuild()
        elif c == "read":
            try:
                n = int(args[0])
            except Exception:
                return self.note("usage: /read N")
            if 1 <= n <= len(self.lines):
                pass
            self.jump_to(n)
        elif c == "ask":
            self.cmd_ask(args)
        elif c == "replay":
            self.cmd_replay(args)
        else:
            self.note(f"unknown command: /{c} — try /help")

    def jump_to(self, n):
        """Scroll to the first line of exchange n.

        This used to re-render every earlier exchange with the per-exchange
        TEXT renderer and count the lines it produced — a layout the console
        does not use, so the offset was unrelated to what is on screen.
        rebuild() already records where each exchange starts.
        """
        if not 1 <= n <= len(self.recs):
            return self.note(f"no exchange #{n} — the recording has {len(self.recs)}")
        near = None
        for line, rec_i in (self.starts or []):
            if rec_i == n - 1:              # starts holds 0-based record indexes
                self.top = line
                self.follow = False
                return
            if rec_i < n - 1:
                near = line
        # No line of its own means every message it carried was already on
        # screen — the request was entirely re-sent history. Land next to it
        # and say so, because "nothing new" is a real measurement.
        if near is not None:
            self.top = near
            self.follow = False
            self.note(f"exchange #{n} added no new messages — all re-sent "
                      f"history. Showing where it begins; /repeats to list them")
        else:
            self.note(f"exchange #{n} is not in the current view — "
                      f"/repeats to show re-sent messages")

    def header_line(self):
        """The participants, and which ones the filter is showing.

        The party row is pinned chrome now, not part of the transcript, so it
        stopped appearing in a copy — a pasted buffer no longer said which
        harness, which server build or which model produced it. That is most
        of what makes a buffer worth keeping when comparing harnesses later.
        """
        chips = getattr(self, "chips", None)
        if not chips:
            return ""
        show = getattr(self, "show", None) or set(PARTIES)
        names = [label if party in show else f"({label})"
                 for party, label, _x0, _x1 in chips]
        line = "in this thread:  " + "   ".join(names)
        if any(party not in show for party, _l, _a, _b in chips):
            # Say it in the copy itself. A filtered buffer that looks
            # complete is worse than no buffer.
            line += "\n(parenthesised parties were hidden by the filter)"
        return line + "\n"

    def buffer_text(self, raw=None):
        """Exactly what is on screen, as plain text.

        This used to re-render each exchange separately, which reproduced the
        OLD per-exchange layout: every exchange replaying the entire history,
        with its own header, so one short conversation came out as dozens of
        repeated blocks. The buffer is already built and already deduplicated —
        copy that.
        """
        if raw is not None and raw != self.opts.get("rawjson", False):
            # rebuild() reads "rawjson"; toggling "raw" here was a no-op, so
            # `/copy raw` silently copied the readable rendering instead.
            keep = self.opts.get("rawjson", False)
            self.opts["rawjson"] = raw
            self.rebuild()
            out = "\n".join("".join(c for _, c, _ in spans) for spans in self.lines)
            self.opts["rawjson"] = keep
            self.rebuild()
            return self.header_line() + out
        return self.header_line() + "\n".join(
            "".join(c for _, c, _ in spans) for spans in self.lines)

    def cmd_copy(self, args):
        raw = bool(args) and args[0].lower() in ("raw", "--raw")
        text = self.buffer_text(raw=True if raw else None)
        tool = clipboard(text)
        if tool:
            # What was COPIED, not how many records exist. With three parties
            # hidden it still claimed every exchange while copying a
            # fraction of them — a confirmation that overstates its own
            # result is worse than none.
            show = getattr(self, "show", None) or set(PARTIES)
            scope = ("" if set(show) == set(PARTIES)
                     else " · " + ", ".join(p for p in PARTIES if p in show)
                          + " only")
            self.note(f"copied {len(text):,} chars, "
                      f"{text.count(chr(10)) + 1:,} lines"
                      f"{', raw' if raw else ''}{scope} via {tool}")
        else:
            self.note("no clipboard tool found (wl-copy / xclip / xsel) — try /save")

    def cmd_save(self, args):
        path = os.path.expanduser(args[0]) if args else \
            os.path.join(_STATE, "llamatap-buffer.txt")
        try:
            with open(path, "w") as f:
                f.write(self.buffer_text())
            self.note(f"wrote {path}")
        except Exception as e:
            self.note(f"save failed: {e}")

    def cmd_pager(self):
        """Suspend curses and hand the buffer to less.

        This is the escape hatch for anything the TUI makes awkward: in the
        pager your terminal's own selection, search and scrollback all work
        normally. Returning restores the console.
        """
        text = self.buffer_text()
        curses.endwin()
        try:
            subprocess.run(["less", "-R"], input=text.encode())
        except Exception:
            try:
                subprocess.run(["more"], input=text.encode())
            except Exception:
                pass
        self.scr.clear()
        curses.doupdate()
        self.scr.refresh()

    def cmd_ask(self, args):
        """Send a message to the model IN THE HARNESS'S CONTEXT.

        Takes the most recent real request — its system prompt, its tool
        definitions, the whole conversation so far — appends your message and
        sends it. So you are talking to the model the way the harness would,
        without switching apps.

        What this is NOT: a harness. If the model asks for a tool, nothing will
        run it and nothing will come back. That is a true picture of the model
        without scaffolding, which is worth seeing, but it is not the agent
        loop.
        """
        text = " ".join(args).strip()
        if not text:
            return self.note("usage: /ask <your message>")
        rec = self._dash_rec or (self.recs[-1] if self.recs else None)
        if rec is None:
            return self.note("nothing captured yet — run one turn through the harness first")
        if not (rec.get("req") or {}).get("messages"):
            return self.note("the captured request has no messages to continue")
        req = ask_payload(rec, text)
        self.busy = f"asking… {text[:40]}"

        def work():
            import urllib.request, urllib.error
            up = os.environ.get("LLAMATAP_UPSTREAM", _UPSTREAM_DEFAULT)
            url = up.rstrip("/") + (rec.get("path") or "/v1/chat/completions")
            t0 = time.time()
            try:
                r = urllib.request.Request(
                    url, data=json.dumps(req).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer dummy"}, method="POST")
                with urllib.request.urlopen(r, timeout=900) as resp:
                    body, status = resp.read().decode("utf8", "replace"), resp.status
            except Exception as e:
                self.busy = ""
                return self.note(f"ask failed: {e}")
            try:
                d = json.loads(body)
            except Exception:
                d = {}
            with open(self.path, "a") as f:
                f.write(json.dumps({
                    "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "path": rec.get("path"), "status": status, "req": req,
                    "meta": {"elapsed_ms": int((time.time() - t0) * 1000),
                             "injected": True,
                             "usage": d.get("usage"), "timings": d.get("timings")},
                    "resp": body}) + "\n")
            self.busy = ""
            self.note("sent from llamatap — no tools will run, see /help")

        threading.Thread(target=work, daemon=True).start()

    def cmd_clear(self):
        """Archive, then truncate. Never a plain delete: a recording is the
        only copy of a result.

        Delegates to llamatap-archive, which picks a destination that cannot
        collide. The stamp here was minute-resolution with
        `makedirs(exist_ok=True)`, so clearing twice inside a minute silently
        overwrote the first archive.
        """
        try:
            dest = ARCHIVE.archive([self.path, LOG], RUNS)
        except Exception as e:
            # Nothing was truncated, so say so plainly.
            return self.note(f"NOT cleared — archiving failed: {e}")
        # Only now is it safe to start fresh.
        try:
            for f in (self.path, LOG):
                if os.path.exists(f):
                    open(f, "w").close()
        except Exception as e:
            return self.note(f"archived to {dest}, but could not truncate: {e}")
        self.recs, self.extra, self.top, self.follow = [], [], 0, True
        if dest:
            name = os.path.basename(dest)
            self.note(f"archived to runs/{name}/  "
                      f"(llamatap read --file {dest}/"
                      f"{os.path.basename(self.path)})", secs=12)
        else:
            self.note("nothing to archive; recording was already empty")
        self.rebuild()

    def cmd_replay(self, args):
        if not args:
            return self.note("usage: /replay N [--model NAME] [--diff]")
        cmd = replay_cmd(args, self.path)
        self.busy = f"replaying {' '.join(args)}…"

        def work():
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                out = (r.stdout or "") + (r.stderr or "")
            except Exception as e:
                out = f"replay failed: {e}"
            self.busy = ""
            # the replay appends to the recording, so the new exchanges arrive
            # on the next poll; only the comparison table needs showing here
            tail = "\n".join(l for l in out.split("\n") if "comparison" in l or "t/s" in l
                             or "decode" in l or l.strip().startswith("qwen"))
            self.note("replay finished" + (f"\n{tail}" if tail.strip() else ""))

        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------------------- loop
    def run(self):
        self.scr.nodelay(True)
        self.scr.keypad(True)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            curses.mouseinterval(0)
        except Exception:
            pass
        self.recs = load_records(self.path)
        self.rebuild()
        seen, last_check = len(self.recs), 0.0

        while True:
            self.draw()
            ch = self.scr.getch()
            if ch != -1:
                if ch == curses.KEY_MOUSE:
                    try:
                        _, mx, my, _, bstate = curses.getmouse()
                    except curses.error:
                        continue
                    if bstate & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
                        if my < HEADER_H:
                            party = self.chip_at(mx)
                            if party:
                                self.toggle_party(party)
                            else:
                                self.note("click a party's name to show or "
                                          "hide its messages")
                            continue
                        hs = self.hotspot_at(my, mx)
                        if hs:
                            self.toggle(hs[3])
                        else:
                            self.note("nothing to expand on that line — "
                                      "a green + or − marks what opens")
                    elif bstate & getattr(curses, "BUTTON4_PRESSED", 0):
                        self.top = max(0, self.top - 3); self.follow = False
                    elif bstate & getattr(curses, "BUTTON5_PRESSED", 0):
                        self.top = min(max(0, len(self.lines) - self.view_h()), self.top + 3)
                    continue
                if self.cmd is not None:
                    matches = match_commands(self.cmd)
                    if ch in (10, 13, curses.KEY_ENTER):
                        line = self.cmd
                        # Enter runs the HIGHLIGHTED command unless arguments
                        # were typed, in which case the typed line wins.
                        if matches and " " not in line.strip():
                            line = matches[self.cmd_sel][0]
                        elif matches and " " in line.strip():
                            head, _, rest = line.strip().partition(" ")
                            if not any(c[0] == head for c in COMMANDS):
                                line = matches[self.cmd_sel][0] + " " + rest
                        self.cmd, self.cmd_sel = None, 0
                        self.run_cmd(line)
                    elif ch == 27:
                        self.cmd, self.cmd_sel = None, 0
                    elif ch == curses.KEY_UP:
                        self.cmd_sel = max(0, self.cmd_sel - 1)
                    elif ch == curses.KEY_DOWN:
                        self.cmd_sel = min(max(0, len(matches) - 1), self.cmd_sel + 1)
                    elif ch == 9:                     # Tab completes the name
                        if matches:
                            self.cmd = matches[self.cmd_sel][0] + " "
                    elif ch in (curses.KEY_BACKSPACE, 127, 8):
                        self.cmd, self.cmd_sel = self.cmd[:-1], 0
                    elif 32 <= ch < 127:
                        self.cmd, self.cmd_sel = self.cmd + chr(ch), 0
                    continue

                self.msg = ""
                vh = self.view_h()
                if ch != ord("g"):
                    self._g_pending = False
                if ch == ord("/"):
                    self.cmd = ""
                elif ch == 27:
                    # Escape cancels, it does not quit. It is the key you hit
                    # to back out of something, and quitting the whole console
                    # because a stray Esc arrived is the wrong answer — arrow
                    # keys and mouse reports are escape sequences, so a dropped
                    # byte can look like a bare Esc.
                    self.msg = " nothing to cancel — press q to quit "
                    self.msg_until = time.time() + 3
                elif ch == ord("q"):
                    return
                elif ch in (curses.KEY_UP, ord("k")):
                    self.top = max(0, self.top - 1); self.follow = False
                elif ch in (curses.KEY_DOWN, ord("j")):
                    self.top = min(max(0, len(self.lines) - vh), self.top + 1)
                elif ch in (curses.KEY_PPAGE, 2):          # ctrl-b
                    self.top = max(0, self.top - vh); self.follow = False
                elif ch in (curses.KEY_NPAGE, ord(" "), 6):   # ctrl-f
                    self.top = min(max(0, len(self.lines) - vh), self.top + vh)
                elif ch == 21:                             # ctrl-u, half page
                    self.top = max(0, self.top - vh // 2); self.follow = False
                elif ch == 4:                              # ctrl-d, half page
                    self.top = min(max(0, len(self.lines) - vh),
                                   self.top + vh // 2)
                elif ch == ord("g"):
                    # vim's gg. Home needs Fn on a laptop, which is the wrong
                    # key for something you press while reading. A bare g waits
                    # for the second one; any other key cancels the prefix and
                    # is handled normally on the next pass.
                    if getattr(self, "_g_pending", False):
                        self._g_pending = False
                        self.top = 0; self.follow = False
                    else:
                        self._g_pending = True
                        self.msg = " g again for the top "
                        self.msg_until = time.time() + 2
                        continue
                elif ch in (curses.KEY_HOME,):
                    self.top = 0; self.follow = False
                elif ch in (curses.KEY_END, ord("G")):
                    self.follow = True; self.to_bottom()
                elif ch == ord("r"):
                    self.run_cmd("raw")
                elif ch == ord("x"):
                    self.expand_menu()
                elif 49 <= ch <= 57 and self.menu:
                    n = ch - 48
                    if n <= len(self.menu):
                        self.toggle(self.menu[n - 1])
                elif ch in (ord("o"), ord("O")):
                    # One key that toggles. Two keys differing only by shift,
                    # one expanding the current exchange and the other
                    # collapsing the whole buffer, were neither symmetrical
                    # nor memorable — you could not undo `o` with `o`.
                    self.expand_menu()
                    if any(nid in self.open for nid in self.menu):
                        for nid in self.menu:
                            self.open.discard(nid)
                        self.note("collapsed this exchange")
                    else:
                        for nid in self.menu:
                            self.open.add(nid)
                        self.note("expanded this exchange")
                    self.rebuild()
                elif ch == ord("y"):
                    self.cmd_copy([])
                elif ch == ord("p"):
                    self.run_cmd("preamble")
                elif ch == ord("f"):
                    self.opts["flags"] = not self.opts["flags"]; self.rebuild()
                elif ch == ord("t"):
                    self.run_cmd("thinking")
                elif ch == ord("s"):
                    self.run_cmd("stats")
                elif ch == curses.KEY_RESIZE:
                    self.rebuild()
                if self.top >= max(0, len(self.lines) - vh):
                    self.follow = True
                continue

            now = time.time()
            if self.msg and now > getattr(self, "msg_until", 0):
                self.msg = ""
            if now - getattr(self, "_fl", 0) > 0.15:
                self._fl = now
                self.tick += 1
                prev = [(j.get("text"), j.get("thinking")) for j in self.inflight]
                self.inflight = read_inflight()
                cur = [(j.get("text"), j.get("thinking")) for j in self.inflight]
                # The streamed text changes several times a second, so
                # rebuilding on every change would eat the event loop and make
                # scrolling stutter while the model is talking. Rebuild at most
                # once per 4x its own measured cost, and never more than twice
                # a second — so a long recording backs off on its own instead
                # of freezing the view, which is what an 18 MB recording did.
                gap = max(0.5, 4 * getattr(self, "_rb_s", 0.05))
                if cur != prev and now - getattr(self, "_live_rb", 0) > gap:
                    self._live_rb = now
                    self.rebuild()
            if now - last_check > 0.5:
                last_check = now
                recs = load_records(self.path)
                if len(recs) != len(self.recs):
                    grew = len(recs) > len(self.recs)
                    self.recs = recs
                    self.rebuild()
                    if grew and not self.follow:
                        self.msg = (f" {len(recs) - seen} new exchange(s) below — "
                                    f"g or End to jump to the bottom ")
                    seen = len(recs)
            time.sleep(0.05)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preamble", action="store_true")
    ap.add_argument("--raw-json", action="store_true",
                    help="show tool-call arguments as raw JSON off the wire")
    ap.add_argument("--no-flags", action="store_true")
    ap.add_argument("--stats", choices=("right", "bottom", "off"),
                    default="right",
                    help="where the statistics go (default: right)")
    ap.add_argument("--fold-thinking", action="store_true",
                    help="start with the model's thinking folded to one line")
    ap.add_argument("--repeats", action="store_true",
                    help="show every message each time the harness re-sent it")
    ap.add_argument("--name")
    ap.add_argument("--file", default=REC)
    a = ap.parse_args()
    if not sys.stdout.isatty():
        sys.exit("llamatap console needs a terminal. Use `llamatap follow` when piping.")
    # The console is the readable view, always. Raw JSON and icon-less output
    # live in `llamatap read --raw` / `--no-icons`, which is the right place
    # for them: those are for piping and diffing, not for watching.
    # The launch-settings board is twenty rows; it would own the screen. Off by
    # default, shown with /flags when you want to know how the server was told
    # to run.
    opts = {"rawjson": a.raw_json, "stats": a.stats,
            # Off by default: folding changes what the thread looks like, and
            # the default should be "show me everything that happened".
            "fold_thinking": a.fold_thinking,
            "preamble": a.preamble, "flags": False,
            "repeats": a.repeats}

    def boot(scr):
        curses.start_color()
        curses.use_default_colors()
        fgs = {31: curses.COLOR_RED, 32: curses.COLOR_GREEN, 33: curses.COLOR_YELLOW,
               34: curses.COLOR_BLUE, 35: curses.COLOR_MAGENTA, 36: curses.COLOR_CYAN,
               37: curses.COLOR_WHITE, 90: curses.COLOR_BLACK, 94: curses.COLOR_BLUE}
        for code, pair in PAIRS.items():
            if code < 100:
                curses.init_pair(pair, fgs[code], -1)
        # A subtly lighter background for message panels. 236 is a dark grey
        # in the xterm-256 palette; on a terminal without 256 colours fall
        # back to plain black, which still separates a block from a wallpaper.
        panel_bg = 236 if getattr(curses, "COLORS", 8) >= 256 else curses.COLOR_BLACK
        for pair, name in PANEL_FG:
            curses.init_pair(pair, getattr(curses, f"COLOR_{name}"), panel_bg)
        try:
            Console(scr, a.file, a.name, opts).run()
        except KeyboardInterrupt:
            pass

    curses.wrapper(boot)


if __name__ == "__main__":
    main()
