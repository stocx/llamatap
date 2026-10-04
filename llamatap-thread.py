#!/usr/bin/env python3
"""Render a tap recording as a conversation thread.

Reads llamatap-records.jsonl (written by llamatap-proxy.py) and prints the exchange
the way you would read a text message thread: what the HARNESS said, what the
MODEL said back, tool calls and their results in between.

    llamatap read                      the latest exchange, pretty
    llamatap follow                    live, as new turns arrive
    llamatap read --raw                raw JSON instead of prose
    llamatap all                       every recorded request, not just the last
    llamatap read --name opencode      override the detected harness name

Why the LAST record by default: every request carries the whole conversation so
far, so the final request plus its response reconstructs the entire thread.
Printing every record would repeat the history once per turn.
"""
import argparse, json, os, re, select, shutil, subprocess, sys, termios, time, tty, unicodedata


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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "tap_catalog", os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamatap-catalog.py"))
cat = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cat)

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

# Harness signatures, matched against the system prompt. Order matters:
# "Claude Code" before "code".
SIGNATURES = [
    ("opencode", "opencode"), ("Claude Code", "Claude Code"), ("claude code", "Claude Code"),
    ("You are Pi", "Pi"), ("pi-coding-agent", "Pi"), ("Cline", "Cline"),
    ("Roo", "Roo Code"), ("aider", "Aider"), ("Codex CLI", "Codex"),
    ("codex", "Codex"), ("Zed", "Zed"), ("Goose", "Goose"),
]

# Colour note: \033[90m ("bright black") is unreadable on a dark theme — it was
# used here for the stats panels and the thinking text, i.e. for CONTENT, and
# the result was grey-on-grey. Rule now: `rule` is for separator lines only,
# everything a human has to read uses the terminal's default foreground or a
# real colour.
C = {"harness": "\033[36m", "model": "\033[33m", "tool": "\033[35m",
     "you": "\033[32m", "think": "\033[94m", "label": "\033[36m",
     "rule": "\033[90m", "dim": "", "off": "\033[0m", "bold": "\033[1m"}


def nocolor():
    for k in C:
        C[k] = ""


def width():
    return min(shutil.get_terminal_size((100, 24)).columns, 100)


def flatten(content):
    """Content arrives as a string (OpenAI) or a list of typed blocks
    (Anthropic, newer OpenAI). Return (text, [tool_use], [tool_result])."""
    if content is None:
        return "", [], []
    if isinstance(content, str):
        return content, [], []
    text, uses, results = [], [], []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                text.append(str(b))
                continue
            t = b.get("type")
            if t == "text" or b.get("text"):
                text.append(b.get("text") or "")
            elif t == "tool_use":
                uses.append({"name": b.get("name"), "args": b.get("input")})
            elif t == "tool_result":
                rc = b.get("content")
                rt, _, _ = flatten(rc) if not isinstance(rc, str) else (rc, [], [])
                results.append({"id": b.get("tool_use_id"), "text": rt})
    return "\n".join(x for x in text if x), uses, results


def normalise(req):
    """(preamble, turns) — turns being {who, kind, text, tool}.

    Returns a PAIR on every path. The early guard used to return a bare list
    while the end returned a 2-tuple, so `preamble, turns = normalise(req)`
    raised ValueError for anything that was not a dict. A bodyless request —
    any `GET /v1/models` — is recorded as `req: null`, and one of those made
    the whole recording unopenable in the console.
    """
    turns = []
    if not isinstance(req, dict):
        return "", turns

    sys_text = []
    s = req.get("system")
    if s:
        t, _, _ = flatten(s)
        sys_text.append(t)
    if req.get("instructions"):
        sys_text.append(req["instructions"])

    msgs = req.get("messages") if isinstance(req.get("messages"), list) else []
    items = req.get("input") if isinstance(req.get("input"), list) else []

    for m in (msgs or items):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text, uses, results = flatten(m.get("content"))
        if role in ("system", "developer"):
            sys_text.append(text)
            continue
        if role == "user":
            for r in results:
                turns.append({"who": "tool", "kind": "tool result", "text": r["text"]})
            if text.strip():
                turns.append({"who": "you", "kind": "", "text": text})
        elif role == "assistant":
            if text.strip():
                turns.append({"who": "model", "kind": "", "text": text})
            for u in uses:
                turns.append({"who": "model", "kind": "tool call",
                              "text": json.dumps(u["args"], indent=2) if u["args"] else "",
                              "tool": u["name"]})
            for tc in (m.get("tool_calls") or []):
                fn = (tc.get("function") or {})
                turns.append({"who": "model", "kind": "tool call",
                              "text": fn.get("arguments") or "", "tool": fn.get("name")})
        elif role == "tool":
            turns.append({"who": "tool", "kind": "tool result", "text": text,
                          "tool": m.get("name")})

    preamble = "\n\n".join(x for x in sys_text if x)
    return preamble, turns


# Parsed responses, keyed by the raw body. A record's response never changes
# once written, but the console reloads the whole recording twice a second and
# re-parsed every one: 20,000 json.loads calls per rebuild on a 14 MB
# recording. The key holds a reference to a string that is already in memory,
# so this costs dict overhead and nothing else.
_RESP_CACHE = {}
_RESP_CACHE_MAX = 512


def from_response(resp):
    """The assistant turn that came back. Handles plain JSON and SSE."""
    if not resp:
        return []
    hit = _RESP_CACHE.get(resp)
    if hit is not None:
        return hit
    out = _from_response_uncached(resp)
    if len(_RESP_CACHE) >= _RESP_CACHE_MAX:
        # Oldest first; a long session's recording grows without bound and
        # this must not grow with it.
        for k in list(_RESP_CACHE)[:_RESP_CACHE_MAX // 4]:
            del _RESP_CACHE[k]
    _RESP_CACHE[resp] = out
    return out


def _from_response_uncached(resp):
    if not resp:
        return []
    out, text, calls = [], "", []
    reasoning, finish = "", None
    try:
        d = json.loads(resp)
        msg = ((d.get("choices") or [{}])[0].get("message")) or {}
        if d.get("content") and not msg:                       # Anthropic shape
            t, uses, _ = flatten(d["content"])
            text = t
            calls = [{"name": u["name"], "args": json.dumps(u["args"])} for u in uses]
        else:
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            finish = ((d.get("choices") or [{}])[0]).get("finish_reason")
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                calls.append({"name": fn.get("name"), "args": fn.get("arguments")})
    except Exception:
        reasoning, finish = "", None
        # SSE: concatenate the deltas
        chunks, names, args = [], {}, {}
        for line in resp.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p in ("", "[DONE]"):
                continue
            try:
                d = json.loads(p)
            except Exception:
                continue
            for ch in (d.get("choices") or []):
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning += delta["reasoning_content"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                for tc in (delta.get("tool_calls") or []):
                    i = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        names[i] = fn["name"]
                    if fn.get("arguments"):
                        args[i] = args.get(i, "") + fn["arguments"]
            if d.get("delta", {}).get("text"):                 # Anthropic SSE
                chunks.append(d["delta"]["text"])
        text = "".join(chunks)
        calls = [{"name": names.get(i), "args": args.get(i)} for i in sorted(set(names) | set(args))]

    # A thinking model puts its working here. Shown separately because it is
    # often most of what came back, and because an empty `content` with a full
    # reasoning_content is how a truncated thought presents itself.
    if reasoning.strip():
        out.append({"who": "model", "kind": "thinking", "text": reasoning})
    if text.strip():
        out.append({"who": "model", "kind": "", "text": text})
    for c in calls:
        out.append({"who": "model", "kind": "tool call", "text": c["args"] or "", "tool": c["name"]})
    # Only `length` and `content_filter` are real problems. `stop` and
    # `tool_calls` are both normal endings — flagging tool_calls as truncated
    # was wrong and alarming.
    if finish in ("length", "content_filter"):
        why = ("hit max_tokens — the reply is incomplete" if finish == "length"
               else "stopped by a content filter")
        out.append({"who": "model", "kind": "cut off", "text": f"{finish}: {why}"})
    elif not finish and reasoning.strip() and not text.strip() and not calls:
        # Reasoned, never answered, and no finish_reason ever arrived: the
        # stream stopped mid-chunk. Almost always the HARNESS giving up and
        # disconnecting. Saying nothing here makes it look like the model
        # simply had nothing to say.
        out.append({"who": "model", "kind": "cut off",
                    "text": "no reply — the stream ended mid-thought with no "
                            "finish_reason, which means the connection closed "
                            "before an answer was produced"})
    return out


PROPS_CACHE = {}
FLAGS_CACHE = {}


def props(model, upstream=None):
    """Everything the server will tell us about the loaded model.

    Two routes, because both deployments are supported and documented:
    llama-swap exposes each backend at /upstream/<model>/props, while a direct
    llama-server — now the default upstream — serves /props and has no
    /upstream route at all. Trying only the swap route meant a direct install
    showed no model details whatsoever.

    Only a SUCCESSFUL lookup is cached. Caching the failure meant one
    unreachable moment hid the model details for the rest of the session.
    """
    if not model:
        return {}
    if model in PROPS_CACHE:
        return PROPS_CACHE[model]
    base = (upstream or os.environ.get("LLAMATAP_UPSTREAM",
                                       _UPSTREAM_DEFAULT)).rstrip("/")
    import urllib.request
    for url in (f"{base}/upstream/{model}/props", f"{base}/props"):
        try:
            with urllib.request.urlopen(url, timeout=6) as r:
                d = json.loads(r.read())
            if isinstance(d, dict) and d:
                PROPS_CACHE[model] = d
                return d
        except Exception:
            continue
    return {}


def launch_flags(model):
    """The llama-server command line, best source first. Command string only."""
    return launch_info(model)[0]


# Where the last known command line is kept between runs, so the panel is not
# blank just because the model is idle or the ssh agent was cleared.
FLAGS_DISK = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "llamatap", "launch-flags.json")


def _disk_flags():
    try:
        with open(FLAGS_DISK) as f:
            return json.load(f)
    except Exception:
        return {}


def _remember_flags(model, cmd):
    """Persist a command line so an idle model still has one to show."""
    try:
        d = _disk_flags()
        d[model] = {"cmd": cmd, "at": time.time()}
        os.makedirs(os.path.dirname(FLAGS_DISK), exist_ok=True)
        tmp = FLAGS_DISK + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, FLAGS_DISK)
    except Exception:
        pass


def launch_info(model):
    """(command, source, age_seconds) for how llama-server was started.

    props does NOT report launch flags — no -ctk, no -fa, no -ngl, no draft
    model — and those are exactly the settings that decide speed and whether a
    long context fits. Three sources, in descending order of truth:

    1. llama-swap's `/running`, which reports the command line of the process
       that is ACTUALLY running. This beats reading the config file: if
       llama-swap was started before the YAML was last edited, the file
       describes a launch that never happened. It also needs no ssh — just the
       upstream llamatap already forwards to.
    2. A disk cache of the last value seen, because `/running` lists only
       LOADED models and llama-swap unloads on its ttl. Reported with its age
       rather than passed off as current.
    3. The config file over ssh, for a model that has never been loaded in
       this llama-swap's lifetime. Needs the ssh agent; last resort.

    `age` is None when the value is live.
    """
    if not model:
        return (None, "none", None)
    if model in FLAGS_CACHE:
        return FLAGS_CACHE[model]

    out = None
    base = os.environ.get("LLAMATAP_UPSTREAM", _UPSTREAM_DEFAULT)

    # 1. the running process
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base}/running", timeout=6) as r:
            for ent in (json.loads(r.read()) or {}).get("running") or []:
                if ent.get("model") == model and ent.get("cmd"):
                    out = (" ".join(ent["cmd"].split()), "running", None)
                    break
    except Exception:
        pass

    # 2. what it was last time
    if out is None:
        ent = _disk_flags().get(model) or {}
        if ent.get("cmd"):
            out = (ent["cmd"], "cached", max(0.0, time.time() - (ent.get("at") or 0)))

    # 3. the config file, over ssh
    if out is None:
        # Tier 3 is opt-in. Guessing a hostname meant anyone else's install
        # tried to ssh somewhere that does not exist, waited for the timeout,
        # and reported nothing useful. Unset means "skip this tier".
        host = os.environ.get("LLAMATAP_SSH_HOST")
        cfg = os.environ.get("LLAMATAP_SWAP_CONFIG")
        try:
            if not host or not cfg:
                raise RuntimeError("ssh fallback not configured")
            env = dict(os.environ)
            env.setdefault("SSH_AUTH_SOCK",
                           f"/run/user/{os.getuid()}/ssh-agent.socket")
            raw = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host,
                 f"cat {cfg}"],
                capture_output=True, text=True, timeout=15, env=env).stdout
            lines, grab, cmd = raw.splitlines(), False, []
            for ln in lines:
                st = ln.strip()
                if re.match(rf'^"?{re.escape(model)}"?:\s*$', st):
                    grab = True
                    continue
                if grab:
                    if st.startswith("cmd:"):
                        continue
                    if (st.startswith("ttl:") or re.match(r'^"[^"]+":\s*$', st)
                            or st.startswith("groups:")):
                        break
                    if st and not st.startswith("#"):
                        cmd.append(st)
            if cmd:
                out = (" ".join(cmd), "config", None)
        except Exception:
            pass

    if out is None:
        out = (None, "none", None)
    elif out[1] in ("running", "config"):
        _remember_flags(model, out[0])
    FLAGS_CACHE[model] = out
    return out


# Launch flags worth calling out, with what they mean. Anything not listed is
# still shown verbatim in the raw command line.
# The settings worth knowing about, as a board: what is set, what is not.
# llama-server has hundreds of flags; these are the ones that change behaviour
# enough to explain a result. `None` as the pattern means "not in the command
# line", which is itself the answer.
FLAG_BOARD = [
    ("GPU Layers",     r"-ngl\s+(\S+)",         "on CPU"),
    ("Experts on CPU", r"--n-cpu-moe\s+(\S+)",  "all on GPU"),
    ("Context Window", r"-c\s+(\d+)",           "default"),
    ("K Cache",        r"-ctk\s+(\S+)",         "f16"),
    ("V Cache",        r"-ctv\s+(\S+)",         "f16"),
    ("Flash Attention", r"-fa\s+(\S+)",         "off"),
    ("Jinja Template", r"(--jinja)",             "off, no tool parsing"),
    ("Reasoning",      r"--reasoning\s+(\S+)",  "on"),
    ("Draft Model",    r"-md\s+(\S+)",          "none"),
    ("Threads",        r"-t\s+(\d+)",           "auto"),
    ("Batch Size",     r"-b\s+(\d+)",           "default"),
    ("Micro Batch",    r"-ub\s+(\d+)",          "default"),
    ("Memory Lock",    r"(--mlock)",             "off"),
    ("Memory Map",     r"(--no-mmap)",           "on"),
]


def flag_board(cmd):
    """[(label, value, is_set)] for the settings board."""
    out = []
    for label, pat, default in FLAG_BOARD:
        m = re.search(pat, cmd or "")
        if m:
            v = m.group(1)
            out.append((label, "on" if v.startswith("--") else v, True))
        else:
            out.append((label, default, False))
    return out


FLAG_NOTES = [
    # Plain text, no emoji. These are rendered inside a bordered panel, and
    # emoji without an emoji-presentation default (U+1F5DC among them) are
    # drawn one cell wide by some terminals and two by others — there is no
    # width calculation that is right for both, so the box breaks.
    ("-ctk q8_0", "", "K cache q8_0"),
    ("-ctv q8_0", "", "V cache q8_0"),
    ("-fa on", "", "flash attention"),
    ("--reasoning off", "", "thinking OFF"),
    ("--jinja", "", "jinja template + tool parsing"),
    ("-ngl 999", "", "all layers on GPU"),
    ("--n-cpu-moe", "", "some experts on CPU"),
    ("-md ", "", "draft model"),
]


def server_panel(model, w, show_flags=True):
    d = props(model)
    if not d:
        return
    g = (d.get("default_generation_settings") or {})
    prm = g.get("params") or {}
    caps = d.get("chat_template_caps") or {}
    print(f"{C['rule']}{'─' * w}{C['off']}")
    print(f"{C['bold']}  {d.get('model_alias', model)}{C['off']}"
          f"   {d.get('model_ftype','?')}   "
          f"{os.path.basename(d.get('model_path','') or '')}")
    print(f"  context {g.get('n_ctx', '?'):,} tokens"
          f"   slots {d.get('total_slots','?')}"
          f"   llama.cpp {d.get('build_info','?')}")
    sampl = []
    for k, label in (("temperature", "temp"), ("top_k", "top_k"),
                     ("top_p", "top_p"), ("min_p", "min_p")):
        if k in prm:
            v = prm[k]
            sampl.append(f"{label} {round(v, 3) if isinstance(v, float) else v}")
    if sampl:
        print(f"  {'   '.join(sampl)}")
    cap_bits = []
    if caps.get("supports_tools"):
        cap_bits.append("tools")
    if caps.get("supports_parallel_tool_calls"):
        cap_bits.append("parallel calls")
    if caps.get("supports_system_role"):
        cap_bits.append("system role")
    if not caps.get("supports_preserve_reasoning"):
        cap_bits.append("no reasoning preservation")
    if cap_bits:
        print(f"  template: {', '.join(cap_bits)}")
    mod = d.get("modalities") or {}
    on = [k for k, v in mod.items() if v]
    if on:
        print(f"  modalities: {', '.join(on)}")

    if show_flags:
        cmd = launch_flags(model)
        if cmd:
            notes = [f"{i} {t}" for pat, i, t in FLAG_NOTES if pat in cmd]
            if notes:
                print(f"  launch: {'   '.join(notes)}")
            short = re.sub(r"\S*llama-server\s*", "", cmd)
            short = re.sub(r"-m \S+\s*", "", short)
            print(f"  $ {short.strip()[:w-6]}")
    print(f"{C['rule']}{'─' * w}{C['off']}\n")


CTX_CACHE = {}


def ctx_limit(model, upstream=None):
    """Ask llama-server what context the loaded model actually has.

    llama-swap exposes each backend at /upstream/<model>/props, where
    default_generation_settings.n_ctx is the per-request window. Cached per
    model; failure is silent because the panel is still useful without it.
    """
    if not model:
        return None
    if model in CTX_CACHE:
        return CTX_CACHE[model]
    base = upstream or os.environ.get("LLAMATAP_UPSTREAM", _UPSTREAM_DEFAULT)
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base}/upstream/{model}/props", timeout=6) as r:
            d = json.loads(r.read())
        n = ((d.get("default_generation_settings") or {}).get("n_ctx"))
    except Exception:
        n = None
    CTX_CACHE[model] = n
    return n


def bar(frac, width_chars=28):
    """`░` renders as a near-solid block in several terminal fonts, which makes
    an EMPTY bar look full. `·` cannot be misread."""
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width_chars)
    return "█" * full + "·" * (width_chars - full)


def stats_panel(rec, w, model):
    meta = rec.get("meta") or {}
    u = meta.get("usage") or {}
    tm = meta.get("timings") or {}
    pt = u.get("prompt_tokens") or u.get("input_tokens") or tm.get("prompt_n")
    ct = u.get("completion_tokens") or u.get("output_tokens") or tm.get("predicted_n")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None and tm.get("cache_n") is not None:
        cached = tm["cache_n"]
    ttft = meta.get("ttft_ms")
    total = meta.get("elapsed_ms")
    if not any([pt, ct, ttft, total]):
        return

    print(f"{C['rule']}{'─' * w}{C['off']}")
    rows = []
    limit = ctx_limit(model)
    if pt is not None:
        if limit:
            frac = pt / limit
            rows.append(f"  context    {bar(frac)}  {pt:,} / {limit:,} tokens  ({frac*100:.0f}%)")
        else:
            rows.append(f"  context    {pt:,} tokens in prompt  (window size unavailable)")
    if cached is not None and pt:
        frac = cached / pt
        fresh = pt - cached
        rows.append(f"  kv cache   {bar(frac)}  {cached:,} reused, {fresh:,} fresh prefill")
    # llama.cpp's own numbers when present; wall clock only as a fallback.
    if tm:
        if tm.get("prompt_per_second"):
            # prompt_n is tokens actually PROCESSED, i.e. cache misses only —
            # which is why it is smaller than the context figure above.
            rows.append(f"  prefill    {tm['prompt_n']:,} new tokens at {tm['prompt_per_second']:,.0f}/sec"
                        f"   ({tm.get('prompt_ms', 0)/1000:.2f}s)")
        if tm.get("predicted_per_second"):
            rows.append(f"  decode     {tm['predicted_n']:,} tokens at {tm['predicted_per_second']:.1f}/sec"
                        f"   ({tm.get('predicted_ms', 0)/1000:.2f}s)")
    else:
        bits = []
        if meta.get("decode_tps"):
            bits.append(f"{meta['decode_tps']}/sec decode")
        elif meta.get("overall_tps"):
            bits.append(f"{meta['overall_tps']}/sec overall (not streamed)")
        if ttft is not None:
            bits.append(f"{ttft/1000:.1f}s to first token")
        if ct:
            bits.append(f"{ct} tokens generated")
        if bits:
            rows.append("  speed      " + "   ".join(bits))
    if total is not None:
        rows.append(f"  wall clock {total/1000:.1f}s end to end")
    for r in rows:
        print(r)
    print(f"{C['rule']}{'─' * w}{C['off']}\n")


def detect(preamble):
    for needle, label in SIGNATURES:
        if needle in (preamble or ""):
            return label
    return "HARNESS"


def wrap(text, w, indent):
    lines = []
    for para in (text or "").split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            if len(cur) + len(word) + 1 > w:
                lines.append(cur)
                cur = word
            else:
                cur = word if not cur else cur + " " + word
        lines.append(cur)
    return "\n".join(indent + l for l in lines)


def show(turn, name, w, raw, icons=True, model="the model", stat=""):
    who, kind = turn["who"], turn.get("kind")
    if who == "you":
        icon, label, col, ind = cat.PARTY["you"], "you", C["you"], ""
    elif who == "model":
        icon, label, col, ind = cat.PARTY["model"], model, C["model"], "    "
    elif who == "tool":
        icon, label, col, ind = cat.PARTY["harness"], f"{name} → tool result", C["tool"], ""
    else:
        icon, label, col, ind = cat.PARTY["harness"], name, C["harness"], ""
    if kind == "tool call":
        tcat, ticon = cat.classify_tool(turn.get("tool"))
        icon, col, ind = ticon, C["tool"], "    "
        label = f"{model} → {turn.get('tool')}()   [{tcat}]"
    elif kind == "thinking":
        icon, label, col, ind = "", f"{model} (thinking)", C["think"], "    "
    elif kind == "cut off":
        icon, label, col, ind = "", "TRUNCATED", "\033[31m", "    "
    elif kind == "tool result":
        icon, col, ind = "", C["tool"], ""
    elif kind == "background":
        icon, col, ind = cat.PARTY["harness"], C["harness"], ""
        label = f"{name.split(' · ')[0]} → asks on its own behalf, not typed by you"

    # chips for anything recognised inside the body
    if icons and not raw:
        found = cat.markers_in(turn.get("text"))
        if found:
            label += "   " + " ".join(f"{i}{l}" for l, i in found)
    # --raw implies no icons: raw mode is "show me the ugly truth", so the
    # decoration goes too, not just the prose formatting.
    if icons and not raw:
        label = f"{icon}  {label}"
    # Size (and, where known, duration) right-aligned on the rule — the same
    # habit as a shell prompt reporting how long a command took. Consistent on
    # every turn so you can compare them at a glance.
    if raw:
        stat = ""

    body = turn["text"]
    if raw:
        rule = "─" * max(0, w - dwidth(label) - 6 - len(ind) - dwidth(stat))
        print(f"{col}{ind}── {label} {rule}{C['off']}{C['label']}{stat}{C['off']}")
        print(ind + (body or "").replace("\n", "\n" + ind))
    else:
        preformatted = False
        if turn.get("kind") == "tool call":
            try:
                body = json.dumps(json.loads(body), indent=2)
                preformatted = True
            except Exception:
                pass
        rule = "─" * max(0, w - dwidth(label) - 6 - len(ind) - dwidth(stat))
        print(f"{col}{ind}── {label} {rule}{C['off']}{C['label']}{stat}{C['off']}")
        if preformatted:
            # keep JSON indentation exactly; wrapping it makes it unreadable
            print("\n".join(ind + " " + l for l in body.split("\n")))
        else:
            print(wrap(body, w - len(ind) - 2, ind + " "))
    print()


def render(rec, name_override, raw, show_preamble, icons=True, flags=True,
           panels=True, header=True):
    """`panels=False` omits the server and cost blocks; `header=False` omits the
    title, participants, preamble tree and tool list.

    The console draws its own header so the tree can be expanded in place and
    clicked. The TURNS stay shared, so the conversation itself can never differ
    between the console and `llamatap read`.

    The console pins those in its own dashboard, so emitting them here too put
    them inside the scrolling conversation — twelve lines of duplicate chrome
    between every exchange."""
    w = width()
    preamble, turns = normalise(rec.get("req"))
    name = name_override or detect(preamble)
    turns = turns + from_response(rec.get("resp"))

    # Is this the harness talking to itself? If so the "user" turns were
    # written by the harness, not typed by anyone, and labelling them YOU is a
    # lie that makes a background call look like your own request.
    probe = preamble + " " + " ".join(t.get("text") or "" for t in turns if t["who"] == "you")
    bg_label, bg_icon = cat.background_kind(probe)
    if bg_label:
        # The harness is unchanged — it is opencode doing something on its own
        # behalf. Renaming the party to "title generator" invented a
        # participant that does not exist.
        name = f"{name} · background: {bg_label}"
        for t in turns:
            if t["who"] == "you":
                t["kind"] = "background"

    model = (rec.get("req") or {}).get("model") or "the model"
    if not header:
        for t in turns:
            n = len(t.get("text") or "")
            show(t, name, w, raw, icons, model, f"  {n:,} chars · ~{n//4:,} tokens")
        if panels:
            if not raw:
                server_panel(model, w, show_flags=flags)
            stats_panel(rec, w, model)
        return len(turns)
    hdr = f" {name}  ↔  {model}   {rec.get('t','')} "
    if bg_label and not raw:
        hdr = f" {bg_icon}{name}  ↔  {model}   {rec.get('t','')} "
    print(f"{C['bold']}{hdr.center(w, '=')}{C['off']}\n")

    # Who put text into this exchange. Four parties, not two — and
    # llama-server is the one people forget: it renders the whole `tools` array
    # into prose inside the prompt, which is why 11 tool definitions cost
    # thousands of tokens.
    if not raw:
        build = ""
        try:
            build = (props(model) or {}).get("build_info", "")
        except Exception:
            pass
        P = cat.PARTY
        short = name.split(" · ")[0]
        # Party icons are empty now, so join the names directly rather than
        # leaving the gaps where pictures used to be.
        print(f"{C['label']}  in this thread:{C['off']} "
              f"you · {short} (harness) · "
              f"llama-server {build or '(version unknown)'} · "
              f"{model}\n")

    if preamble:
        n = len(preamble)
        if show_preamble:
            show({"who": "harness", "kind": "", "text": preamble},
                 f"{name} → system prompt", w, raw, icons, model)
        elif raw:
            print(f"  {name} system prompt: {n:,} chars, ~{n//4:,} tokens\n")
        else:
            lead = (cat.PARTY["harness"].rstrip() + " ") if icons else ""
            sname = name.split(" · ")[0]
            rule = "─" * max(0, w - dwidth(f"{lead}{sname} → sets up the request") - 10
                             - dwidth(f"{n:,} chars · ~{n//4:,} tokens"))
            print(f"{C['harness']}── {lead}{sname} → sets up the request {rule}{C['off']}"
                  f"{C['label']}  {n:,} chars · ~{n//4:,} tokens{C['off']}")
            print(f"{C['label']}     system prompt{C['off']}")
            # Sections from the prompt's OWN headings and tag opens, the same
            # source the console uses. This used to call a hardcoded list of
            # opencode's phrases, so the two front ends disagreed about the
            # structure of the same prompt — and on any other harness the
            # reader silently found nothing.
            found = cat.sections(preamble)
            for k, sec in enumerate(found):
                # Plain indentation; the tree glyphs said what the indent
                # already said.
                branch = " "
                tip = f"{cat.section_icon(sec['label'])} " if icons else ""
                pct = (100 * sec["length"] // n) if n else 0
                print(f"{C['label']}        {branch} {tip}{sec['label'][:30]:<32}{C['off']}"
                      f" {sec['length']:,} chars · {pct}% of the prompt")
            print()

    tools = rec.get("req", {}).get("tools") if isinstance(rec.get("req"), dict) else None
    if tools:
        names = []
        for t in tools:
            if isinstance(t, dict):
                names.append(t.get("name") or (t.get("function") or {}).get("name"))
        pretty = []
        for n_ in names:
            if not n_:
                continue
            if icons and not raw:
                _c, _i = cat.classify_tool(n_)
                pretty.append(f"{_i}{n_}")
            else:
                pretty.append(n_)
        lead = (cat.TOOLBOX + " ") if icons and not raw else ""
        print(f"{C['label']}     {lead}{len(tools)} tools{C['off']} {'  '.join(pretty)}\n")

    meta = rec.get("meta") or {}
    tmg = meta.get("timings") or {}
    last_model_turn = max((i for i, t in enumerate(turns) if t["who"] == "model"),
                          default=-1)
    for i, t in enumerate(turns):
        n = len(t.get("text") or "")
        stat = f"  {n:,} chars · ~{n//4:,} tokens"
        if i == last_model_turn and tmg.get("predicted_n"):
            stat = (f"  {tmg['predicted_n']:,} tokens · "
                    f"{tmg.get('predicted_ms', 0)/1000:.1f}s · "
                    f"{tmg.get('predicted_per_second', 0):.0f} t/s")
        show(t, name, w, raw, icons, model, stat)
    if panels:
        if not raw:
            server_panel(model, w, show_flags=flags)
        stats_panel(rec, w, model)
    return len(turns)


def harness_of(rec):
    pre, _ = normalise(rec.get("req"))
    return detect(pre)


def cmd_list(recs, w):
    """Numbered index of the recording. `replay` takes these numbers."""
    print(f"{C['bold']}  #  time      model                 path                  msgs  "
          f"prompt  out  tools  harness{C['off']}")
    print(f"{C['rule']}{'─' * w}{C['off']}")
    for i, rec in enumerate(recs, 1):
        req = rec.get("req") or {}
        meta = rec.get("meta") or {}
        u = meta.get("usage") or {}
        tm = meta.get("timings") or {}
        msgs = req.get("messages") or req.get("input") or []
        tools = req.get("tools") or []
        pt = u.get("prompt_tokens") or u.get("input_tokens") or tm.get("prompt_n") or ""
        ct = u.get("completion_tokens") or u.get("output_tokens") or tm.get("predicted_n") or ""
        t = (rec.get("t") or "")[-8:]
        tag = ""
        if meta.get("replay_of"):
            tag = f" {C['tool']}(replay of #{meta['replay_of']}){C['off']}"
        print(f"  {i:<2} {t}  {str(req.get('model',''))[:20]:<20}  "
              f"{str(rec.get('path',''))[:20]:<20}  {len(msgs):>4}  "
              f"{str(pt):>6}  {str(ct):>3}  {len(tools):>5}  {harness_of(rec)}{tag}")
    print(f"{C['dim']}{'─' * w}\n  {len(recs)} exchanges. "
          f"llamatap read N  |  llamatap replay N --model <other>{C['off']}")


HELP = "r raw/pretty   i icons   p preamble   f flags   l list   c clear   q quit"


def status(opts, n, w):
    """One line pinned under the output saying what mode you are in.

    Without this the toggles are invisible until the next request arrives,
    which makes the keybindings feel broken.
    """
    mode = "RAW" if opts["raw"] else "pretty"
    bits = [f"{C['bold']}{mode}{C['off']}"]
    bits.append(("icons on" if opts["icons"] else "icons off"))
    bits.append(("preamble FULL" if opts["preamble"] else "preamble summary"))
    bits.append(("flags on" if opts["flags"] else "flags off"))
    bits.append(f"{n} exchanges")
    print(f"{C['rule']}{'─' * w}{C['off']}")
    print(f"  {'   '.join(bits)}")
    print(f"{C['label']}  {HELP}{C['off']}")


def console_loop(path, opts, name_override):
    """Live view: render each exchange as it arrives, toggle format with a key.

    Deliberately not curses. A scrolling transcript is the right shape for a
    conversation — you want to scroll back through it with the terminal's own
    buffer — so this only adds single-key input on top of plain printing.
    """
    # line buffering: a live console must flush as it goes, and when stdout is
    # a pipe Python buffers 8 KB by default, so output appears only at exit
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    w = width()
    recs = records(path) if os.path.exists(path) else []
    seen = len(recs)
    last = recs[-1] if recs else None

    def draw(rec):
        render(rec, name_override, opts["raw"], opts["preamble"], opts["icons"], opts["flags"])

    print(f"{C['bold']}{' llamatap console ':=^{w}}{C['off']}")
    print(f"{C['dim']}  watching {path}{C['off']}\n")
    if last:
        draw(last)
    status(opts, seen, w)

    fd = sys.stdin.fileno()
    interactive = sys.stdin.isatty()
    old = termios.tcgetattr(fd) if interactive else None
    try:
        if interactive:
            tty.setcbreak(fd)
        while True:
            if interactive and select.select([sys.stdin], [], [], 0.4)[0]:
                ch = sys.stdin.read(1)
                if ch in ("q", "\x03", "\x04"):
                    break
                redraw = True
                if ch == "r":
                    opts["raw"] = not opts["raw"]
                elif ch == "i":
                    opts["icons"] = not opts["icons"]
                elif ch == "p":
                    opts["preamble"] = not opts["preamble"]
                elif ch == "f":
                    opts["flags"] = not opts["flags"]
                elif ch == "c":
                    print("\033[2J\033[H", end="")
                elif ch == "l":
                    cmd_list(records(path), w)
                    redraw = False
                else:
                    redraw = False
                if redraw and last is not None:
                    # re-render the SAME exchange in the new mode, so a toggle
                    # has an immediate visible effect instead of waiting for
                    # the next request
                    print()
                    draw(last)
                status(opts, seen, w)
            else:
                time.sleep(0.1)

            if not os.path.exists(path):
                continue
            recs = records(path)
            if len(recs) > seen:
                while seen < len(recs):
                    print()
                    draw(recs[seen])
                    last = recs[seen]
                    seen += 1
                status(opts, seen, w)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive and old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


def header_parts(rec, name_override=None):
    """The facts a header is made of, without any formatting.

    Returned so an interactive front end can lay them out as a clickable tree
    rather than parsing them back out of rendered text.
    """
    req = rec.get("req") or {}
    preamble, turns = normalise(req)
    name = name_override or detect(preamble)
    probe = preamble + " " + " ".join(t.get("text") or "" for t in turns if t["who"] == "you")
    bg_label, bg_icon = cat.background_kind(probe)
    if bg_label:
        # Same harness, acting on its own behalf — not a new participant.
        # A caller that knows which harness this session belongs to passes it
        # as name_override; otherwise detection genuinely cannot tell.
        name = f"{name} · background: {bg_label}"
    tools, tools_chars = [], 0
    for t in (req.get("tools") or []):
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or t
        n = fn.get("name")
        if n:
            size = len(json.dumps(t))
            tools_chars += size
            tools.append((n, (fn.get("description") or "").strip().split("\n")[0], size))
    return {"name": name, "model": req.get("model") or "the model",
            "t": rec.get("t", ""), "preamble": preamble,
            "sections": cat.sections(preamble),
            "tools": tools, "tools_chars": tools_chars, "background": bg_label}


def conversation(recs, name_hint=None, show_repeats=False):
    """Flatten every recorded exchange into ONE chronological group chat.

    The hard part is de-duplication. Every request re-sends the whole history,
    so rendering exchanges one after another repeats the entire conversation
    once per turn. Here each message is emitted the FIRST time it appears and
    never again, which is what makes four parties talking to each other read as
    a single thread rather than a stack of transcripts.

    A background request (a title generator, a summariser) is still the harness
    speaking — it just carries its own system prompt, so it shows up as the
    harness saying something on its own behalf, in sequence, where it happened.
    """
    seen = set()
    msgs = []
    last_setup = None
    harness = name_hint
    # The MAIN thread, for telling a harness's side-calls apart structurally.
    #
    # Deciding that from wording — "Generate a title for this conversation" and
    # ten other English phrases — only works for harnesses whose phrasing has
    # been seen before. On any other one the side-call fell through to
    # `who: "you"`, so the console claimed the USER had typed the harness's
    # internal prompt. For a tool that exists to be pointed at an unfamiliar
    # harness, guessing wrong there is the worst failure available.
    #
    # Two observable facts decide it instead, and neither depends on language:
    #   1. the main thread re-sends its history and ADDS to it, so every
    #      message said so far reappears in the next request; a side-call
    #      carries at most a quote of it
    #   2. a side-call carries its own system prompt
    # Both are required. A harness that legitimately changes its system prompt
    # mid-conversation still satisfies (1), so it stays in the main thread;
    # one that sends no history at all fails (2) and stays there too.
    main_texts = set()
    main_pre = None

    for ri, rec in enumerate(recs):
        h = header_parts(rec, name_hint)
        short = h["name"].split(" · ")[0]
        if short == "HARNESS" and harness:
            short = harness
            h = header_parts(rec, harness)
        elif not h.get("background"):
            harness = short
        t = rec.get("t", "")
        meta = rec.get("meta") or {}
        tmg = meta.get("timings") or {}
        if meta.get("injected"):
            # sent by llamatap itself, not by the harness — say so, or the
            # recording claims the harness did something it never did
            short = f"{short} + llamatap"

        # the harness assembling the request is itself a message
        req_pre, req_turns = normalise(rec.get("req"))
        said = {(x.get("who"), (x.get("text") or "").strip())
                for x in req_turns if (x.get("text") or "").strip()}
        if main_pre is None:
            side = False            # the first request defines the main thread
        else:
            # A main-thread request EXTENDS the conversation: it re-sends
            # everything said so far and adds to it. A side-call does not.
            # "Shares no messages at all" was too strict — a title generator
            # quotes the user's own words back, so it overlapped by one and
            # was read as part of the thread.
            side = (req_pre.strip() != main_pre
                    and not main_texts.issubset(said))
        if side:
            # `background_kind` is now only asked what to CALL it. An
            # unrecognised side-call is still a side-call; it just goes
            # unnamed rather than being mistaken for the user.
            h["background"] = h.get("background") or "side call"
        else:
            h["background"] = None
            main_texts |= said
            main_pre = req_pre.strip()

        sig = (len(h["preamble"]), h.get("tools_chars", 0), bool(h.get("background")))
        if sig != last_setup:
            total = len(h["preamble"]) + h.get("tools_chars", 0)
            # What is OBSERVABLE, and therefore true for any harness:
            #   - the system prompt changed from the previous request
            #   - this request offers no tools although an earlier one did
            # What is a GUESS: the name, from matching known prompt wording.
            # Keep them apart so a harness we have never seen still gets the
            # structural facts right.
            msgs.append({"t": t, "who": "harness", "name": short, "rec": ri,
                         "kind": "setup", "text": "", "chars": total,
                         "own_system": last_setup is not None
                                       and sig[0] != last_setup[0],
                         "no_tools": not h["tools"] and last_setup is not None
                                     and last_setup[1] > 0,
                         "background": h.get("background"),
                         "preamble_chars": len(h["preamble"]),
                         "tools_chars": h.get("tools_chars", 0),
                         "sections": h["sections"], "tools": h["tools"],
                         "prefill_ms": tmg.get("prompt_ms"),
                         "prefill_n": tmg.get("prompt_n"),
                         "prefill_tps": tmg.get("prompt_per_second")})
            last_setup = sig

        _, turns = normalise(rec.get("req"))
        turns = turns + from_response(rec.get("resp"))
        # Decode is one continuous run at a near-constant rate, so a turn's
        # share of the output tokens gives its share of the time. Marked `~`
        # because it is apportioned, not separately measured.
        per_tok_ms = tmg.get("predicted_per_token_ms")
        probe = h["preamble"] + " " + " ".join(x.get("text") or "" for x in turns
                                               if x["who"] == "you")
        bg, _ = cat.background_kind(probe)
        last_model = max((i for i, x in enumerate(turns) if x["who"] == "model"),
                         default=-1)

        # Count what this request RE-SENT rather than silently skipping it:
        # every request carries the whole history, which is most of the token
        # bill, and that fact should be visible even though re-printing the
        # text would make the thread unreadable.
        # Nothing on the wire says "this is a repeat". The proxy sees new text
        # every single time; identifying a repeat is string comparison done
        # here, nothing more. So show the text — each repeated message is
        # emitted in place, marked `repeat`, rather than summarised into a
        # count that claims knowledge the protocol does not carry.

        for i, turn in enumerate(turns):
            text = turn.get("text") or ""
            key = (turn["who"], turn.get("kind"), turn.get("tool"), text)
            # Every request re-sends the whole history, so showing repeats
            # renders the entire conversation again on each turn — a
            # 25-exchange session draws itself 25 times. Hidden by default,
            # and the cost of re-sending is still visible in the dashboard's
            # `kv cache N reused`. `show_repeats` puts them back, marked.
            repeat = key in seen
            if repeat and not show_repeats:
                continue
            seen.add(key)
            # What a repeat costs. Context space always; TIME only when the
            # prefix cache missed it. Both are reported, because "cached" is
            # what makes long sessions survivable and is worth seeing.
            if repeat:
                tps_in = tmg.get("prompt_per_second")
                cached = (tmg.get("cache_n") or 0) > 0
            m = {"t": t, "rec": ri, "name": short, "text": text,
                 "who": turn["who"], "kind": turn.get("kind") or "",
                 "tool": turn.get("tool"), "chars": len(text), "repeat": repeat}
            if repeat:
                m["prefill_tps"] = tps_in
                m["cached"] = cached
            if turn["who"] == "you" and bg:
                m["who"], m["kind"] = "harness", "background"
            if turn["who"] == "model" and per_tok_ms:
                m["secs"] = (len(text) / 4) * per_tok_ms / 1000
                m["secs_estimated"] = True
                m["tps"] = tmg.get("predicted_per_second", 0)
            if i == last_model and tmg.get("predicted_n"):
                m["tokens"] = tmg["predicted_n"]
                m["secs"] = tmg.get("predicted_ms", 0) / 1000
                m["secs_estimated"] = False
                m["tps"] = tmg.get("predicted_per_second", 0)
            msgs.append(m)

        # llama-server's own voice. A non-2xx response produced NO message at
        # all, so a 500, a 503 or a refused connection left a silent gap in
        # the thread — the one failure mode this tool exists to catch was the
        # one thing it could not show. It is also what makes the server a
        # participant rather than a name in the header that filters nothing.
        st = rec.get("status")
        if isinstance(st, int) and not 200 <= st < 300:
            why = (rec.get("error") or "").strip()
            if not why:
                raw = rec.get("resp")
                why = raw.strip() if isinstance(raw, str) else ""
            # Bounded. An upstream that returns an HTML error page would
            # otherwise paste the whole page into the conversation.
            if len(why) > 2000:
                why = why[:2000] + f"\n… {len(why) - 2000:,} more characters"
            msgs.append({"t": t, "rec": ri, "name": "llama-server",
                         "who": "server", "kind": "error",
                         "text": f"HTTP {st}\n{why or 'no response body'}",
                         "chars": len(why), "repeat": False})
    return msgs


def records(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="raw JSON bodies instead of prose")
    ap.add_argument("--all", action="store_true", help="every request, not just the last")
    ap.add_argument("--follow", "-f", action="store_true", help="live, as new exchanges arrive")
    ap.add_argument("--preamble", action="store_true", help="print the system prompt in full")
    ap.add_argument("--name", help="harness label, if detection gets it wrong")
    ap.add_argument("n", nargs="?", type=int, help="exchange number from `llamatap list`")
    ap.add_argument("--list", action="store_true", help="numbered index of the recording")
    ap.add_argument("--no-icons", action="store_true", help="plain labels, no icons")
    ap.add_argument("--no-flags", action="store_true", help="skip the SSH read of launch flags")
    ap.add_argument("--file", default=REC)
    a = ap.parse_args()
    if not sys.stdout.isatty():
        nocolor()

    if a.follow:
        opts = {"raw": a.raw, "icons": not a.no_icons,
                "preamble": a.preamble, "flags": not a.no_flags}
        console_loop(a.file, opts, a.name)
        return

    recs = records(a.file)
    if a.list:
        if not recs:
            sys.exit("recording is empty")
        cmd_list(recs, width())
        return
    if a.n:
        if not 1 <= a.n <= len(recs):
            sys.exit(f"no exchange #{a.n} — the recording has {len(recs)}")
        render(recs[a.n - 1], a.name, a.raw, a.preamble, not a.no_icons, not a.no_flags)
        return
    if not recs:
        sys.exit("recording is empty")
    for r in (recs if a.all else recs[-1:]):
        render(r, a.name, a.raw, a.preamble, not a.no_icons, not a.no_flags)


if __name__ == "__main__":
    main()
