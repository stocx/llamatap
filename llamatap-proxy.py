#!/usr/bin/env python3
"""Logging passthrough proxy for the Anthropic /v1/messages endpoint.

Sits between a client (Claude Code) and llama-swap, writes every request body
to a log, and forwards the response verbatim including SSE streams. Written
2026-09-18 to capture the exact request shape that makes llama-server's chat
template raise.

    llamatap serve                 listens on 9293, forwards to model-host :9292
    llamatap log

Works with ANY harness that speaks an OpenAI-compatible, Anthropic, or
Responses-API endpoint — point the harness at :9293 instead of :9292 and
nothing else changes. The two body rewrites below are self-guarding and are
no-ops on OpenAI-shaped requests.

Every request gets a compact summary in tap.log: preamble size, tool count and
names, message shape, and total tokens. LLAMATAP_FULL=1 also dumps the whole request
body. Full bodies are always kept in llamatap-records.jsonl regardless.
"""
import http.server, json, os, sys, threading, time, urllib.request, urllib.error

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
UPSTREAM = os.environ.get("LLAMATAP_UPSTREAM", _UPSTREAM_DEFAULT)
LOGFILE = os.environ.get("LLAMATAP_LOG", os.path.join(_STATE, "llamatap.log"))
RECFILE = os.environ.get("LLAMATAP_REC", os.path.join(_STATE, "llamatap-records.jsonl"))
# In-flight status, so a viewer can show that something is HAPPENING. Records
# are only written when an exchange completes, so without this the console has
# nothing to report during the 50 seconds a long reasoning reply takes — which
# is exactly when you want to know it is alive.
INFLIGHT = os.environ.get("LLAMATAP_INFLIGHT",
                          os.path.join(_STATE, "llamatap-inflight.json"))


_active = {}
_active_lock = threading.Lock()


def _publish():
    try:
        tmp = INFLIGHT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(list(_active.values()), f)
        os.replace(tmp, INFLIGHT)      # atomic: a reader never sees half a file
    except Exception:
        pass


def inflight_set(key, **kw):
    with _active_lock:
        _active.setdefault(key, {}).update(kw)
        _publish()


def inflight_clear(key):
    with _active_lock:
        _active.pop(key, None)
        _publish()
PORT = int(os.environ.get("LLAMATAP_PORT", "9293"))


def _est_tok(n_chars):
    """Rough token estimate. ~4 chars/token is close enough to compare
    harnesses against each other, which is all this is used for."""
    return n_chars // 4


def _text_len(v):
    """Character count of a content field, whichever shape it arrives in.

    OpenAI sends a plain string; Anthropic and newer OpenAI send a list of
    {"type":"text","text":...} blocks. Both appear here because every harness
    in this project picks its own.
    """
    if v is None:
        return 0
    if isinstance(v, str):
        return len(v)
    if isinstance(v, list):
        n = 0
        for b in v:
            if isinstance(b, dict):
                n += len(b.get("text") or "")
                if b.get("type") in ("tool_use", "tool_result"):
                    n += len(json.dumps(b))
            else:
                n += len(str(b))
        return n
    return len(json.dumps(v))


def timings_from(resp_bytes):
    """llama.cpp returns its own `timings` block: prompt_n/prompt_ms and
    predicted_n/predicted_ms, i.e. real prefill and decode rates measured
    inside the server. Always prefer these over wall-clock arithmetic here,
    which cannot separate prefill from decode on a non-streamed response."""
    if not resp_bytes:
        return None
    txt = resp_bytes.decode("utf8", "replace")
    try:
        d = json.loads(txt)
        if isinstance(d, dict) and d.get("timings"):
            return d["timings"]
    except Exception:
        pass
    best = None
    for line in txt.splitlines():
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
        if isinstance(d, dict) and d.get("timings"):
            best = d["timings"]
    return best


def usage_from(resp_bytes):
    """Pull the REAL token counts out of the response.

    The summary above estimates from character counts, which undercounts badly
    — the chat template adds role markers and renders the whole `tools` array
    into prose, so a 130-token estimate measured 329 actual. These are the
    numbers llama-server itself reports, so they are the ones to trust.

    Handles a plain JSON body and an SSE stream, where usage arrives in one of
    the final `data:` chunks.
    """
    if not resp_bytes:
        return None
    txt = resp_bytes.decode("utf8", "replace")
    try:
        d = json.loads(txt)
        if isinstance(d, dict) and d.get("usage"):
            return d["usage"]
    except Exception:
        pass
    best = None
    for line in txt.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            d = json.loads(payload)
        except Exception:
            continue
        u = d.get("usage") if isinstance(d, dict) else None
        if u:
            best = u
    return best


def summarise(path, body):
    """One compact block per request — what the HARNESS sent, not what
    llama-server made of it.

    This is the whole reason the proxy exists rather than tailing a server log:
    once the chat template has run, tool definitions and the system prompt are
    indistinguishable bytes in one long string. Here they are still separate.
    """
    try:
        d = json.loads(body) if body else None
    except Exception:
        return ["  (request body is not JSON)"]
    if not isinstance(d, dict):
        return []
    out = []
    model = d.get("model")
    if model:
        out.append(f"  model      {model}")

    # system prompt: top-level `system` (Anthropic), `instructions`
    # (Responses), or a leading system/developer message (OpenAI).
    sys_chars = _text_len(d.get("system")) + len(d.get("instructions") or "")
    msgs = d.get("messages") if isinstance(d.get("messages"), list) else []
    items = d.get("input") if isinstance(d.get("input"), list) else []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            sys_chars += _text_len(m.get("content"))
    if sys_chars:
        out.append(f"  preamble   {sys_chars} chars  ~{_est_tok(sys_chars)} tok   <- charged on EVERY request")

    tools = d.get("tools")
    if isinstance(tools, list):
        names = []
        tool_chars = len(json.dumps(tools))
        for t in tools:
            if not isinstance(t, dict):
                continue
            n = t.get("name") or (t.get("function") or {}).get("name")
            if n:
                names.append(n)
        out.append(f"  tools      {len(tools)}  ~{_est_tok(tool_chars)} tok  [{', '.join(names)}]")

    conv = msgs or items
    if conv:
        roles = {}
        for m in conv:
            if isinstance(m, dict):
                roles[m.get("role", "?")] = roles.get(m.get("role", "?"), 0) + 1
        shape = " ".join(f"{k}:{v}" for k, v in sorted(roles.items()))
        out.append(f"  messages   {len(conv)}  ({shape})")

    # the last user turn, which is what you actually typed
    for m in reversed(conv):
        if isinstance(m, dict) and m.get("role") == "user":
            t = _text_len(m.get("content"))
            txt = m.get("content")
            if isinstance(txt, list):
                txt = " ".join((b.get("text") or "") for b in txt if isinstance(b, dict))
            txt = (str(txt) or "").replace("\n", " ")[:120]
            out.append(f"  last user  {t} chars: {txt}")
            break

    total = len(body or b"")
    out.append(f"  TOTAL      {total} chars  ~{_est_tok(total)} tok")
    return out


def log(msg):
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")
        f.flush()


_cap = int(os.environ.get("LLAMATAP_CAP", "0") or 0)


def record(path, status, req_bytes, resp_bytes, meta=None):
    """One JSON object per exchange, for measure-agents.py to parse later.

    Each request carries the whole conversation so far, so the last record of
    a run reconstructs everything except the final assistant turn -- which is
    why the response body is kept too, capped so a long stream cannot fill
    the disk.
    """
    try:
        req = json.loads(req_bytes) if req_bytes else None
    except Exception:
        req = None
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "path": path,
           "status": status, "req": req, "meta": meta or {},
           # No cap. There used to be a 256 KB slice here, which chopped long
           # tool-call arguments mid-string — the JSON then failed to parse and
           # a 3,000-line file being written rendered as raw escaped text. The
           # arguments ARE the content worth reading.
           "resp": resp_bytes[:_cap].decode("utf8", "replace") if _cap
                   else resp_bytes.decode("utf8", "replace")}
    try:
        with open(RECFILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
    except Exception as e:
        log(f"[tap] record failed: {e}")


class Tap(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _proxy(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        url = UPSTREAM + self.path
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "accept-encoding")}
        body = self._hoist_system(body)
        body = self._merge_developer(body)
        if body:
            hdrs["Content-Length"] = str(len(body))
        req = urllib.request.Request(url, data=body or None, headers=hdrs, method=method)
        t0 = time.time()
        t_first = None
        key = f"{id(self)}-{t0}"
        model = None
        try:
            model = (json.loads(body) or {}).get("model") if body else None
        except Exception:
            pass
        inflight_set(key, t0=t0, model=model, path=self.path,
                     state="prefill", tokens=0, req_bytes=len(body or b""))
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            status, rhdrs = resp.status, resp.headers
        except urllib.error.HTTPError as e:
            resp, status, rhdrs = e, e.code, e.headers
            err = resp.read()
            log(f"\n===== {status} {method} {self.path} =====")
            if body:
                try:
                    log("REQUEST:\n" + json.dumps(json.loads(body), indent=2)[:20000])
                except Exception:
                    log("REQUEST (raw):\n" + body.decode("utf8", "replace")[:20000])
            log("RESPONSE:\n" + err.decode("utf8", "replace")[:4000])
            inflight_clear(key)
            record(self.path, status, body, err,
                   {"elapsed_ms": int((time.time() - t0) * 1000)})
            self.send_response(status)
            self.send_header("Content-Type", rhdrs.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        log(f"\n----- {status} {method} {self.path} -----")
        for line in summarise(self.path, body):
            log(line)
        if os.environ.get("LLAMATAP_FULL"):
            try:
                log("REQUEST (full):\n" + json.dumps(json.loads(body), indent=2))
            except Exception:
                pass
        self.send_response(status)
        for k, v in rhdrs.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        # Capture head AND tail. The old code kept only the first 256 KB, which
        # silently lost the END of a long SSE stream — and the final chunks are
        # exactly where `usage` and `timings` live. A 1,000-token reasoning
        # reply overflows this easily (one JSON object per token), so the stats
        # vanished on precisely the responses most worth measuring.
        # Nothing is dropped by default. An earlier version kept only the first
        # 256 KB and lost the end of long streams — including `usage` and
        # `timings`. Disk is cheap and these recordings are the evidence, so
        # the cap is off unless LLAMATAP_CAP is set to a byte count.
        _cap = int(os.environ.get("LLAMATAP_CAP", "0") or 0)
        HEAD_CAP = _cap if _cap else float("inf")
        TAIL_CAP = 131072
        ntok, last_pub = 0, 0.0
        # Decoded text so far, so a viewer can show the reply AS IT ARRIVES.
        # Records are only written when an exchange completes, so without this
        # the newest turn is invisible for as long as it takes to generate —
        # twenty seconds of nothing on a long reasoning reply.
        live_txt, live_think, sse_buf = "", "", ""
        seen = bytearray()
        tail = bytearray()
        overflowed = False
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            if t_first is None:
                t_first = time.time()
                inflight_set(key, state="decoding", ttft=t_first - t0)
            # One SSE object per token, so counting delta chunks is a close
            # enough live token counter for a progress display.
            ntok += chunk.count(b'"delta"')
            sse_buf += chunk.decode("utf8", "replace")
            lines = sse_buf.split("\n")
            sse_buf = lines.pop()          # keep the partial last line
            for ln in lines:
                ln = ln.strip()
                if not ln.startswith("data:"):
                    continue
                payload = ln[5:].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                for ch2 in (d.get("choices") or []):
                    dl = ch2.get("delta") or {}
                    if dl.get("content"):
                        live_txt += dl["content"]
                    if dl.get("reasoning_content"):
                        live_think += dl["reasoning_content"]
            if time.time() - last_pub > 0.3:
                last_pub = time.time()
                inflight_set(key, tokens=ntok, state="decoding",
                             text=live_txt[-1200:], thinking=live_think[-1200:])
            if len(seen) < HEAD_CAP:
                seen.extend(chunk)
            else:
                overflowed = True
                tail.extend(chunk)
                if len(tail) > TAIL_CAP:
                    del tail[:len(tail) - TAIL_CAP]
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        if overflowed:
            seen.extend(b"\n\n[... llamatap: middle of response omitted, "
                        b"head and tail kept ...]\n\n")
            seen.extend(tail)
        t_end = time.time()
        meta = {"elapsed_ms": int((t_end - t0) * 1000),
                "ttft_ms": int(((t_first or t_end) - t0) * 1000)}
        u = usage_from(bytes(seen))
        if u:
            pt = u.get("prompt_tokens") or u.get("input_tokens")
            ct = u.get("completion_tokens") or u.get("output_tokens")
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            line = f"  ACTUAL     prompt {pt} tok, completion {ct} tok"
            if cached is not None and pt:
                fresh = pt - cached
                pct = (100 * cached // pt) if pt else 0
                line += f"  |  cached {cached} ({pct}%), fresh prefill {fresh}"
            # Decode rate over the generation phase only: total wall time minus
            # the wait for the first byte, which is prefill. Reported as
            # wall-clock from this proxy's view, not llama-server's internal
            # timings, so it includes network — negligible on a LAN.
            meta["usage"] = u
            tm = timings_from(bytes(seen))
            if tm:
                meta["timings"] = tm
            else:
                # Fallback only. A non-streamed response arrives in one burst,
                # so t_first == t_end and a decode rate cannot be derived —
                # report overall throughput instead of a fictional one.
                gen_s = (t_end - (t_first or t_end))
                if ct and gen_s > 0.05:
                    meta["decode_tps"] = round(ct / gen_s, 2)
                elif ct:
                    meta["overall_tps"] = round(ct / max(0.001, t_end - t0), 2)
            log(line)
            tm = meta.get("timings") or {}
            if tm:
                log(f"  TIMING     prefill {tm.get('prompt_n')} tok at "
                    f"{tm.get('prompt_per_second', 0):.0f} tok/s, decode "
                    f"{tm.get('predicted_n')} tok at "
                    f"{tm.get('predicted_per_second', 0):.1f} tok/s")
            else:
                log(f"  TIMING     {meta['ttft_ms']} ms to first token, "
                    f"{meta['elapsed_ms']} ms total")
        inflight_clear(key)
        record(self.path, status, body, bytes(seen), meta)


    def _hoist_system(self, body):
        """Move any non-leading system-role message into the top-level system.

        Claude Code v2.1.277 emits {"role":"system","text":"Today's date is ..."}
        as messages[1]. Qwen3.5's chat template raises "System message must be
        at the beginning." The date is already duplicated in the top-level
        system prompt, so hoisting it changes nothing the model sees in
        substance. Logged on every rewrite so runs stay auditable.
        """
        if not body:
            return body
        try:
            d = json.loads(body)
        except Exception:
            return body
        msgs = d.get("messages")
        if not isinstance(msgs, list):
            return body
        moved, kept = [], []
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") == "system" and i > 0:
                moved.append(m)
            else:
                kept.append(m)
        if not moved:
            return body
        sys_field = d.get("system")
        blocks = []
        for m in moved:
            c = m.get("content")
            blocks.extend(c if isinstance(c, list) else [{"type": "text", "text": str(c)}])
        if isinstance(sys_field, list):
            d["system"] = sys_field + blocks
        elif isinstance(sys_field, str):
            d["system"] = [{"type": "text", "text": sys_field}] + blocks
        else:
            d["system"] = blocks
        d["messages"] = kept
        log(f"[shim] hoisted {len(moved)} system-role message(s) out of messages")
        return json.dumps(d).encode()


    def _merge_developer(self, body):
        """Fold Responses-API developer/system input items into `instructions`.

        Codex 0.155 sends `instructions` (its system prompt) plus an input item
        with role "developer" carrying skills instructions. llama.cpp maps both
        to system messages, producing two systems in a row, and Qwen3.5's
        template raises "System message must be at the beginning." on the
        second. Merging the text into `instructions` keeps every word the model
        would have seen, in the same order, as a single system message.
        """
        if not body:
            return body
        try:
            d = json.loads(body)
        except Exception:
            return body
        items = d.get("input")
        if not isinstance(items, list):
            return body
        extra, kept = [], []
        for it in items:
            if isinstance(it, dict) and it.get("role") in ("developer", "system"):
                for c in it.get("content") or []:
                    t = c.get("text") if isinstance(c, dict) else None
                    if t:
                        extra.append(t)
            else:
                kept.append(it)
        if not extra:
            return body
        instr = d.get("instructions") or ""
        d["instructions"] = (instr + "\n\n" + "\n\n".join(extra)).strip()
        d["input"] = kept
        log(f"[shim] merged {len(extra)} developer block(s) into instructions")
        return json.dumps(d).encode()

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")

    def log_message(self, *a):
        pass


class QuietServer(http.server.ThreadingHTTPServer):
    """Suppress the traceback when a client hangs up.

    ConnectionResetError / BrokenPipeError here mean the HARNESS closed the
    socket — a cancelled request, a timeout, a keepalive expiring. That is
    normal traffic, not a fault in the proxy, and dumping a 20-line traceback
    into the operator's terminal makes a healthy session look broken. Logged to
    the file instead so it is still visible if you go looking.
    """
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            log(f"[tap] client {client_address[0]}:{client_address[1]} hung up "
                f"({type(exc).__name__}) — normal if a request was cancelled")
            return
        super().handle_error(request, client_address)


def _count_records():
    try:
        with open(RECFILE) as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


if __name__ == "__main__":
    # Bind BEFORE announcing anything: a cheerful startup banner followed by a
    # bind failure reads as though the server came up and then died.
    try:
        srv = QuietServer(("127.0.0.1", PORT), Tap)
    except OSError as e:
        if e.errno != 98:
            raise
        # "Address already in use" with a traceback tells you nothing. The
        # cause is almost always a llamatap left running from earlier, and the
        # fix needs its pid — so find it and say so.
        import subprocess as _sp
        pids = ""
        try:
            n = "llamatap"; n = n + "-proxy"
            pids = _sp.run(["pgrep", "-f", n], capture_output=True, text=True).stdout.split()
            pids = " ".join(p for p in pids if p != str(os.getpid()))
        except Exception:
            pass
        print(f"\nport {PORT} is already in use — another llamatap is probably running."
              f"\n", file=sys.stderr)
        if pids:
            print(f"  kill {pids}", file=sys.stderr)
        else:
            print(f"  pgrep -f llamatap-proxy     then  kill <pid>", file=sys.stderr)
        print(f"\n  Or run this one elsewhere:  LLAMATAP_PORT=9294 llamatap serve",
              file=sys.stderr)
        sys.exit(1)

    # Start clean by default. A session's recording is almost never something
    # you want to continue into the next one, and a mixed recording is worse
    # than useless — exchange numbers shift and two harnesses blur together.
    # Nothing is deleted: the old one is archived and the path is printed.
    if "--keep" not in sys.argv and not os.environ.get("LLAMATAP_KEEP"):
        try:
            have = [f for f in (RECFILE, LOGFILE)
                    if os.path.exists(f) and os.path.getsize(f) > 0]
            if have:
                stamp = time.strftime("%m%d-%H%M")
                dest = os.path.join(os.path.dirname(RECFILE), "runs",
                                    f"llamatap-{stamp}")
                os.makedirs(dest, exist_ok=True)
                for f in have:
                    os.replace(f, os.path.join(dest, os.path.basename(f)))
                print(f"archived the previous recording to {dest}", flush=True)
                print(f"  read it with: llamatap read --file "
                      f"{os.path.join(dest, os.path.basename(RECFILE))}", flush=True)
        except Exception as e:
            print(f"could not archive the previous recording: {e}", file=sys.stderr)

    print(f"tap: 127.0.0.1:{PORT} -> {UPSTREAM}", flush=True)
    print(f"log: {LOGFILE}", flush=True)
    print("     nothing prints here — watch it live with `llamatap console`", flush=True)
    print("     ctrl-c to stop", flush=True)
    try:
        os.remove(INFLIGHT)        # a crash can leave a stale marker behind
    except Exception:
        pass
    before = _count_records()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        # ctrl-c is the intended way to stop this; a traceback here is noise,
        # not an error.
        pass
    finally:
        srv.server_close()
        try:
            os.remove(INFLIGHT)
        except Exception:
            pass
        n = _count_records()
        print(f"\nstopped. {n - before} exchange(s) recorded this session, "
              f"{n} in {os.path.basename(RECFILE)}", flush=True)
        print("     llamatap list   |   llamatap read   |   llamatap console", flush=True)
