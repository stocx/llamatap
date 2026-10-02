#!/usr/bin/env python3
"""Re-send a recorded request, optionally to a different model.

This is what turns the recording from a transcript into an experiment. A
captured request is the exact prompt a harness built — every token of preamble,
every tool definition, the whole conversation so far. Replaying it against a
second model or a second set of launch flags changes ONE variable and leaves
everything else byte-identical, which hand-driving a harness cannot do.

    llamatap list
    llamatap replay 3
    llamatap replay 3 --model qwen3.5-9b-nothink
    llamatap replay 3 --model qwen3.5-9b --model qwen3.5-9b-nothink --diff

Replays are appended to the recording with meta.replay_of set, so they show up
in `llamatap list` and can be read with `llamatap read N` like anything else.

Streaming is forced OFF for a replay: a non-streamed response returns one JSON
body carrying llama.cpp's `timings` block, which is what the comparison needs.
The original request's own `stream` setting is left untouched in the record.
"""
import argparse, importlib.util as _ilu, json, os, sys, time
import urllib.request, urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "llamatap_thread", os.path.join(_HERE, "llamatap-thread.py"))
T = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(T)

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
UPSTREAM = os.environ.get("LLAMATAP_UPSTREAM", _UPSTREAM_DEFAULT)

# Same rule as the renderer: bright black is unreadable on a dark theme, so it
# is used for separator rules only and never for text a human must read.
C = {"hdr": "\033[1m", "model": "\033[33m", "dim": "", "think": "\033[94m",
     "rule": "\033[90m", "good": "\033[32m", "bad": "\033[31m", "off": "\033[0m"}
if not sys.stdout.isatty():
    for k in C:
        C[k] = ""


def load(path):
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
        sys.exit(f"no recording at {path}")
    return out


def append_record(path, status, req, resp, meta, path_=None, **kw):
    """Append one replay exchange to a recording.

    No cap on `resp`. There used to be a `[:262144]` slice here — the same one
    removed from the proxy, left behind in this file because nothing re-checked
    it. It cut the JSON mid-document, so a long generated file became
    unparsable and later readers lost the entire answer rather than its tail.
    """
    dest = kw.get("path") or path_ or REC
    line = json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "path": path, "status": status, "req": req,
                       "meta": meta, "resp": resp}) + "\n"
    newfile = not os.path.exists(dest)
    with open(dest, "a") as f:
        f.write(line)
    if newfile:
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass


def reply_of(resp_text):
    """(text, reasoning, [tool call names], finish_reason) from a response.

    Delegates to the reader's parser rather than keeping a second one. This
    was `json.loads` only, so every SSE-framed original — which is what any
    streaming harness produces, 85 of 85 real records in one sample — showed
    as an empty response. `replay --diff` was therefore comparing against a
    blank. Two parsers for one format is how they drifted apart; now there is
    one.
    """
    turns = T.from_response(resp_text)
    text = "\n".join(t.get("text") or "" for t in turns
                      if t.get("who") == "model" and t.get("kind") == "")
    reasoning = "\n".join(t.get("text") or "" for t in turns
                          if t.get("kind") == "thinking")
    calls = [t.get("tool") for t in turns
             if t.get("kind") == "tool call" and t.get("tool")]
    finish = next((t.get("finish") for t in turns if t.get("finish")), None)
    if finish is None:
        try:
            d = json.loads(resp_text)
            finish = ((d.get("choices") or [{}])[0]).get("finish_reason")
        except Exception:
            finish = None
    return text.strip(), reasoning.strip(), calls, finish


def send(req_body, path, upstream):
    url = upstream.rstrip("/") + path
    data = json.dumps(req_body).encode()
    r = urllib.request.Request(url, data=data,
                               headers={"Content-Type": "application/json",
                                        "Authorization": "Bearer dummy"},
                               method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=900) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    except Exception as e:
        return None, 0, int((time.time() - t0) * 1000), str(e)
    return body.decode("utf8", "replace"), status, int((time.time() - t0) * 1000), None


def summarise(label, resp_text, elapsed_ms, w):
    text, reasoning, calls, finish = reply_of(resp_text)
    try:
        d = json.loads(resp_text)
    except Exception:
        d = {}
    tm = d.get("timings") or {}
    u = d.get("usage") or {}
    print(f"{C['model']}  ── {label} {'─' * max(0, w - len(label) - 6)}{C['off']}")
    if reasoning.strip():
        print(f"{C['think']}     💭 {reasoning.strip()[:400]}{C['off']}")
    if text.strip():
        print(f"     {text.strip()[:600]}")
    if calls:
        print(f"     🔧 called: {', '.join(calls)}")
    if not text.strip() and not calls and not reasoning.strip():
        print(f"{C['bad']}     (empty response){C['off']}")
    bits = []
    if tm.get("prompt_n") is not None:
        bits.append(f"prefill {tm['prompt_n']} new tok at {tm.get('prompt_per_second', 0):,.0f} t/s")
    if tm.get("predicted_n") is not None:
        bits.append(f"decode {tm['predicted_n']} tok at {tm.get('predicted_per_second', 0):.1f} t/s")
    if u.get("prompt_tokens"):
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        bits.append(f"context {u['prompt_tokens']} tok"
                    + (f", {cached} cached" if cached is not None else ""))
    bits.append(f"{elapsed_ms/1000:.1f}s wall")
    if finish and finish not in ("stop", "tool_calls"):
        bits.append(f"{C['bad']}finish={finish}{C['off']}")
    print(f"{C['dim']}     {'   '.join(bits)}{C['off']}\n")
    return {"text": text, "reasoning": reasoning, "calls": calls,
            "finish": finish, "timings": tm, "usage": u}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, help="exchange number from `llamatap list`")
    ap.add_argument("--model", action="append", default=[],
                    help="replay against this model instead; repeat to compare several")
    ap.add_argument("--to", default=UPSTREAM, help="upstream base URL")
    ap.add_argument("--diff", action="store_true",
                    help="also print the ORIGINAL reply, for comparison")
    ap.add_argument("--max-tokens", type=int, help="override max_tokens for the replay")
    ap.add_argument("--file", default=REC)
    a = ap.parse_args()

    recs = load(a.file)
    if not 1 <= a.n <= len(recs):
        sys.exit(f"no exchange #{a.n} — the recording has {len(recs)}")
    rec = recs[a.n - 1]
    req = rec.get("req")
    if not isinstance(req, dict):
        sys.exit(f"exchange #{a.n} has no usable request body")

    w = min(100, (os.get_terminal_size().columns if sys.stdout.isatty() else 100))
    orig_model = req.get("model")
    targets = a.model or [orig_model]

    msgs = req.get("messages") or req.get("input") or []
    print(f"{C['hdr']}{f' replaying #{a.n} ':=^{w}}{C['off']}")
    print(f"{C['dim']}  {len(msgs)} messages, {len(req.get('tools') or [])} tools, "
          f"captured {rec.get('t','')}  |  original model {orig_model}{C['off']}\n")

    if a.diff:
        summarise(f"ORIGINAL  ({orig_model})", rec.get("resp") or "",
                  (rec.get("meta") or {}).get("elapsed_ms") or 0, w)

    out = []
    for model in targets:
        body = dict(req)
        body["model"] = model
        # non-streamed so the reply carries llama.cpp's timings block
        body["stream"] = False
        if a.max_tokens:
            body["max_tokens"] = a.max_tokens
        resp, status, ms, err = send(body, rec.get("path") or "/v1/chat/completions", a.to)
        if err:
            print(f"{C['bad']}  ── {model}: request failed — {err}{C['off']}\n")
            continue
        if status != 200:
            print(f"{C['bad']}  ── {model}: HTTP {status}{C['off']}")
            print(f"{C['dim']}     {resp[:400]}{C['off']}\n")
            continue
        res = summarise(f"REPLAY    ({model})", resp, ms, w)
        out.append((model, res))
        meta = {"elapsed_ms": ms, "replay_of": a.n, "replay_model": model}
        try:
            d = json.loads(resp) if resp else {}
            meta["usage"], meta["timings"] = d.get("usage"), d.get("timings")
        except Exception:
            pass
        append_record(rec.get("path"), status, body, resp, meta, path=a.file)

    if len(out) > 1:
        print(f"{C['hdr']}{' comparison ':=^{w}}{C['off']}")
        print(f"  {'model':<26} {'decode t/s':>11} {'out tok':>8} {'tool calls':>11}  thought?")
        for model, r in out:
            tm = r["timings"] or {}
            print(f"  {model:<26} {tm.get('predicted_per_second', 0):>11.1f} "
                  f"{tm.get('predicted_n', 0):>8} {len(r['calls']):>11}  "
                  f"{'yes' if r['reasoning'].strip() else 'no'}")
        print()
    print(f"{C['dim']}  appended to the recording — `llamatap list` to see it{C['off']}")


if __name__ == "__main__":
    main()
