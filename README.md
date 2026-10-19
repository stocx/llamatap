# llamatap

Watch what an AI coding harness and a local model actually say to each other.

`llamatap` sits between any harness and your model server, records every
request and response, and renders them as **one group chat with four
participants** — you, the harness, llama-server, and the model.

```
harness  ->  127.0.0.1:9293 (llamatap)  ->  your model server  ->  llama-server
```

Point a harness at the tap by changing one string — its base URL — and nothing
else about its behaviour changes.

## Why

A llama-server log gives you token counts, timings, and at high verbosity the
*rendered* prompt: one string, produced after the Jinja chat template has run.
By then the system prompt and the tool definitions are indistinguishable bytes
inside that string.

llamatap sees what the **harness** sent, before the template touched it:

|  | server log | llamatap |
|---|---|---|
| token counts, prefill and decode rates | yes | yes |
| the `tools` array as JSON, each schema and its size | **no** | yes |
| which bytes were system prompt vs tool definitions | **no** | yes |
| the `messages` array and its roles | **no** | yes |
| background requests told apart from yours | hard | yes |

That is the difference between knowing a request cost 11,431 tokens and knowing
that 22,702 characters of it were tool schemas and 14,018 were a skill index.

## Requirements

Python 3 and nothing else — standard library only, no pip, no venv. A terminal
that can do colour and mouse reporting for the console. `wl-copy` or `xclip`
if you want `/copy` to reach the system clipboard; it degrades gracefully
without them.

## Install

```bash
git clone https://github.com/stocx/llamatap.git ~/GIT/llamatap
ln -s ~/GIT/llamatap/llamatap ~/.local/bin/llamatap
```

The dispatcher resolves its own location, following symlinks, so the clone can
live anywhere and the symlink is all that goes on `PATH`.

## Tell it where the model is

Three ways, highest precedence first:

```bash
llamatap --upstream http://127.0.0.1:8080 console    # just this command
export LLAMATAP_UPSTREAM=http://127.0.0.1:8080       # this shell
```

```ini
# ~/.config/llamatap/config          <- the normal case, set once
upstream = http://127.0.0.1:8080
port     = 9293
```

The built-in default is `http://127.0.0.1:8080`, llama-server's own port, which
assumes the model runs on the same machine as the tap.

Check what it resolved to, and whether it answers:

```bash
llamatap config
```

```
upstream      http://127.0.0.1:8080
              from: config file
listen port   9293
config file   /home/you/.config/llamatap/config
state dir     /home/you/.local/state/llamatap
reachable?
  yes — http://127.0.0.1:8080/v1/models answered 200
```

It works against anything OpenAI-compatible: llama-server directly, llama-swap,
Ollama, vLLM, LM Studio. Some extras — the live launch command, the context
window, the sampling settings — come from llama.cpp and llama-swap endpoints
(`/props`, `/running`) and are simply left out, labelled, when the server does
not offer them.

**Protocol coverage, stated plainly.** Requests are understood in all three
shapes: Chat Completions, Anthropic (`/v1/messages`) and Responses
(`/v1/responses`). **Responses are parsed for Chat Completions only**, SSE
included. A harness speaking Responses (codex) or Anthropic (Claude Code) is
recorded in full and replays correctly, but its model turn renders as nothing
in the reader until those formats are implemented. The shape is the harness's
choice, not the model's — a local llama-server answers all three.

## Use

For colleague onboarding, download the [Word handout](docs/user-guide.docx) or read the [Markdown User Guide](docs/user-guide.md). Both cover installation, llama-server setup, harness configuration, console keys, and slash commands.

Two windows. In the first:

```bash
llamatap serve
```

In the second:

```bash
llamatap console
```

Then point your harness at `http://127.0.0.1:9293/v1` and work normally.
`serve` prints a banner and then nothing, which is correct — it is a proxy.

| command | |
|---|---|
| `llamatap serve` | the proxy. Archives the previous recording and starts clean; `--keep` appends |
| `llamatap console` | live group chat with the dashboard. The one you want |
| `llamatap list` | numbered index of the recording |
| `llamatap read [N]` | render exchange N as text, pipeable |
| `llamatap all` | every exchange as text |
| `llamatap follow` | plain scrolling live view, for piping |
| `llamatap replay N` | re-send exchange N, optionally to another model |
| `llamatap log` | tail the one-line-per-request summary |
| `llamatap clear` | archive, then start fresh. Refuses rather than clearing if the archive fails |
| `llamatap archive` | archive without clearing |
| `llamatap config` | resolved settings, and whether the server answers |

All of them take `--file <path>` to open an archived recording.

## Replay turns a recording into an experiment

A captured request **is** the exact prompt a harness built. Re-sending it to a
different model changes one variable and leaves every other byte identical,
which driving a harness twice by hand cannot do.

```bash
llamatap replay 3 --model qwen3.5-9b --model qwen3.5-9b-nothink
```

## The design rule

What the protocol states is reported as fact. Everything else is derived from
content and labelled as such.

The system prompt is identified by `role: system` — that is protocol, true on
any harness. Its *sections* come from the prompt's own markdown headings and
tag opens, labelled with the prompt's own words, so an unfamiliar harness is
described in its own vocabulary rather than silently producing nothing. An
earlier version matched a hardcoded list of one harness's phrases, which would
have found nothing anywhere else and given no way to tell "no sections" from
"not recognised".

Nothing on the wire says "this message is a repeat", so repeats are string
comparison done in the reader, and repeated text is shown as itself rather than
annotated with a claim the protocol does not support.

## Privacy

Recordings contain your prompts, your file contents and your paths.

- The state directory is created `0700`; recordings, logs and the in-flight
  file are created `0600`, with the mode set at creation rather than by a
  later `chmod`.
- They are gitignored.
- `clear` and `serve` **archive, never delete**, into a directory that cannot
  collide with an existing one. If archiving fails, nothing is truncated.
- The recording holds the request the harness sent, byte for byte. If a
  compatibility shim rewrote the outgoing request, `meta.shims` says so.

If you share a recording, read it first.

## Reference

`docs/reference.md` — every key and command, what is observed versus inferred,
the numeric traps, and the findings this has produced so far.
