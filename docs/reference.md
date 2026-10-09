# llamatap — watching a harness and a model talk

`llamatap` sits between any harness and llama-swap, records every request and
response, and renders them as **one group chat with four participants**.

    harness  ->  127.0.0.1:9293 (llamatap)  ->  model-host:9292 (llama-swap)  ->  llama-server

Point a harness at the tap by changing one string — its base URL — and nothing
else about its behaviour changes.

| piece | file |
|---|---|
| dispatcher | `~/GIT/llamatap/llamatap` (symlinked to `~/.local/bin/llamatap`) |
| proxy | `~/GIT/llamatap/llamatap-proxy.py` |
| reader / thread model | `~/GIT/llamatap/llamatap-thread.py` |
| live console (curses) | `~/GIT/llamatap/llamatap-console.py` |
| replay | `~/GIT/llamatap/llamatap-replay.py` |
| what it recognises | `~/GIT/llamatap/llamatap-catalog.py` |
| summary log | `~/.local/state/llamatap/llamatap.log` |
| recording | `~/.local/state/llamatap/llamatap-records.jsonl` |
| archived sessions | `~/.local/state/llamatap/runs/llamatap-<MMDD-HHMM>/` |
| cached launch flags | `~/.cache/llamatap/launch-flags.json` |

---

## The commands

| command | what it does |
|---|---|
| `llamatap serve` | run the proxy on :9293. **Archives the previous recording and starts clean**; `--keep` appends instead |
| `llamatap console` | the live group chat with the dashboard. This is the one you want |
| `llamatap list` | numbered index of the recording |
| `llamatap read [N]` | render exchange N as text (pipeable); default is the most recent |
| `llamatap all` | every exchange as text |
| `llamatap follow` | plain scrolling live view, for piping. Has its own `r`/`i`/`p`/`f` keys and `--raw` / `--no-icons` |
| `llamatap replay N` | re-send exchange N, optionally to another model |
| `llamatap log` | tail the one-line-per-request summary |
| `llamatap clear` | archive, then start fresh |
| `llamatap archive` | copy the recording into runs/, leaving the live one alone |
| `llamatap config` | resolved settings, and whether the server answers |

Every one accepts `--file <path>` to open an archived session:

```bash
llamatap console --file ~/.local/state/llamatap/runs/llamatap-1002-1326/llamatap-records.jsonl
```

### Two windows

```bash
llamatap serve
```

```bash
llamatap console
```

`serve` prints a banner and then nothing — that is correct, it is a proxy. The
console is where you watch. Ctrl-c stops the proxy and reports what it caught.

---

## Why a proxy and not a server log

A llama-server log can give token counts, timings, and at high verbosity the
**rendered** prompt: the single string produced *after* the Jinja chat template
has run. What it cannot give is the structure that existed before.

| you want to see | server log | llamatap |
|---|---|---|
| token counts, prefill and decode rates | yes | yes |
| the `tools` array as JSON, each schema | **no** | yes |
| which bytes were the system prompt vs tool definitions | **no** | yes |
| the `messages` array and its roles | **no** | yes |
| background requests told apart from yours | hard | yes |

Once the template has run, tool definitions and the system prompt are
indistinguishable bytes in one long string. The point of measuring harnesses is
seeing what the **harness** sent.

**The exception:** for preamble size alone, llama-swap's own log is enough.
Reach for llamatap when you need to read the content.

---

## The four participants

    you              what you typed
    the harness      writes the system prompt, defines the tools, runs
                     them, and speaks for itself in background requests
    llama-server     renders the chat template, turns the tool list into
                     prose inside the prompt, parses tool calls back out
    the model        thinks, replies, asks for tools

`llama-server` is the participant people forget. It is why 11 tool definitions
cost thousands of tokens: they arrive as JSON and leave as English.

Each party has **one icon and one colour**, used everywhere. Speech appears in
a bubble tinted to the speaker; the model's thinking gets a dotted bubble with a
thought-bubble tail, because thinking is not speech.

---

## The console

One chronological stream, the participants pinned in a header row, no
per-exchange headers. Every message appears where it happened; a `+` opens
structure in place.

```
  Filter:  You ▼   opencode ▼   Llama b11040 ▼   qwen3.5-35b-moe ▽

  opencode → prompting                    45,724 chars · ~11,431 tokens · 11.4s prefill
  ▌ + system prompt   23,022 chars · 50% of the request
  ▌      + Tone and style        2,290 chars ·  9% of the prompt
  ▌      + <available_skills>   14,018 chars · 60% of the prompt
  ▌ + 11 tools        22,702 chars · 49% of the request

  you                                                       5 chars · ~1 tokens
  ▌ hello
  ▌ 12:56pm

  qwen3.5-35b-moe → thinking                            ~26 tokens · ~0.5s
  ▌ + Thought

  qwen3.5-35b-moe                              34 tokens · 0.6s · 57/sec
  ▌ Hello. What would you like to do?
  ▌ 12:56pm
```

Each block is a filled panel behind an accent bar in the speaker's colour.
The speaker's name sits above the bar with that message's cost beside it, so
the block itself holds nothing but the text — including for a thought, which
is labelled `→ thinking` the same way a tool result is labelled
`→ tool result`.

Times are 12-hour. The recordings keep ISO timestamps, which sort and never
need a timezone; only the display is converted.

### The header row filters the conversation

The four participants are a **pinned row**, not the first line of the
transcript — they used to scroll away, and they are now the filter control.
It is labelled `Filter:` and each name carries a funnel: `▼` showing,
`▽` hidden. The mark matters because dimness alone is lost on a low-contrast
theme, and it sits inside the click target, since it is the affordance.

**Click a name to show or hide that party.** All four are on by default; a
dimmed name is a party whose messages are being withheld, which stays visible
so a filtered view can never be mistaken for a quiet session. Any combination
works, including none.

A tool result filters with the harness, not with you: it comes back from the
harness process, which is also why it is labelled `<harness> → tool result`
rather than attributed to a person.

llama-server speaks when a request fails. Any non-2xx response becomes a
`llama-server → error` block carrying the status and the response body, which
is both what makes its name a working filter and the only way a 500, a 503 or
a refused connection is visible at all — those used to leave a silent gap in
the thread.

**The filter changes what you read, never what is counted.** It is applied
after the statistics are derived, so hiding the model does not alter a single
number in the dashboard.

Streaming is live: a message appears as it arrives, with a spinner and the
running token count, before `usage` and `timings` land in the final chunks.
**The view follows the stream** down to the newest row while it is being
written, and goes back to showing each finished message whole once it lands.
Scrolling up stops the following, as it does anywhere else.

**What was asked is on screen while it is being answered.** A record is only
written when an exchange completes, so the request half used to appear after
the reply it caused — most visibly on a title generator, which arrives with
its own system prompt and a question nothing on screen had asked. The proxy
publishes the parsed request once per request, beside the small entry it
rewrites per token, so the console can draw the prompt immediately without
re-serialising a 23 KB system prompt hundreds of times a reply.

Only the turns that are **new** are drawn, and they are attributed by the same
structural test the completed thread uses: a side-call carries its own system
prompt, compared here against the last completed request. Both blanket rules
are wrong, and both were tried — attributing everything to the user puts a
title generator's prompt in your mouth, and attributing everything to the
harness puts your own "hello" in opencode's. The expandable nodes are inert
for those few seconds and work normally the moment the record arrives.

### The dashboard

**This turn and Session carry the same measurements**, under the same names,
so the two can be read against each other: KV cache hits, Prefill, Decode. A
metric that appears in one and not the other is one that genuinely does not
exist at that scale — Context and Requests belong to the thread, Tools Offered
to the single request that put them on the table.

**Thinking is shown as a share of the context**, because a window that is 40%
reasoning the model has already finished with is a different thing from one
that is 40% code, and nothing else on screen separates them. The thinking
*level* is shown when a harness asks for one; none seen so far does.

"KV cache hits" is llama.cpp's own name for the thing. Both the rail and the
bottom panel use the same section names and the same labels, so the layout is
learned once.

Three panels, organised by **time scale**, so a reader can ask "is this a fact
about the model, about this turn, or about the whole session?" and the column
answers it.

```
**Two placements, because they are good at different things.** `s` cycles
right → bottom → off.

**Right** is slim: the few numbers you watch while a turn runs. It costs
screen columns, of which a wide terminal has plenty, and it is hidden
automatically below 80 columns. It carries Context, Cache Reuse, Prefill,
Decode, Requests, Tokens Generated and Decode Average.

**Bottom** is everything, in three panels side by side — Model Details /
Current Request / Session Totals — which is the grouping that makes the
numbers scannable. It is the only placement wide enough for columns, and the
only one that shows the settings board (`/flags`) and the launch command. It
costs conversation rows, and is capped at 40% of the screen height; if it
clips, it says how many rows it hid.

The right-hand rail:

Session
Context
  15,427 of 65,536 tokens
Requests         25
Messages         48
Tokens Generated 3,666
Decode Average   66.3/sec
Prefill Average  792/sec
Model Time       74 seconds

This turn
Tools Offered   11
Cache Reuse     ██████████· 100%
  15,407 reused, 20 new
Prefill
  20 tokens at 135/sec
Decode
  40 tokens at 83.7/sec

Model
Model
  qwen3.5-35b-moe
Quantised       IQ4_XS - 4.25 bpw
Runtime
  llama.cpp b11040
Slots           4
```

| panel | scope | changes |
|---|---|---|
| **Model Details** | identity and configuration | never, during a session |
| **Current Request** | the most recent real turn | every turn |
| **Session Totals** | everything recorded so far | accumulates |

**Context lives in Session Totals, not Current Request.** The API is stateless,
so every request re-sends the whole history and prompt tokens only ever grow.
`15,427 of 65,536` is not what this turn cost — it is how full the conversation
now is, and when it fills the harness has to compact or fail.

**Current Request deliberately skips background calls.** A title generator's
numbers are never what you opened the panel to learn; a footnote says when the
newest request was one.

The dashboard is **capped at 40% of the screen** and names any rows it clips.
It is reference; the conversation is the point of the tool.

### The settings board

`/flags` or `f` adds one row per llama-server setting inside Model Details —
set values coloured, defaults plain, because a default is information too
(`Jinja Template off` explains a harness whose tool calls never parse).

It is **14 rows and off by default.** The `$ …` launch command underneath is one
row and always shown; the two no longer share a toggle.

### Keys

    j k  or  ↑ ↓     scroll one line
    ctrl-d / ctrl-u  scroll half a screen
    ctrl-f / ctrl-b  scroll a screen (also Space, PgDn, PgUp)
    gg               jump to the start of the conversation
    G                jump to the newest message and follow new ones
    click a node row expand or collapse it — anywhere on the line, not just
                     the + itself
    click a name     in the header row: show or hide that party
    x                number the expandable nodes, then press 1-9 to toggle
    o                expand or collapse everything in this exchange
    r                tool-call arguments: raw JSON off the wire vs readable
    p                open or close the system prompt
    f                show or hide the launch settings in Model Details
    t                fold the model's thinking to one line, or unfold it
    s                statistics: right (slim), bottom (full), off
    y                copy the whole scroll buffer to the clipboard
    /                command palette, filters as you type
    Esc              cancel — it does NOT quit
    q                quit

**Escape does not quit.** Arrow keys and mouse reports are escape sequences, so
a dropped byte looks like a bare Esc — quitting on that was wrong. Esc cancels.

**The motions are vim's, not the terminal's.** `Home` and `End` still work, but
on a laptop they need `Fn`, which is the wrong thing to press while reading. A
lone `g` arms the `gg` prefix and says so; any other key cancels it.

### The command palette

`/` opens a centred list that filters as you type — prefix matches first, then
substring matches on name or description, so `arch` finds `/clear`. ↑↓ chooses,
Tab completes, Enter runs, Esc cancels.

| command | what it does |
|---|---|
| `/copy [raw]` | copy the scroll buffer to the clipboard |
| `/save [path]` | write the buffer to a file |
| `/pager` | page the buffer in `less`, where selection works |
| `/clear` | archive this recording, start a fresh one |
| `/ask <message>` | ask the model directly, in the harness's captured context |
| `/replay N [--model M]` | re-send exchange N, optionally to another model |
| `/read N` | jump to exchange N |
| `/list` | print the index into the conversation |
| `/raw` | tool-call arguments: raw JSON off the wire vs readable |
| `/preamble` | open or close the system prompt node |
| `/flags` | show or hide the launch settings in Model Details |
| `/repeats [open\|close]` | show re-sent messages as collapsed lines; open or close them all |
| `/thinking [open\|close]` | fold the model's thinking to one line; open or close them all |
| `/stats [right\|bottom\|off]` | statistics: slim on the right, full across the bottom, or hidden |
| `/name <label>` | override the detected harness name |
| `/help` | list these commands and the keys |
| `/quit` | exit |

`/pager` drops out of the TUI into `less`, where your terminal's own selection
works — a curses app owns the screen, so dragging the mouse selects the painted
rectangle, dashboard included, not the logical buffer.

`/clear` **archives, never deletes.** It moves the recording into
`runs/llamatap-<stamp>/` and prints the command to read it back. `serve` does
the same on every start, so you never have to remember to.

### Edits are shown as diffs

An `edit` call carries both halves on the wire, so what changed is observable
and does not have to be read out of two blobs by eye. They render as one diff
inside the same collapsible node every other long argument gets:

```
  qwen3.5-35b-moe → tool call [edit]                334 tokens · 5.9s · 57 t/s
  ▌ filePath: utils/phone_validator.py
  ▌ − diff   +3 −0 lines
  ▌      1   def validate_phone(phone):
  ▌      2       """Validate a phone number format."""
  ▌      3 +     digits = re.sub(r"[^\d]", "", phone)
  ▌      4 +     if len(digits) < 7:
  ▌      5 +         return False
  ▌      6       pattern = r"^[\+]?[(]?[0-9]{1,4}..."
```

**The line numbers are the file's, not the fragment's.** An edit's payload is
an excerpt and its replacement — never a line number, and never the file. The
harness knows where the excerpt sits because it opens the file itself, and
that is local I/O between the harness and your disk: it never crosses the tap,
so llamatap cannot see it.

What does cross the tap is every file the harness READS or WRITES, because
those go to the model as tool traffic. So the thread is reconstructed as it
plays — a write sets a file's contents, a numbered read listing reveals them,
an edit applies itself — and the excerpt is then located in what the file is
known to contain.

Reconstructed rather than read off disk on purpose: a recording replayed next
month must show what was true when it was made, not what the file says now.
That is also why it is a better number than the harness's own, which is only
true at the moment it was drawn.

**A file the thread never saw gets no numbers at all.** Numbering the excerpt
from 1 would look absolute and be wrong, which is worse than a blank gutter.

**The colour is the background, not the text.** Tinted text is easy to miss on
one short line among many, and illegible once a line wraps and only part of it
carries the colour. **Every row is numbered**, because "which line" is most of
what makes a diff readable and a `@@ -31,5 +31,5 @@` header makes you count
rows to work it out. A removed row carries its number in the old text, an
added row its number in the new, so the two disagree wherever lines were
inserted. A long unchanged run is elided and says how many rows it swallowed.

The two halves are found by SHAPE, not by one harness's spelling: any argument
whose name starts with "old" pairs with the one whose name is the same with
"new" in its place. That covers `oldString`/`newString`, `old_string`/
`new_string`, `old_str`/`new_str` and plain `old`/`new` without a table of
names to maintain — the same reason `sections()` reads headings rather than a
list of phrases.

**Nesting varies more than spelling does**, so the search goes to any depth:

    opencode   {"filePath": …, "oldString": …,  "newString": …}
    pi         {"path": …, "edits": [{"oldText": …, "newText": …}]}

A call carrying several edits renders several diffs.

The diff is derived rather than observed, which is safe here only because
`/raw` still shows the strings exactly as they crossed the wire. A wrong diff
costs legibility, never accuracy.

### Repeats

Every request re-sends the whole conversation, so most of what arrives has been
seen before. Four states:

| | |
|---|---|
| `/repeats` | hidden ⇄ shown. **Hidden is the default** |
| `/repeats open` | shown, with every one expanded |
| `/repeats close` | collapse them all again |
| click a `↻` row | expand just that one |

A shown repeat is a collapsed one-liner that still carries its tokens and
seconds, so you can tell whether it mattered.

Scale, on a 25-exchange assembly task: 63 messages hidden, 654 shown, 591 of
them repeats, forming 24 consecutive runs of 25, 28, 29, 31 … 48 — one step per
turn.

**A repeat is not the harness misbehaving.** The API is stateless, so the whole
history has to go out every time or the model has no context. What the staircase
shows is how much *context window* each turn consumes, not wasted compute —
with the prefix cache working, re-sent tokens are nearly free to process.

### Thinking

A thinking block is routinely the largest part of a turn — 1,370 tokens and
20 seconds in one measured case — so reading the conversation around one means
scrolling past all of it.

**Every thought is a `+`/`−` node from the start**, like anything else that
opens. Click one to collapse it:

    qwen3.5-35b-moe → thinking            ~1,370 tokens · ~22.4s
    ▌ − Thought                                                  expanded
    ▌ + Thought                                                  collapsed

The cost is on the name row, outside the block — so folding still never
hides what the thought spent.

`/thinking` sets the default for all of them at once:

| | |
|---|---|
| click a `Thought` row | collapse or expand just that one, any time |
| `/thinking` or `t` | flip the default: folded ⇄ full. **Full is the default** |
| `/thinking close` | fold every one |
| `/thinking open` | expand every one |

A command sets the default and clears per-node choices, so "fold every one"
means every one — including any you had opened by hand. `--fold-thinking`
starts the console folded.

**Folding hides the text, never the cost.** A collapsed row always carries its
tokens and its seconds, and when the duration cannot be derived it says
`duration unknown` rather than showing tokens alone — a row with no time on it
reads as a thought that was free.

---

## Where the launch flags come from

`props` does **not** report launch flags — no `-ngl`, `-c`, `-ctk`, `-fa`, no
draft model — and those are exactly the settings that decide speed and whether
a long context fits. Three sources, best first:

| tier | source | when | shown as |
|---|---|---|---|
| 1 | llama-swap `/running` | model is loaded | `$ -ngl 999 …` |
| 2 | `~/.cache/llamatap/launch-flags.json` | model idle or unloaded | `$ … (cached 20 min ago)` |
| 3 | config YAML over ssh | never loaded this llama-swap lifetime | `$ … (from the config file, not the running process)` |
| — | nothing | — | `$ no launch command — model is not loaded and no cached copy` |

**Tier 1 is more truthful than the config file, not just faster.** `/running`
reports the command line of the process that is *actually running*; the YAML
describes the launch that was *configured*. If llama-swap started before the
file was last edited, the file describes a launch that never happened.

It also needs no ssh — just the upstream llamatap already forwards to. Resolves
in about 90 ms with no ssh agent loaded at all.

```bash
curl -s http://model-host:9292/running | python3 -m json.tool
```

Also on llama-swap, worth knowing: `/metrics` is Prometheus-format CPU, memory,
load and network for the server box.

---

## What is observed and what is inferred

**A harness's side-calls are found structurally, not by their wording.** A
title generator or summariser used to be recognised by matching eleven English
phrases. On any harness that phrased its own differently, the side-call fell
through to `you` — the console claimed the user had typed the harness's
internal prompt, which is the worst thing this tool can get wrong.

Two observable facts decide it now, neither of them language:

1. a main-thread request **extends** the conversation — every message said so
   far reappears in it; a side-call carries at most a quote
2. a side-call carries **its own system prompt**

Both are required, which is what keeps the rule from over-firing. A harness
that legitimately swaps its system prompt mid-thread still carries the
history, so it stays in the main thread. One that re-sends no history at all
keeps the same system prompt, so it stays too.

The phrase table survives only to **name** what a side-call is — "title
generator", "summariser". An unrecognised one is still correctly a side-call;
it is just labelled "side call" rather than named.


This matters more than any feature here, because the whole project is about
measuring harnesses and a measurement that quietly guesses is worse than none.

**Read off the wire — true for any harness:**

- the system prompt, because the protocol names it: `messages[0].role ==
  "system"`, or the top-level `system` field, or `instructions`
- the `tools` array, each schema and its size
- every message, its role, and the order they arrived
- token counts, prefill and decode rates, cache hits — llama.cpp reports these
  itself in a `timings` block
- that a request carried its **own** system prompt and offered **no tools**,
  which is how a background call is identified structurally

**Derived from the content, not from a flag:**

- **Sections of the system prompt** come from the prompt's *own* markdown
  headings and XML-ish tag opens, and are labelled with the prompt's own words.
  An earlier version matched a hardcoded list of phrases — `AGENTS.md`,
  `<available_skills>` — which was opencode's vocabulary. On any other harness
  it would have found nothing, and you could not tell "this prompt has no
  sections" from "llamatap does not recognise this prompt". If a prompt has no
  headings it gets no sections, and that is the truth about it.
- **Repeats.** Nothing on the wire says "this is a repeat". Every request
  re-sends the whole history and the proxy sees new text each time; matching
  them is string comparison done in the reader. So repeated text is shown as
  itself, in place, with no commentary.
- **A background request's name** (`title generator`, `summariser`) is matched
  against known prompt wording and is decorative. The structural facts stand
  without it.
- **Icons** are cosmetic throughout. No label ever depends on one.

---

## The numbers

| line | source |
|---|---|
| `Prefill N tokens at X/sec` | llama.cpp's `timings.prompt_*`. `prompt_n` counts tokens actually **processed**, so cache hits do not appear |
| `Decode N tokens at X/sec` | `timings.predicted_*`, measured inside the server |
| `Context N of M` | `usage.prompt_tokens` against `n_ctx` from `/upstream/<model>/props` |
| `Cache Reuse N reused` | `prompt_tokens_details.cached_tokens`, or `timings.cache_n` |
| `Decode Average` | token-weighted across every request, so a 3-token reply does not count as much as a 3,000-token one |
| a turn's `~N.Ns` | **apportioned**, not measured: decode is one continuous run at a near-constant rate, so a turn's share of the tokens gives its share of the time. The `~` says so |

**Never compute tokens per second from wall clock.** A non-streamed response
arrives in one burst, so time-to-first-byte equals total time and the derived
rate is fiction — an early version reported `120000 tokens/s`. Use the server's
`timings` block.

**`Prefill 23 tokens` next to `Context 636` is not a contradiction.** The gap is
the prefix cache working.

**Cache Reuse can never be 0 after the first turn.** History is re-sent
additively, so a second request must hit the cache. A zero there means the
dashboard is reading the wrong record, not that the cache missed.

**Wall clock across models is not a speed comparison.** llama-swap may have had
to load the model. Compare decode rates.

---

## Replay — the recording as an experiment

A captured request **is** the exact prompt a harness built. Re-sending it to
another model changes one variable and leaves every other byte identical, which
driving a harness twice by hand cannot do.

```bash
llamatap list
llamatap replay 3 --model qwen3.5-9b-nothink --diff
llamatap replay 3 --model qwen3.5-9b --model qwen3.5-9b-nothink
```

Replays are appended to the recording, so they appear in `list` and open like
anything else. Streaming is forced off for a replay, because a non-streamed
response carries the `timings` block the comparison needs.

**First result, 2026-10-02:** same request, thinking on versus off —

    qwen3.5-9b            120.1/sec   54 tokens out   1 tool call   thought: yes
    qwen3.5-9b-nothink    119.3/sec   28 tokens out   1 tool call   thought: no

Same tool call, half the output tokens. n=1 on a trivial prompt, so it settles
nothing — but it took one command instead of two hand-driven sessions.

Note a replay of a prompt still in the prefix cache reports a huge cache hit and
is **not** a cold-start measurement.

---

## Pointing a harness at it

Change the base URL to `http://127.0.0.1:9293/v1`.

| harness | where |
|---|---|
| opencode | `~/.config/opencode/opencode.json`, the provider `baseURL` |
| Pi | `~/.pi/agent/models.json`, the provider `baseUrl` |
| Cline | the OpenAI-compatible **Base URL** field |
| litellm | `OPENAI_API_BASE` |
| Claude Code | `ANTHROPIC_BASE_URL` |
| codex | its config's base URL |

**What is supported, honestly.** Requests are understood in all three shapes —
Chat Completions, Anthropic and Responses. **Responses** are currently parsed
for Chat Completions only, including its SSE framing. That means a harness
speaking the Responses API (codex) or the Anthropic messages API (Claude Code)
is recorded in full and replays correctly, but its model turn renders as
nothing in the reader until those response formats are implemented. This is not
a remote-model issue: llama-server answers all three shapes locally, so the
harness picks the shape.

**The two body rewrites.** `_hoist_system` and `_merge_developer` exist because
Qwen3.5's chat template raises "System message must be at the beginning." on a
non-leading system message. Each logs a `[shim]` line, and the exchange's
`meta.shims` names it.

They are **lossless**, which they once were not:

- On `/v1/messages`, a non-leading system message is hoisted into the
  top-level `system` field — that is Anthropic's protocol.
- On Chat Completions it is **merged into the leading system message**. It used
  to be relocated to a top-level `system` field here too, which an
  OpenAI-compatible server simply ignores — so the instruction was silently
  deleted in transit. A tap that drops "always answer in French" is worse than
  one that passes the template error through.
- A string-valued `developer` item in a Responses request is merged rather
  than dropped.

**The recording always holds the original request**, not the rewritten one.
Those were the same variable once, so llamatap reported its own edit as the
harness's request — in a tool whose whole claim is seeing what the harness
sent.

**If a harness is pointed at the tap, the tap must be running** or it fails with
a connection error. Revert with the backup beside the config, e.g.
`~/.config/opencode/opencode.json.model-host`.

---

## Environment

| var | default | why |
|---|---|---|
| `LLAMATAP_UPSTREAM` | `http://model-host:9292` | where to forward. Set to `127.0.0.1:9292` to tap the laptop's own llama-swap |
| `LLAMATAP_PORT` | `9293` | one above llama-swap so both run at once |
| `LLAMATAP_REC` / `LLAMATAP_LOG` | in `~/Work/bench/` | recording and summary log |
| `LLAMATAP_CAP` | unset = **no cap** | bytes of response to keep. See the traps |
| `LLAMATAP_KEEP` | unset | append to the existing recording instead of archiving it |
| `LLAMATAP_SPINNER` | ASCII | braille spinners render as mush in some fonts |
| `LLAMATAP_SSH_HOST` | `model-host` | tier-3 launch-flag source only |
| `LLAMATAP_SWAP_CONFIG` | `/home/<user>/llama-swap.yaml` | tier-3 launch-flag source only |

---

## Four traps paid for

**1. Two separate 256 KB caps, both silent.** One truncated the *stream*, which
is where `usage` and `timings` live, so a long response lost its own numbers.
The second was in `record()` and chopped long tool-call arguments mid-string —
the JSON then failed to parse, and a 3,000-line file being written rendered as
raw escaped text while "readable" was on. Both removed; `LLAMATAP_CAP` is opt-in
and defaults to no cap.

The general lesson: **a silent truncation reads as real data.** `15,427 of 65`
looks like a number, not a slice. The console's `fit()` now ends any clipped
value with `…` for exactly this reason.

**2. Port already in use** means an old `llamatap` is still running. The error
prints the pids. Finding them needs care: `pkill -f` matches *your own shell's*
command line and has killed it twice. Safe form:

```bash
for P in $(pgrep -x python3); do
  tr '\0' ' ' < /proc/$P/cmdline | grep -q llamatap-pro && kill "$P"
done
```

A stale proxy also makes changes look like they did nothing — if llamatap seems
to ignore an edit, check for an old process holding 9293.

**3. Emoji are two cells wide but one character.** Using `len()` for column
arithmetic puts every span after an emoji one cell early and the terminal
overwrites it — `🎓skill` rendered as `🎓sill`, and later the same bug broke the
dashboard border. Measure in cells (`dwidth`), not characters. Some emoji
(U+1F5DC `🗜`) have no emoji-presentation default and render *one* cell, so no
width rule is correct for every terminal: boxed panels avoid them entirely.

**4. Chrome with no height cap eats the content.** The dashboard was verified at
three widths and never once measured for height; the settings board drew
unconditionally and took 23 of 30 rows. Check both axes, and cap anything that
competes with the content.

### Smaller ones

- **Bright-black (`\033[90m`) is unreadable on this theme.** Use it for
  separator rules only, never for text.
- **`░` reads as a solid block in some fonts**, so an empty meter looked full.
  Empty is `·`.
- **`finish_reason: tool_calls` is normal.** Only `length` and `content_filter`
  mean trouble.
- **A thinking model can return empty `content`** with everything in
  `reasoning_content`. A harness reading only `content` sees nothing.
- **Two code paths that must agree will not.** `read` and the console both
  render the conversation; the reader was still calling the deleted hardcoded
  marker list long after the console had moved to structural sections. The fix
  is to delete the old path, not to update both.

---

## What it has already found

**Preamble cost**

- opencode sends **45,724 characters (~11,431 tokens) to answer "hello"**:
  23,022 of system prompt and 22,702 of tool schemas. Roughly half the bill is
  tool definitions, not prose.
- **60% of that system prompt is `<available_skills>`** — 14,018 characters of
  skill index, on every request.
- `bash`'s schema alone costs **1,327 tokens**, more than the entire
  title-generator system prompt.

**Background requests**

- opencode fires its **title generator concurrently** with your request, so both
  prefill against the same GPU and slow each other down.
- It **never completes**: ~1,370 tokens and 20–25 seconds of reasoning, no
  `finish_reason`, stream cut. One run went 58 seconds and 3,808 tokens and
  still produced no title.
- It **can never hit the prefix cache**, because it carries its own system
  prompt and reuse requires matching from token 0.

**The cost of being stateless** — and what is actually the harness's fault

- One assembly task: **25 requests carrying 591 re-sent messages.** One cold
  prefill, then six near-free turns (`15,407 reused / 20 new`).
- **Resending is mandatory, not a harness choice.** No session exists on the
  server, so the conversation, the system prompt and every tool schema ride
  along on every request.
- **What the harness chooses is the size** of that mandatory payload, and
  **whether the prefix stays stable.** Caching matches only from token 0, so a
  harness that reorders tools, injects a timestamp or varies its system prompt
  between turns drops reuse to zero and pays full prefill every turn. That is a
  real defect and it is invisible without a tap.
- Therefore `Cache Reuse 100%` is a **pass**, not a cost. The bill that does
  come due is **context window**: re-sent tokens count toward `n_ctx` whether
  cached or not, so preamble size sets how fast you hit the wall.

**Harness behaviour**

- Adding the words **"and test"** turned the same request from a chat answer
  into a four-tool agent loop.
- The asteroids run is a **check-12 case**: pytest, pip, pip3, apt-get and
  pygame were all missing, so the model wrote a third file re-implementing
  `Vector2` inline and reported "8 tests passing" — for a copy of the class,
  not the one it shipped.

---

## If this ever grows up: mitmproxy as a host

mitmproxy is MIT-licensed Python with an addon API. The domain layer here —
party attribution, tool categories, prompt sectioning, the stats panel, thread
reconstruction — could run as an addon and inherit intercept-and-edit (pause a
request, trim the preamble, release it), server-side replay, TLS and a filter
language. What would not be inherited is thread reconstruction: mitmproxy shows
isolated flows, and every LLM request resends the whole conversation.

Parked for its own conversation.

## Adjacent, not part of this: llama-bench

`llama-bench` and `llama-batched-bench` ship with llama.cpp and are installed on
model-host at `/home/<user>/llama.cpp/llama-b11040/`. They sweep *server* settings
and report prompt-processing and generation throughput. They send no chat
template, no tool schemas and no conversation, so they say nothing about a
harness — but they give the hardware ceiling that makes llamatap's
`Decode Average 66.3/sec` interpretable, and they price the settings on the
board (`-ctk q8_0` versus `f16`, `--n-cpu-moe 15` versus other splits).

`llama-batched-bench` is the closer fit, since it varies prompt length and
parallel slots, which is what an agent loop actually does.

Needs the GPU to itself: `model-host stop` first, and a window when nobody is
gaming on that machine. Its own project, not this one.
