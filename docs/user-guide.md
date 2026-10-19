# Llamatap User Guide

This guide is for someone installing Llamatap to inspect conversations between an AI harness and a local model. It covers a local `llama-server` setup, connecting a harness through Llamatap, and reading the console.

## What Llamatap does

Llamatap is a local recording proxy. The harness sends API requests to Llamatap; Llamatap forwards them to the model server and records the request and response for inspection.

```text
AI harness  →  Llamatap (127.0.0.1:9293)  →  llama-server (127.0.0.1:8080)
```

The harness still does its normal work. Llamatap observes and records the traffic; it does not execute tool calls for the harness. The recording can contain system prompts, source code, file paths, and other sensitive content, so treat it like private project data.

## Requirements

- A Bash-compatible shell and Python 3.
- A terminal with colour and mouse support for the interactive console.
- A running OpenAI-compatible model server. This guide uses `llama-server`.
- `wl-copy`, `xclip`, or `xsel` is optional for clipboard copying. Without one, use `/save` instead.

Llamatap uses Python's standard library; there is no package installation step.

## Install Llamatap

Clone the repository and put its launcher on your `PATH`:

```bash
git clone https://github.com/stocx/llamatap.git ~/GIT/llamatap
mkdir -p ~/.local/bin
ln -s ~/GIT/llamatap/llamatap ~/.local/bin/llamatap
```

If `~/.local/bin` is not already on your `PATH`, add it in your shell's startup file. For Bash, add this line to `~/.bashrc`, then open a new terminal or run `source ~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Check the installation:

```bash
llamatap --help
python3 --version
```

The launcher follows the symlink to find the rest of the checkout, so the repository can live somewhere other than `~/GIT/llamatap`; adjust the clone path and symlink command together if you choose another location.

## Start llama-server

Install or build `llama-server` and obtain a model file using the [official llama.cpp server guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). With a local GGUF model, a typical command is:

```bash
llama-server -m /path/to/model.gguf -c 8192 --host 127.0.0.1 --port 8080
```

Replace `/path/to/model.gguf` with the real model path. `-c` selects the context length; choose a value supported by your model and machine. The server guide documents the model, context, host, and port options; server builds may expose additional options through `llama-server --help`.

Keep this process running in its own terminal. Llamatap's built-in upstream is `http://127.0.0.1:8080`, which matches this example. Binding to `127.0.0.1` keeps the server on the local machine; only bind to another interface when you intentionally need network access and have secured the server appropriately.

## Configure Llamatap

The default setup needs no configuration file when `llama-server` uses port 8080. Check the resolved settings and whether the server responds:

```bash
llamatap config
```

To use a different upstream address or tap port, create `~/.config/llamatap/config`:

```ini
upstream = http://127.0.0.1:8080
port = 9293
```

Create its parent directory first if needed:

```bash
mkdir -p ~/.config/llamatap
```

You can set the upstream just for one command:

```bash
llamatap --upstream http://127.0.0.1:8080 config
```

Or for commands launched from the current shell:

```bash
export LLAMATAP_UPSTREAM=http://127.0.0.1:8080
```

The precedence is built-in default, config file, environment variable, then `--upstream`. The tap listens on `127.0.0.1:9293` by default. To change its port for a shell session, set `LLAMATAP_PORT`, for example `export LLAMATAP_PORT=9294`.

## Connect a harness

Start Llamatap in one terminal:

```bash
llamatap serve
```

It prints the listen and upstream addresses, then waits for requests. That quiet terminal is expected. Stop the proxy with `Ctrl-C` when finished.

In your AI harness's model or provider settings, change its **base URL** to:

```text
http://127.0.0.1:9293/v1
```

If you changed `LLAMATAP_PORT`, use that port instead. Keep the harness's model name and other settings as they were unless your model server requires a change. The harness must be able to reach the machine and port where Llamatap is running.

Llamatap also supports an upstream that is itself an OpenAI-compatible service, such as llama-swap. In that case, configure `upstream` to that service's base URL, and keep the harness pointed at Llamatap.

### If the model server is on another machine

This guide's commands run Llamatap on Linux. The model server can be on the same machine or another machine; Llamatap needs an HTTP address it can reach. For a server on another host, set `upstream` to that host's network name or address and port, for example `http://model-host:8080`. The model server must listen on an interface reachable from the Llamatap machine, and the network must allow that connection. In this arrangement, the harness still points to the Llamatap machine. Do not use `127.0.0.1` for a remote model server: it always means the machine where Llamatap is running.

### View the live conversation

Open a second terminal and run:

```bash
llamatap console
```

The console starts with an empty conversation if no requests have passed through yet. Send a prompt through the harness; the captured exchange will appear. The header row names the participants—You, Harness, Server, and Model—and also filters the conversation. The dashboard reports request and session measurements when the server provides them.

## Console keyboard controls

| Key | Action |
|---|---|
| `↑` / `k` | Scroll up one line |
| `↓` / `j` | Scroll down one line |
| `Ctrl-U` / `Ctrl-D` | Scroll up/down half a screen |
| `Ctrl-B` / `Page Up` | Scroll up one screen |
| `Ctrl-F` / `Space` / `Page Down` | Scroll down one screen |
| `Home` / `gg` | Jump to the beginning |
| `End` / `G` | Jump to the newest content and follow new arrivals |
| Mouse wheel | Scroll the conversation |
| Click a participant in the header | Show or hide that participant's messages |
| Click a marked conversation row | Expand or collapse that item |
| `o` | Expand or collapse the current exchange |
| `r` | Toggle readable tool arguments and raw JSON |
| `p` | Open or close the system prompt |
| `f` | Show or hide launch settings in Model Details |
| `t` | Fold or unfold model thinking |
| `s` | Cycle dashboard placement: right, bottom, off |
| `y` | Copy the visible scroll buffer to the clipboard |
| `/` | Open the searchable command palette |
| `Esc` | Cancel the current palette entry; does not quit |
| `q` | Quit the console |

In the command palette, type to filter commands, use `↑`/`↓` to choose a suggestion, `Tab` to complete it, and `Enter` to run it. Press `Esc` to close the palette.

The dashboard can also be placed at startup:

```bash
llamatap console --stats right
llamatap console --stats bottom
llamatap console --stats off
```

Other console startup options are `--fold-thinking` (start with thinking folded), `--repeats` (show repeated history), `--preamble` (open the system prompt), `--name LABEL` (override the detected harness name), `--no-flags` (skip launch-flag lookup), and `--file PATH` (open a saved recording).

## Console slash commands

Type `/` and choose a command, or type the command and its arguments. `/help` lists the commands inside the console.

| Command | What it does |
|---|---|
| `/copy [raw]` | Copy the scroll buffer; `raw` copies raw tool-call JSON. |
| `/save [path]` | Save the rendered buffer to a file. Without a path, saves to the Llamatap state directory. |
| `/pager` | Open the buffer in `less` (or `more`) for terminal search and selection. |
| `/clear` | Archive the current recording, then start a fresh one. If archiving fails, it leaves the recording intact. |
| `/ask <message>` | Send a follow-up directly to the model in the latest captured context. This bypasses the harness: tools are not run. |
| `/replay N [--model NAME] [--diff]` | Replay exchange N; optionally target another model or compare with the original. |
| `/read N` | Jump to exchange N. |
| `/list` | Show the numbered exchange index. |
| `/raw` | Toggle raw JSON and readable tool-call arguments. |
| `/preamble` | Open or close the system prompt in the conversation. |
| `/flags` | Show or hide launch settings in Model Details. |
| `/repeats [open\|close]` | Toggle repeated messages, or expand/collapse them all. |
| `/thinking [open\|close]` | Toggle thinking display, or expand/collapse all thinking blocks. |
| `/stats [right\|bottom\|off]` | Set dashboard placement; without an argument, cycle through placements. |
| `/name <label>` | Set the displayed harness name. |
| `/help` | Show commands and key bindings. |
| `/quit` | Exit the console. |

Aliases accepted by the command handler: `/exit` and `/q` for `/quit`; `/think` for `/thinking`; `/panel` and `/rail` for `/stats`; `/json` for `/raw`. `/dedupe` also toggles repeated messages.

## Other ways to inspect recordings

The default recording is stored at `~/.local/state/llamatap/llamatap-records.jsonl`. Llamatap archives the previous recording when a new `serve` starts; `llamatap clear` archives before clearing. Archives are stored under `~/.local/state/llamatap/runs/` by default.

```bash
llamatap list                         # list captured exchanges
llamatap read 3                       # read exchange 3
llamatap all                          # render all exchanges
llamatap follow                       # follow the recording in a plain-text view
llamatap log                          # follow the compact request log
llamatap archive                      # archive without clearing the live recording
llamatap read --file /path/to/recording.jsonl
llamatap console --file /path/to/recording.jsonl
llamatap replay 3 --model another-model --diff
```

Most inspection commands accept `--file PATH` to use an archived JSONL recording. `llamatap replay N` sends the captured request to the configured upstream; use `--model` to select another model. Replay generates a new response and appends it to the recording being used.

## Privacy and sharing

Recordings and logs live in the Llamatap state directory, normally `~/.local/state/llamatap`. The dispatcher creates the state directory with owner-only access and recording files with private permissions. Recordings may include prompts, tool schemas, source snippets, paths, and credentials included in requests. Review a recording before sharing it; archive files are equally sensitive.

## Troubleshooting

**`llamatap: command not found`** — Confirm `~/.local/bin` is on `PATH`, or run the launcher by its full path, such as `~/GIT/llamatap/llamatap --help`.

**`llamatap config` says the upstream is unreachable** — Confirm `llama-server` is running, its `--port` matches `upstream`, and the configured URL has no extra `/v1` suffix. Llamatap appends `/v1/models` for its reachability check.

**The harness cannot connect to Llamatap** — Confirm `llamatap serve` is still running and that the harness base URL uses Llamatap's port (`http://127.0.0.1:9293/v1` by default), not the model server's port.

**The port is already in use** — Stop the other process using the port, or choose a different tap port with `LLAMATAP_PORT` and update the harness URL to match.

**The console says it needs a terminal** — Run `llamatap console` in an interactive terminal. For a pipeable live view, use `llamatap follow`.

**The console cannot copy** — Install one of `wl-copy`, `xclip`, or `xsel`, or use `/save PATH` and copy from the saved file.

## Known protocol limitation

Llamatap records Chat Completions, Anthropic Messages, and Responses API traffic. The current reader renders complete model turns only for Chat Completions. Anthropic and Responses exchanges are recorded and can be replayed, but their model turns may not appear in the readable thread yet.
