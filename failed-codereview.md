# Failed code review

Do not release yet: normal traffic can crash the TUI, streaming is buffered, instructions can be lost, and recording-management paths can permanently delete data. Full-application inspection and focused in-memory reproductions confirmed several failures; live network integration was restricted, and no automated test suite was found. No repository files were changed during the review.

## Full review comments

- **[P1] Return a consistent result for requests without JSON bodies** — `/home/<user>/GIT/llamatap/llamatap-thread.py:111-113`
  A normal `GET /v1/models` is recorded with `req: null`. Loading that recording in the console calls `conversation()`, which unpacks this return value into two variables and raises `ValueError`. One model-discovery request therefore makes the entire recording unusable in the TUI. Return an empty preamble and turns consistently, and handle non-conversation exchanges explicitly.

- **[P1] Forward available stream bytes without filling a 4096-byte buffer** — `/home/<user>/GIT/llamatap/llamatap-proxy.py:356-361`
  `HTTPResponse.read(4096)` waits for that many bytes or EOF, including across upstream HTTP chunks. Small SSE events consequently accumulate instead of reaching the harness immediately; short replies can arrive only after generation finishes. This also inflates measured time to first token and distorts fallback decode rates. Use an available-data read such as `read1()`; a simulated chunked response reproduced the delay.

- **[P1] Archive recordings before the CLI clear command truncates them** — `/home/<user>/GIT/llamatap/llamatap:185-185`
  `llamatap clear` immediately destroys both files without making an archive. This contradicts the command help and README and differs from the console's `/clear` behavior. A user following the documented archive-and-clear workflow permanently loses the only recording. Share the archive implementation and only clear after successful preservation.

- **[P1] Allocate unique archive destinations instead of overwriting them** — `/home/<user>/GIT/llamatap/llamatap-proxy.py:602-607`
  Two restarts with recorded traffic within the same minute select the same archive directory, and `os.replace()` silently overwrites the previous recording. The console's `cmd_clear()` has the same problem. Archive creation needs collision-resistant names and protection against replacing existing archives; minute-resolution timestamps do not preserve the promised history.

- **[P1] Restrict system-message hoisting to the intended protocol** — `/home/<user>/GIT/llamatap/llamatap-proxy.py:463-469`
  This rewrite also runs on OpenAI chat-completion requests. Any system message after index zero is removed from `messages` and placed in Anthropic's top-level `system` field, which OpenAI-compatible servers can ignore or reject. Valid requests therefore lose instructions or fail merely by passing through the tap. Gate the compatibility shim by protocol and preserve OpenAI system messages.

- **[P2] Preserve string-valued developer content during merging** — `/home/<user>/GIT/llamatap/llamatap-proxy.py:510-517`
  Responses input messages can contain a plain string. This loop iterates that string character by character and extracts nothing, but still excludes the message from `kept`. If another developer/system item supplies a text block, the rewrite proceeds and silently deletes the string-valued instruction. Normalize both supported content forms before removing any input items.

- **[P2] Finalize request state even when forwarding fails** — `/home/<user>/GIT/llamatap/llamatap-proxy.py:444-445`
  Client cancellations, upstream disconnects, and read timeouts bypass these finalization calls. The exchange is never recorded and its entry remains in `_active`, making the console display a permanently running request until proxy restart. Upstream connection failures likewise escape the `HTTPError` handler. Put cleanup and upstream-response closure in `finally`, retain partial/error records, and return a gateway error when headers have not yet been sent.

- **[P2] Handle chunked request bodies before forwarding their headers** — `/home/<user>/GIT/llamatap/llamatap-proxy.py:278-282`
  For a client using `Transfer-Encoding: chunked`, there is no Content-Length, so the proxy reads no body while forwarding the transfer-encoding header. The upstream receives an empty or unterminated request, and the unread chunk bytes remain on the downstream connection. Decode chunked requests correctly, or explicitly reject them and close the connection instead of forwarding inconsistent framing.

- **[P2] Parse Responses API output rather than displaying no answer** — `/home/<user>/GIT/llamatap/llamatap-thread.py:165-172`
  The advertised Responses API path produces `output` items and `response.output_text.delta` events, but this parser only understands Chat Completions and parts of Anthropic responses. A valid Responses answer renders as no model turn; Responses function-call input/output items are also omitted by `normalise()`. Add protocol-specific normalization so captured Responses exchanges remain inspectable.

- **[P2] Reconstruct Anthropic streamed tool calls and thinking blocks** — `/home/<user>/GIT/llamatap/llamatap-thread.py:208-211`
  For streamed Anthropic replies, only `delta.text` is extracted. Tool names arrive in `content_block_start`, arguments in `input_json_delta`, and reasoning in thinking deltas, so Claude Code tool-call-only responses render as nothing. Track content blocks by index and reconstruct these events, including stop reasons, rather than silently dropping the main content of agent exchanges.

- **[P2] Include the captured assistant response when continuing with ask** — `/home/<user>/GIT/llamatap/llamatap-console.py:1427-1431`
  `rec.req.messages` ends before the assistant response stored in `rec.resp`. Appending the new user message directly skips that response, so `/ask explain your last answer` cannot see the answer displayed immediately above it. Repeated `/ask` calls similarly omit each preceding answer. Reconstruct the completed exchange before appending the follow-up, with explicit handling for unresolved tool calls.

- **[P2] Pass the console's recording path to replay** — `/home/<user>/GIT/llamatap/llamatap-console.py:1490-1494`
  When the console is opened with `--file archive.jsonl`, `/replay N` launches the replay script without that path. It instead reads and appends to the default live recording, potentially sending an entirely different exchange to the model. Pass `self.path` as the subprocess's `--file` argument.

- **[P2] Decode streamed originals when displaying replay comparisons** — `/home/<user>/GIT/llamatap/llamatap-replay.py:64-67`
  Most harness recordings contain SSE responses, but `reply_of()` only accepts a single JSON document. Consequently `replay N --diff` labels a successfully captured streamed answer as an empty response and omits its reasoning and tool calls. Reuse the stream-aware response normalization for the original side of the comparison.

- **[P2] Preserve complete replay response bodies** — `/home/<user>/GIT/llamatap/llamatap-replay.py:186-186`
  Replay responses exceeding 262,144 characters are cut in the middle of their JSON document before being recorded. A long generated file or tool argument then becomes unparsable, and subsequent readers lose the entire answer rather than just its tail. Preserve the complete response, as the proxy now does, or use an explicit truncation representation that remains readable.

- **[P2] Query the direct llama-server properties endpoint** — `/home/<user>/GIT/llamatap/llamatap-thread.py:256-263`
  The default and documented direct llama-server deployment exposes `/props`, not llama-swap's `/upstream/<model>/props`. This lookup always fails for that deployment and permanently caches an empty result, hiding available context-window and sampling information. Support the direct endpoint as well as the swap-specific route.

- **[P2] Create sensitive recording storage with private permissions** — `/home/<user>/GIT/llamatap/llamatap:77-81`
  With a conventional umask of 022, this creates a 0755 state directory and the Python writers create 0644 recordings, logs, and in-flight files. If parent directories are traversable by other local users, they can read captured source code, prompts, and secrets. Establish owner-only directory/file permissions rather than relying on the user's home permissions or ambient umask.

- **[P2] Avoid reparsing the entire recording on every idle poll** — `/home/<user>/GIT/llamatap/llamatap-console.py:1666-1669`
  Every half-second the UI thread reloads and JSON-decodes the complete recording even when nothing changed. Recordings contain full conversation history per request and uncapped responses, so longer sessions repeatedly allocate and parse large amounts of data while blocking keyboard handling. Check file identity/size first and incrementally consume appended complete records, resetting appropriately after rotation or truncation.
