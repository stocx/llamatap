#!/usr/bin/env python3
"""What we know how to recognise, and how to show it.

Kept separate from the renderer so adding a newly-recognised thing is a one-line
edit here rather than surgery on the display code. The rule for this project:
anything we can identify gets a name and a readable form — and `--raw`
turns all of it off.
"""

# ---------------------------------------------------------------- tool classes
#
# Matched against the tool NAME, first hit wins, so order matters. Harnesses
# all invent their own names for the same four or five operations, which is why
# this is a table rather than a lookup: `bash` / `run_js` / `execute_command`
# are one category, and a reader should not have to know which harness is
# speaking to see that.
TOOL_CLASSES = [
    ("mcp",     "", ("mcp__",)),
    ("skill",   "", ("skill", "load_skill", "use_skill")),
    ("agent",   "", ("agent", "task", "subagent", "delegate", "dispatch")),
    ("shell",   "", ("bash", "shell", "run_js", "run_javascript", "execute",
                        "execute_command", "run_command", "terminal", "sh")),
    ("edit",    "", ("edit", "str_replace", "replace_in_file", "apply_diff",
                        "patch", "multiedit")),
    ("write",   "", ("write", "write_file", "write_to_file", "create_file", "new_file")),
    ("read",    "", ("read", "read_file", "cat", "view", "open_file")),
    ("search",  "", ("grep", "search", "search_files", "ripgrep", "codebase_search")),
    ("list",    "", ("ls", "list", "list_files", "glob", "file_glob_search", "tree")),
    ("web",     "", ("web", "fetch", "webfetch", "websearch", "browser", "curl")),
    ("todo",    "", ("todo", "todowrite", "task_list", "plan")),
    ("ask",     "", ("ask", "ask_followup_question", "followup", "attempt_completion")),
]


def classify_tool(name):
    """(category, icon) for a tool name.

    The icon is now always empty. The CATEGORY was the load-bearing part — it
    groups tools doing the same job under different names across harnesses —
    and callers still get it unchanged. The pictures were decoration, and
    fifteen of them in one pane read as cartoonish.
    """
    n = (name or "").lower()
    for cat, icon, needles in TOOL_CLASSES:
        for needle in needles:
            if n.startswith(needle) or n == needle or needle in n:
                return cat, icon
    return "tool", ""


# ------------------------------------------------------- content-block markers
#
# Things harnesses inject into the prompt that are NOT the user's words. Being
# able to see these separated out is most of the point of the tap: on this
# project an injected environment snapshot and a project instructions file have
# repeatedly turned out to be most of a "small" request.
CONTENT_MARKERS = [
    ("project instructions", "", ("AGENTS.md", "CLAUDE.md", "AGENT.md",
                                    ".cursorrules", "CONVENTIONS.md")),
    ("environment snapshot", "", ("<environment_details>", "# Current Working Directory",
                                    "<workspace_diagnostics>", "Current time:")),
    ("system reminder",      "", ("<system-reminder>",)),
    ("skills index",         "", ("<available_skills>", "Available skills:",
                                    "## Skills", "progressive disclosure")),
    ("mcp servers",          "", ("<mcp_servers>", "MCP SERVERS", "Connected MCP")),
    ("file contents",        "", ("<file_content", "```\n#!/", "<document>")),
    ("git state",            "", ("<git_status>", "Recent commits:", "git diff --stat")),
    ("todo state",           "", ("<todo_list>", "# Task List")),
    # Needles must be SECTION HEADERS, not phrases. "TOOL USE" on its own
    # matched the sentence "not including tool use or code generation", so the
    # chip claimed a tool-rules section that was not there.
    ("tool rules",           "", ("# Tool Use Rules", "# Tool usage policy",
                                    "## Tool Use Guidelines", "# Tool Use Policy")),
]


# NOTE: `markers_in` is the last user of CONTENT_MARKERS, the hardcoded phrase
# list. Prompt STRUCTURE now comes from `sections()` below, which reads the
# prompt's own headings — these phrases are only a decorative hint and must
# never be the source of a section.
#
# Matching is CASE-SENSITIVE throughout: the needles are literal markup like
# `<available_skills>` and literal headings like `# Tool usage policy`.
# Matching case-insensitively in one place and sensitively in another is how a
# chip came to be listed in the menu and then report "not located" when opened.
def markers_in(text):
    """Every recognised block present in a chunk of text, as (label, icon).

    Reported rather than stripped: the point is to tell you what a harness
    quietly added, and how much of the prompt it accounts for.
    """
    found, t = [], text or ""
    for label, icon, needles in CONTENT_MARKERS:
        for needle in needles:
            if needle in t:
                found.append((label, icon))
                break
    return found


# ------------------------------------------------------------------ turn kinds
TURN_STYLE = {
    "you":          ("", "YOU"),
    "model":        ("", "LLAMA"),
    "thinking":     ("", "LLAMA (thinking)"),
    "tool call":    ("", "LLAMA calls"),
    "tool result":  ("", "tool result"),
    "system":       ("", "system prompt"),
    "cut off":      ("", "TRUNCATED"),
}


# ------------------------------------------------------- background requests
#
# Calls a harness makes on its own behalf, not because you asked for anything.
# They matter disproportionately: opencode's title generator has been measured
# taking TWICE as long as the real request and burning 1,000 tokens of
# reasoning to choose the word "Greeting". Showing their prompts as "YOU"
# implies the user typed them, which is the opposite of the point.
BACKGROUND = [
    ("title generator", "", ("Generate a title for this conversation",
                                "You are a title generator",
                                "generate a short title",
                                "Generate a brief title")),
    ("summariser",      "", ("Summarize the conversation so far",
                               "You are a conversation summarizer",
                               "summarize this conversation")),
    ("autocomplete",    "", ("<|fim_prefix|>", "Complete the following code")),
    ("embedding",       "", ("Generate an embedding",)),
]


# Needles lowered once at import, not once per comparison.
_BACKGROUND_LC = [(label, icon, tuple(n.lower() for n in needles))
                  for label, icon, needles in BACKGROUND]


def background_kind(text):
    """(label, icon) if this looks like a harness talking to itself.

    `t.lower()` used to sit INSIDE the needle loop, so a 23,000-character
    system prompt was lowercased once per needle — about twenty times per
    record, 182 ms of a single console rebuild.
    """
    if not text:
        return None, None
    t = text.lower()
    for label, icon, needles in _BACKGROUND_LC:
        for needle in needles:
            if needle in t:
                return label, icon
    return None, None


# ----------------------------------------------------------------- parties
#
# One icon per party, used EVERYWHERE that party appears — the participants
# Parties are told apart by a coloured accent bar down the left of each block
# and by their name in plain text, the way opencode does it. That scales to a
# fifth party without inventing another picture, and it survives being copied
# out of the terminal, which icon-plus-colour does not.
PARTY = {
    "you":     "",
    "harness": "",
    "server":  "",
    "model":   "",
}
TOOLBOX = ""


# ------------------------------------------------- structure, not vocabulary
import re as _re

_HEADING = _re.compile(r"^(#{1,6})\s+(.{1,60}?)\s*$", _re.M)
_TAG = _re.compile(r"^<([a-z_][a-z0-9_.-]{1,40})>\s*$", _re.M | _re.I)


def sections(text):
    """Split a system prompt using ITS OWN structure.

    [(label, offset, length)] from markdown headings and XML-ish tag opens.

    This replaces a hardcoded list of phrases like "AGENTS.md" and
    "<available_skills>". Those were opencode's vocabulary: on any other
    harness they would silently find nothing, and you could not tell "this
    prompt has no sections" from "llamatap does not recognise this prompt".

    Headings and tags are how prompts are actually organised, whoever wrote
    them, and the label shown is the prompt's own words rather than a name
    invented here. Nothing is guessed: if a prompt has no headings, it has no
    sections, and that is the truth about it.
    """
    t = text or ""
    found = []
    for m in _HEADING.finditer(t):
        found.append((m.group(2).strip(), m.start(), len(m.group(1))))
    for m in _TAG.finditer(t):
        found.append((f"<{m.group(1)}>", m.start(), 1))
    found.sort(key=lambda x: x[1])
    out = []
    for i, (label, off, depth) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(t)
        out.append({"label": label, "offset": off, "length": end - off,
                    "depth": depth})
    return out


def section_icon(label):
    """Empty. Kept so callers need not change, and so the one place that
    decided what a section looked like stays the one place to change it.

    The label has always come from the prompt itself; this only ever put a
    picture beside it."""
    return ""
