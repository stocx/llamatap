#!/usr/bin/env python3
"""Move a recording aside without ever destroying one.

The single implementation, used by all three callers — `llamatap serve` on
startup, the console's `/clear`, and the dispatcher's `llamatap clear`. There
were previously two copies and a shell one-liner, which is how they came to
disagree about whether `clear` archives at all.

Two rules, both paid for:

1. **Nothing is overwritten.** Destinations used to be stamped to the minute
   and created with `exist_ok=True`, then files moved in with `os.replace` —
   so two `serve` starts within the same minute silently destroyed the first
   recording. A destination here is created exclusively and a taken name is
   never reused.

2. **Truncation only follows a successful archive.** `llamatap clear` was a
   shell `: > file` with no archive at all, while its own help text and the
   README both promised one. If archiving fails, the recording stays.

Run directly, it prints the destination on success and exits non-zero on
failure, so a shell caller can gate on it:

    llamatap-archive.py --runs DIR FILE [FILE ...]
"""

import os
import sys
import time


def archive(paths, runs_dir, stamp=None):
    """Move each existing non-empty path into a fresh directory under runs_dir.

    Returns the destination directory, or None if there was nothing to do.
    Raises on failure — the caller must not truncate anything after an
    exception.
    """
    have = [p for p in paths
            if p and os.path.exists(p) and os.path.getsize(p) > 0]
    if not have:
        return None
    dest = _fresh_dir(runs_dir, stamp or time.strftime("%m%d-%H%M%S"))
    for p in have:
        target = os.path.join(dest, os.path.basename(p))
        if os.path.exists(target):
            # Two sources with the same basename; keep both rather than
            # letting one quietly win.
            target = _unique_file(target)
        os.replace(p, target)
    return dest


def _fresh_dir(runs_dir, stamp):
    """A directory that did not exist until this call.

    `makedirs(exist_ok=True)` was the bug: it happily handed back a directory
    that already held someone else's recording.
    """
    os.makedirs(runs_dir, exist_ok=True)
    base = os.path.join(runs_dir, f"llamatap-{stamp}")
    for suffix in [""] + [f"-{i}" for i in range(2, 1000)]:
        try:
            os.mkdir(base + suffix, 0o700)
            return base + suffix
        except FileExistsError:
            continue
    raise FileExistsError(f"could not find a free archive name under {runs_dir}")


def _unique_file(path):
    stem, ext = os.path.splitext(path)
    for i in range(2, 1000):
        cand = f"{stem}-{i}{ext}"
        if not os.path.exists(cand):
            return cand
    raise FileExistsError(path)


def main(argv):
    runs = None
    files = []
    it = iter(argv)
    for a in it:
        if a == "--runs":
            runs = next(it, None)
        else:
            files.append(a)
    if not runs:
        print("usage: llamatap-archive.py --runs DIR FILE [FILE ...]",
              file=sys.stderr)
        return 2
    try:
        dest = archive(files, runs)
    except Exception as e:
        print(f"could not archive: {e}", file=sys.stderr)
        return 1
    if dest is None:
        print("nothing to archive")
        return 0
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
