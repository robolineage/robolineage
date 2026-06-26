#!/usr/bin/env python3
"""Verify each file in a diff is owned by exactly one RoboLineage runtime domain.

Reads `scripts/OWNERSHIP.yaml`; for each path, finds the longest matching
prefix in `rules`. Files matching `shared` are exempt from single-owner
checks. With `--expected-owner X`, all touched files must belong to owner X
(or `shared`); useful in CI for focused PRs.

Usage:
    python scripts/check_ownership.py --files <f1> <f2> ...
    python scripts/check_ownership.py --from-git <revspec> [--expected-owner X]
    python scripts/check_ownership.py --from-git HEAD~3..HEAD --expected-owner H

Exit codes:
    0 - every file has a single owner (and matches --expected-owner if given)
    1 - some file has no owner (path not under any rule + not shared)
    2 - some file's owner != --expected-owner
    3 - internal config error (rules conflict, missing yaml, etc.)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OWNERSHIP_PATH = REPO_ROOT / "scripts" / "OWNERSHIP.yaml"


@dataclass(frozen=True)
class OwnershipConfig:
    rules: tuple[tuple[str, str], ...]   # (path_prefix, owner)
    shared: tuple[str, ...]


def _parse_yaml_minimal(text: str) -> OwnershipConfig:
    """Tiny PyYAML-free parser - accepts only the shape OWNERSHIP.yaml uses.

    Recognises:
        rules:
          - path: <str>
            owner: <str>
        shared:
          - <str>

    Comments (#...) and blank lines are skipped. Unrecognised top-level
    keys are silently ignored, so the file can grow optional sections.
    """
    rules: list[tuple[str, str]] = []
    shared: list[str] = []
    current_section: str | None = None
    pending_path: str | None = None

    for raw_line in text.splitlines():
        # Strip trailing comments + whitespace
        line = re.sub(r"\s+#.*$", "", raw_line).rstrip()
        if not line.strip():
            continue
        # Top-level keys
        if line.endswith(":") and not line.startswith(" "):
            key = line[:-1].strip()
            if key == "rules":
                current_section = "rules"
            elif key == "shared":
                current_section = "shared"
            else:
                current_section = None  # unknown section - skip its contents
            pending_path = None
            continue
        if current_section is None:
            continue
        if current_section == "shared":
            m = re.match(r"\s*-\s+(.+?)\s*$", line)
            if m:
                value = m.group(1)
                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    value = value[1:-1]
                shared.append(value)
            continue
        if current_section == "rules":
            m_path = re.match(r"\s*-\s+path:\s*(.+?)\s*$", line)
            m_owner = re.match(r"\s+owner:\s*(.+?)\s*$", line)
            if m_path:
                pending_path = m_path.group(1).strip().strip("'\"")
            elif m_owner and pending_path is not None:
                owner = m_owner.group(1).strip().strip("'\"")
                rules.append((pending_path, owner))
                pending_path = None

    return OwnershipConfig(rules=tuple(rules), shared=tuple(shared))


def load_ownership(path: Path | None = None) -> OwnershipConfig:
    """Load and validate OWNERSHIP.yaml. Detects rule prefix collisions
    (two distinct entries with the same `path`)."""
    target = path or DEFAULT_OWNERSHIP_PATH
    text = target.read_text(encoding="utf-8")
    cfg = _parse_yaml_minimal(text)

    # No two rules may share the same path prefix with different owners -
    # that would make ownership ambiguous.
    seen: dict[str, str] = {}
    for prefix, owner in cfg.rules:
        if prefix in seen and seen[prefix] != owner:
            raise SystemExit(
                f"OWNERSHIP.yaml conflict: prefix {prefix!r} mapped to "
                f"both {seen[prefix]!r} and {owner!r}"
            )
        seen[prefix] = owner

    return cfg


def _matches_shared(path: str, shared_patterns: Sequence[str]) -> bool:
    for pat in shared_patterns:
        # Directory pattern (ends with /) -> prefix match
        if pat.endswith("/"):
            if path.startswith(pat):
                return True
        # Glob-ish pattern (contains *) -> simple regex translation
        elif "*" in pat:
            regex = "^" + re.escape(pat).replace(r"\*", "[^/]*") + "$"
            if re.match(regex, path):
                return True
        # Exact path
        elif path == pat:
            return True
    return False


def find_owner(path: str, cfg: OwnershipConfig) -> str | None:
    """Longest-prefix match. Returns owner letter or None if unowned."""
    best_prefix = ""
    best_owner: str | None = None
    for prefix, owner in cfg.rules:
        if path.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_owner = owner
    return best_owner


def diff_files_from_git(revspec: str) -> list[str]:
    """Run `git diff --name-only <revspec>` and return changed files, except deletions."""
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "diff", "--name-only", "--diff-filter=ACMR", revspec],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# Checking ---------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    path: str
    kind: str          # "unowned" | "wrong_owner" | "ok" | "shared"
    owner: str | None  # actual owner (None for unowned)


def check(
    files: Iterable[str],
    cfg: OwnershipConfig,
    *,
    expected_owner: str | None = None,
) -> list[Finding]:
    """Classify each file. `expected_owner` flag adds wrong_owner findings."""
    findings: list[Finding] = []
    for path in files:
        if _matches_shared(path, cfg.shared):
            findings.append(Finding(path, "shared", None))
            continue
        owner = find_owner(path, cfg)
        if owner is None:
            findings.append(Finding(path, "unowned", None))
            continue
        if expected_owner is not None and owner != expected_owner:
            findings.append(Finding(path, "wrong_owner", owner))
        else:
            findings.append(Finding(path, "ok", owner))
    return findings


def report(findings: list[Finding], expected_owner: str | None = None) -> int:
    """Print findings; return exit code per the table in this script's docstring."""
    unowned = [f for f in findings if f.kind == "unowned"]
    wrong = [f for f in findings if f.kind == "wrong_owner"]
    if unowned:
        print(f"ERROR: {len(unowned)} unowned file(s):", file=sys.stderr)
        for f in unowned:
            print(f"   {f.path}", file=sys.stderr)
    if wrong:
        print(
            f"ERROR: {len(wrong)} file(s) owned by other domain(s) "
            f"(expected={expected_owner!r}):",
            file=sys.stderr,
        )
        for f in wrong:
            print(f"   [{f.owner}] {f.path}", file=sys.stderr)
    if not unowned and not wrong:
        owners = {f.owner for f in findings if f.kind == "ok"}
        share_count = sum(1 for f in findings if f.kind == "shared")
        owned_count = len(findings) - share_count
        print(
            f"OK: {owned_count} owned file(s) "
            f"({sorted(o for o in owners if o)}); "
            f"{share_count} shared",
        )
    if unowned:
        return 1
    if wrong:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_ownership.py")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs="+", help="explicit list of files to check")
    src.add_argument("--from-git", metavar="REVSPEC",
                     help="run `git diff --name-only REVSPEC` and check those files")
    parser.add_argument("--expected-owner",
                        help="if set, fail on any file whose owner != this letter")
    parser.add_argument("--ownership", type=Path, default=None,
                        help="path to OWNERSHIP.yaml (default: scripts/OWNERSHIP.yaml)")
    args = parser.parse_args(argv)

    try:
        cfg = load_ownership(args.ownership)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.files:
        files = list(args.files)
    else:
        try:
            files = diff_files_from_git(args.from_git)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: git diff failed: {exc.stderr}", file=sys.stderr)
            return 3

    findings = check(files, cfg, expected_owner=args.expected_owner)
    return report(findings, expected_owner=args.expected_owner)


if __name__ == "__main__":
    raise SystemExit(main())
