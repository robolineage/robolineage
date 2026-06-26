#!/usr/bin/env python3
"""Forbid direct imports across RoboLineage runtime domains.

Walks `src/` and flags any line of the form

    from robolineage_<sibling>.<...> import ...
    import robolineage_<sibling>.<...>

where `robolineage_<sibling>` is a module owned by a different runtime domain than
the file doing the import. The default rule is strict: shared data types go
through `robolineage_contracts` or `robolineage_schemas`; orchestration exceptions are listed
explicitly below.

Allowed paths:

  - `robolineage_contracts.*` and `robolineage_schemas.*`
      Stable cross-domain contract/schema layer.
  - `robolineage_app` integration set:
      robolineage_data_source.{config.loader, config.schema, orchestrator}
      robolineage_robot.{RobotProfile, RobotProfileRegistry, load_robot_profile, ...}
      robolineage_session.api.create_app
      robolineage_shared_agents.visual_snapshot.realtime.{StreamingRuntimePipeline, run_ros_topic_stream}
      robolineage_shared_agents.visual_snapshot.{TaskConfig, OpenAIVLMRunner, MockVLMRunner, Qwen2VLRunner}
      robolineage_shared_agents.master.MasterAgent
      robolineage_train.{CommandIntake, FrameworkDiscoveryAgent, FrameworkRemoteExecution, TrainingLifecycleRunner, load_framework_profile}
      robolineage_post_rollout.{PostRolloutReviewAgent, PostRolloutReviewWorker}
      robolineage_eval.{EvaluationReviewWorker, PolicyEvaluationAgent, DeploymentGovernanceAgent}
      These imports are confined to the unified runtime / health endpoint.
  - Policy evaluation set:
      src/robolineage_eval may depend on BaseVLMRunner as an optional VLM interface,
      while online priority remains enforced by robolineage_app's coordinator.
      src/robolineage_eval may also reuse post-rollout formal review agents because
      evaluation is the same evidence/annotation/failure pass with dataset
      admission replaced by policy scoring.
  - Post-review set:
      src/robolineage_post_rollout may depend on the VSA VLM runner protocol and VLM
      exception class so offline review can reuse the same provider adapter.
  - Training lifecycle set:
      src/robolineage_train/lifecycle.py may call robolineage_dataset.DatasetUpdater because
      it is the explicit cross-stage runner from post-review selection into a
      frozen dataset version and policy metadata.
  - Realtime VSA ROS2 set:
      src/robolineage_shared_agents/visual_snapshot/realtime may read the stable ROS2 topic
      specs and robot-state decoder from robolineage_data_source while VSA subscribes
      ROS2 topics directly.

Caller-side mapping (file path -> runtime domain owner) reuses
`OWNERSHIP.yaml`.

Usage:
    python scripts/check_contracts_imports.py
    python scripts/check_contracts_imports.py --root src/
    python scripts/check_contracts_imports.py --paths src/robolineage_shared_agents src/robolineage_data_source

Exit codes:
    0 - clean
    1 - found at least one violation
    3 - internal error (cannot resolve owner, etc.)
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

# Reuse OWNERSHIP loader from the sibling check script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ownership import OwnershipConfig, find_owner, load_ownership  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent

# Always-allowed import paths (case sensitive).
ALWAYS_ALLOWED_PREFIXES = (
    "robolineage_contracts.",
    "robolineage_contracts ",
    "robolineage_contracts\n",
    "robolineage_schemas.",
    "robolineage_schemas ",
    "robolineage_schemas\n",
    "robolineage_shared_agents.json_llm",
    "robolineage_shared_agents.llm_routes",
)

# Specific cross-domain whitelisted symbols.
# Format: ("module.path", "symbol_or_*", "caller_path_prefix_or_None")
# - Use "*" for "any symbol from this module"
WHITELIST: tuple[tuple[str, str, str | None], ...] = (
    # The robolineage_app unified launcher imports symbols from the functional domains
    # it composes into one process. These imports are intentional and confined
    # to src/robolineage_app/*.py.
    ("robolineage_data_source.config.loader", "*", "src/robolineage_app/"),
    ("robolineage_data_source.config.schema", "*", "src/robolineage_app/"),
    ("robolineage_data_source.orchestrator", "*", "src/robolineage_app/"),
    ("robolineage_data_source.adapters.base", "*", "src/robolineage_app/"),
    ("robolineage_data_source.config.schema", "ArmTopicSpec", "src/robolineage_shared_agents/visual_snapshot/realtime/"),
    ("robolineage_data_source.adapters.ros2_profile", "_robot_state_to_27_vec", "src/robolineage_shared_agents/visual_snapshot/realtime/"),
    ("robolineage_robot", "*", "src/robolineage_app/"),
    ("robolineage_robot.profile", "*", "src/robolineage_app/"),
    ("robolineage_session.api", "create_app", "src/robolineage_app/"),
    ("robolineage_ar.types", "*", "src/robolineage_app/"),
    ("robolineage_ar.video_source", "*", "src/robolineage_app/"),
    ("robolineage_shared_agents.visual_snapshot.realtime", "*", "src/robolineage_app/"),
    ("robolineage_shared_agents.visual_snapshot", "*", "src/robolineage_app/"),
    ("robolineage_shared_agents.visual_snapshot.vlm_priority", "*", "src/robolineage_app/"),
    ("robolineage_shared_agents.visual_snapshot.vlm_runner", "*", "src/robolineage_app/"),
    ("robolineage_shared_agents.master", "MasterAgent", "src/robolineage_app/"),
    ("robolineage_train", "TrainingLifecycleRunner", "src/robolineage_app/"),
    ("robolineage_train", "CommandIntake", "src/robolineage_app/"),
    ("robolineage_train", "FrameworkDiscoveryAgent", "src/robolineage_app/"),
    ("robolineage_train", "FrameworkRemoteExecution", "src/robolineage_app/"),
    ("robolineage_train", "load_framework_profile", "src/robolineage_app/"),
    ("robolineage_post_rollout", "PostRolloutReviewAgent", "src/robolineage_app/"),
    ("robolineage_post_rollout", "PostRolloutReviewWorker", "src/robolineage_app/"),
    ("robolineage_eval", "EvaluationReviewWorker", "src/robolineage_app/"),
    ("robolineage_eval", "PolicyEvaluationAgent", "src/robolineage_app/"),
    ("robolineage_eval", "DeploymentGovernanceAgent", "src/robolineage_app/"),
    ("robolineage_shared_agents.visual_snapshot.vlm_runner", "BaseVLMRunner", "src/robolineage_eval/"),
    ("robolineage_post_rollout.formal_review", "*", "src/robolineage_eval/"),
    ("robolineage_shared_agents.visual_snapshot.exceptions", "VLMInferenceError", "src/robolineage_post_rollout/"),
    ("robolineage_shared_agents.visual_snapshot.vlm_runner", "BaseVLMRunner", "src/robolineage_post_rollout/"),
    ("robolineage_shared_agents.visual_snapshot.vlm_runner", "make_vlm_runner_from_env", "src/robolineage_post_rollout/"),
    ("robolineage_dataset", "DatasetUpdater", "src/robolineage_train/lifecycle.py"),
    ("robolineage_dataset.version", "parse_version", "src/robolineage_train/lifecycle.py"),
)


# `from X import Y` or `import X` - handles the most common forms. We don't
# implement full Python parsing because we just need to find the module name
# being imported.
_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+(?P<mod>[\w\.]+)\s+import\s+(?P<names>.+?)(?:\s*#.*)?$"
)
_DIRECT_IMPORT_RE = re.compile(
    r"^\s*import\s+(?P<mod>[\w\.]+)(?:\s+as\s+\w+)?(?:\s*,\s*[\w\.]+(?:\s+as\s+\w+)?)*\s*(?:#.*)?$"
)


@dataclass(frozen=True)
class Violation:
    file: str
    line_no: int
    line: str
    importer_owner: str | None
    target_module: str
    target_owner: str | None


def _is_robolineage_module(mod: str) -> bool:
    return mod.startswith("robolineage_")


def _module_to_path_prefixes(mod: str) -> list[str]:
    """Return candidate source path prefixes from most to least specific.

    Examples:
      robolineage_data_source.adapters -> ["src/robolineage_data_source/adapters/", "src/robolineage_data_source/"]
      robolineage_shared_agents.visual_snapshot.agent -> [
          "src/robolineage_shared_agents/visual_snapshot/agent/",
          "src/robolineage_shared_agents/visual_snapshot/",
          "src/robolineage_shared_agents/",
      ]
    """
    parts = mod.split(".")
    return [
        f"src/{'/'.join(parts[:i])}/"
        for i in range(len(parts), 0, -1)
    ]


def _module_owner(mod: str, cfg: OwnershipConfig) -> str | None:
    for prefix in _module_to_path_prefixes(mod):
        owner = find_owner(prefix, cfg)
        if owner is not None:
            return owner
    return None


def _is_whitelisted(target_module: str, names: str, caller_path: str) -> bool:
    """Names is the raw comma-separated list of imports from `from X import a, b`.
    Returns True if every imported name is whitelisted under target_module."""
    name_list = [n.strip().split(" as ")[0].strip() for n in names.split(",") if n.strip()]
    scoped_entries = [
        (entry_name, caller_scope)
        for entry_mod, entry_name, caller_scope in WHITELIST
        if target_module == entry_mod and _caller_matches(caller_path, caller_scope)
    ]
    if not scoped_entries:
        return False
    if any(entry_name == "*" for entry_name, _ in scoped_entries):
        return True
    allowed_names = {entry_name for entry_name, _ in scoped_entries}
    return bool(name_list) and all(name in allowed_names for name in name_list)


def _caller_matches(caller_path: str, caller_scope: str | None) -> bool:
    if caller_scope is None:
        return True
    return caller_path == caller_scope or caller_path.startswith(caller_scope)


def _imports_from_ast(text: str) -> list[tuple[int, str, str, str]]:
    """Extract real Python imports via AST.

    Returns list of (line_no, module, names_str, raw_line) tuples. Skips:
      - text inside docstrings / string literals (these aren't ast.Import nodes)
      - Files that fail to parse (e.g. corrupted, intentionally bad fixtures)

    `names_str` mimics the comma-separated form used by the legacy regex
    scanner so `_is_whitelisted` keeps working unchanged.
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    out: list[tuple[int, str, str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name
                for a in node.names
            )
            raw = lines[node.lineno - 1] if 0 <= node.lineno - 1 < len(lines) else ""
            out.append((node.lineno, mod, names, raw.rstrip()))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                raw = lines[node.lineno - 1] if 0 <= node.lineno - 1 < len(lines) else ""
                out.append((node.lineno, alias.name, "", raw.rstrip()))

    return out


def scan_file(path: Path, cfg: OwnershipConfig) -> List[Violation]:
    """Scan one .py file using AST. Returns any violations.

    Uses real Python AST (not regex) so docstring contents, multi-line
    strings, and comments are correctly ignored. The earlier regex
    implementation flagged `from robolineage_shared_agents.visual_snapshot import ...`
    appearing inside the docstring of `src/robolineage_shared_agents/__init__.py` - the
    AST walker only yields actual `ast.Import` / `ast.ImportFrom` nodes.
    """
    rel = str(path.relative_to(REPO_ROOT))
    importer_owner = find_owner(rel, cfg)
    if importer_owner is None:
        # File not under any owner - skip (probably a test fixture or script
        # we don't enforce on; falls through to the unowned-file check in
        # check_ownership.py).
        return []

    violations: list[Violation] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    for line_no, mod, names, raw in _imports_from_ast(text):
        if not _is_robolineage_module(mod):
            continue
        if any(mod == p.rstrip(" \n.") or mod.startswith(p.rstrip(" \n"))
               for p in ALWAYS_ALLOWED_PREFIXES):
            continue
        target_owner = _module_owner(mod, cfg)
        if target_owner is None or target_owner == importer_owner:
            continue
        # Cross-plan reference. Check whitelist (only meaningful for `from X
        # import a, b` form; bare `import X` always violates if cross-domain).
        if names and _is_whitelisted(mod, names, rel):
            continue
        violations.append(
            Violation(
                file=rel,
                line_no=line_no,
                line=raw,
                importer_owner=importer_owner,
                target_module=mod,
                target_owner=target_owner,
            )
        )
    return violations


def scan_paths(roots: Iterable[Path], cfg: OwnershipConfig) -> List[Violation]:
    """Walk the given roots and scan every .py file."""
    violations: list[Violation] = []
    for root in roots:
        if root.is_file():
            if root.suffix == ".py":
                violations.extend(scan_file(root, cfg))
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "_branch_import" in str(path):
                # Snapshot dirs are exempt - they're temporary
                continue
            violations.extend(scan_file(path, cfg))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_contracts_imports.py")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "src",
                        help="root to walk (default: src/)")
    parser.add_argument("--paths", nargs="+", type=Path, default=None,
                        help="explicit paths to scan (overrides --root)")
    parser.add_argument("--ownership", type=Path, default=None,
                        help="path to OWNERSHIP.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_ownership(args.ownership)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    roots = args.paths if args.paths else [args.root]
    violations = scan_paths(roots, cfg)

    if violations:
        print(f"ERROR: {len(violations)} cross-domain import violation(s):", file=sys.stderr)
        for v in violations:
            print(
                f"  {v.file}:{v.line_no}  "
                f"[{v.importer_owner}] imports {v.target_module} (owner={v.target_owner})\n"
                f"      {v.line}",
                file=sys.stderr,
            )
        print(
            "\nFix: route shared types through robolineage_contracts.* / robolineage_schemas.*",
            file=sys.stderr,
        )
        return 1
    print(f"OK: no cross-domain import violations under: {[str(r) for r in roots]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
