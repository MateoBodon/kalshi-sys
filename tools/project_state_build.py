#!/usr/bin/env python3
"""Generate project_state/_generated artifacts (inventory, symbols, imports, make targets)."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

EXCLUDED_TOP_LEVEL = {".git", ".venv", "__pycache__", "reports", "data"}
EXCLUDED_DIR_NAMES = {"__pycache__", ".pytest_cache"}


@dataclass
class PySymbol:
    name: str
    signature: str
    doc: str


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def should_skip_dir(rel: Path) -> bool:
    if not rel.parts:
        return False
    if rel.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    if rel.parts[0] == "docs" and len(rel.parts) > 1 and rel.parts[1] in {"agent_runs", "gpt_bundles", "gpt_outputs"}:
        return True
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    if "experiments" in rel.parts:
        for part in rel.parts:
            if part.startswith("outputs_"):
                return True
    return False


def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        # prune excluded dirs
        pruned = [d for d in dirnames if should_skip_dir(rel_dir / d)]
        if pruned:
            dirnames[:] = [d for d in dirnames if d not in pruned]
        if should_skip_dir(rel_dir):
            dirnames[:] = []
            continue
        for filename in filenames:
            rel = rel_dir / filename
            if should_skip_dir(rel.parent):
                continue
            yield Path(dirpath) / filename


def role_for_path(rel: Path) -> str:
    parts = rel.parts
    if not parts:
        return "other"
    first = parts[0]
    if first == "src":
        return "source"
    if first == "tests":
        return "test"
    if first == "configs":
        return "config"
    if first == "docs":
        return "docs"
    if first == "project_state":
        return "project_state"
    if first == "tools":
        return "tool"
    if first == "scripts":
        return "script"
    if first in {"docker", "deploy"}:
        return "infra"
    if first in {"jobs", "monitor", "report"}:
        return "ops"
    return "other"


def build_repo_inventory(root: Path) -> Dict:
    files = []
    total_size = 0
    for path in iter_files(root):
        rel = path.relative_to(root)
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        size = stat.st_size
        total_size += size
        files.append(
            {
                "path": rel.as_posix(),
                "size_bytes": size,
                "role": role_for_path(rel),
                "ext": path.suffix,
            }
        )
    excluded = []
    for top in sorted(EXCLUDED_TOP_LEVEL):
        p = root / top
        if p.exists():
            excluded.append({"path": top, "reason": "excluded from deep parsing"})
    ar = root / "docs" / "agent_runs"
    if ar.exists():
        excluded.append({"path": "docs/agent_runs", "reason": "excluded from deep parsing"})
    return {
        "generated_at": utc_now_iso(),
        "root": root.as_posix(),
        "summary": {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "excluded_count": len(excluded),
        },
        "files": sorted(files, key=lambda x: x["path"]),
        "excluded": excluded,
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def format_default(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def format_args(args: ast.arguments) -> str:
    parts: List[str] = []
    posonly = [a.arg for a in args.posonlyargs]
    normal = [a.arg for a in args.args]
    defaults = [None] * (len(posonly) + len(normal) - len(args.defaults)) + list(args.defaults)

    for name, default in zip(posonly + normal, defaults):
        if default is None:
            parts.append(name)
        else:
            parts.append(f"{name}={format_default(default)}")
    if args.posonlyargs:
        parts.insert(len(posonly), "/")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for kw, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is None:
            parts.append(kw.arg)
        else:
            parts.append(f"{kw.arg}={format_default(default)}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def format_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        sig = f"({format_args(node.args)})"
        if node.returns is not None:
            try:
                ret = ast.unparse(node.returns)
            except Exception:
                ret = "..."
            sig = f"{sig} -> {ret}"
        return sig
    if isinstance(node, ast.ClassDef):
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append("...")
        return f"({', '.join(bases)})" if bases else "()"
    return "()"


def first_line(text: Optional[str]) -> str:
    if not text:
        return ""
    line = text.strip().splitlines()[0]
    return line


def module_name_for_path(path: Path, root: Path) -> Optional[str]:
    rel = path.relative_to(root)
    if rel.parts[0] == "src":
        rel = Path(*rel.parts[1:])
    elif rel.parts[0] in {"tools", "experiments"}:
        # treat tools/experiments as top-level namespaces
        rel = Path(*rel.parts)
    else:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if not parts:
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def package_for_module(module: Optional[str], path: Path) -> Optional[str]:
    if module is None:
        return None
    if path.name == "__init__.py":
        return module
    if "." in module:
        return module.rsplit(".", 1)[0]
    return ""


def collect_python_files(root: Path) -> List[Path]:
    targets = []
    for base in (root / "src", root / "tools", root / "experiments"):
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(root)
            pruned = [d for d in dirnames if should_skip_dir(rel_dir / d)]
            if pruned:
                dirnames[:] = [d for d in dirnames if d not in pruned]
            if should_skip_dir(rel_dir):
                dirnames[:] = []
                continue
            for filename in filenames:
                if filename.endswith(".py"):
                    targets.append(Path(dirpath) / filename)
    return sorted(targets)


def parse_symbols(root: Path) -> Tuple[Dict, Dict[str, Set[str]]]:
    files = {}
    module_to_path: Dict[str, str] = {}
    for path in collect_python_files(root):
        mod = module_name_for_path(path, root)
        if mod:
            module_to_path[mod] = path.relative_to(root).as_posix()
    adjacency: Dict[str, Set[str]] = {}
    unresolved: Dict[str, Set[str]] = {}

    for path in collect_python_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            source = read_text(path)
            tree = ast.parse(source)
        except Exception as exc:
            files[rel] = {
                "module_doc": "",
                "functions": [],
                "classes": [],
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            continue
        module_doc = first_line(ast.get_docstring(tree))
        functions: List[Dict[str, str]] = []
        classes: List[Dict[str, str]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    {
                        "name": node.name,
                        "signature": format_signature(node),
                        "doc": first_line(ast.get_docstring(node)),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                classes.append(
                    {
                        "name": node.name,
                        "signature": format_signature(node),
                        "doc": first_line(ast.get_docstring(node)),
                    }
                )
        files[rel] = {
            "module_doc": module_doc,
            "functions": functions,
            "classes": classes,
            "errors": [],
        }

        current_module = module_name_for_path(path, root)
        current_package = package_for_module(current_module, path)
        adjacency.setdefault(rel, set())
        unresolved.setdefault(rel, set())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    resolved = False
                    if name in module_to_path:
                        adjacency[rel].add(module_to_path[name])
                        resolved = True
                    else:
                        # try top-level match (e.g. kalshi_alpha)
                        top = name.split(".")[0]
                        if top in module_to_path:
                            adjacency[rel].add(module_to_path[top])
                            resolved = True
                    if not resolved:
                        unresolved[rel].add(name)
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                base = node.module or ""
                full_module = base
                if level > 0:
                    if current_package is None:
                        full_module = base
                    else:
                        pkg = current_package
                        for _ in range(level - 1):
                            if "." in pkg:
                                pkg = pkg.rsplit(".", 1)[0]
                            else:
                                pkg = ""
                                break
                        full_module = f"{pkg}.{base}".strip(".") if base else pkg
                candidates = []
                if full_module:
                    candidates.append(full_module)
                for alias in node.names:
                    if full_module:
                        candidates.append(f"{full_module}.{alias.name}")
                    else:
                        candidates.append(alias.name)
                resolved_any = False
                for cand in candidates:
                    if cand in module_to_path:
                        adjacency[rel].add(module_to_path[cand])
                        resolved_any = True
                if not resolved_any:
                    unresolved[rel].add("from " + (full_module or ".") )

    import_graph = {
        "generated_at": utc_now_iso(),
        "root": root.as_posix(),
        "adjacency": {k: sorted(v) for k, v in sorted(adjacency.items())},
        "unresolved_imports": {k: sorted(v) for k, v in sorted(unresolved.items())},
    }
    return files, import_graph


def parse_make_targets(makefile: Path) -> List[str]:
    if not makefile.exists():
        return []
    targets = []
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)\s*:")
    for line in makefile.read_text(encoding="utf-8").splitlines():
        if line.startswith("."):
            continue
        match = pattern.match(line)
        if match:
            name = match.group(1)
            if name not in targets:
                targets.append(name)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="project_state/_generated")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    inventory = build_repo_inventory(root)
    symbol_index, import_graph = parse_symbols(root)
    make_targets = parse_make_targets(root / "Makefile")

    (outdir / "repo_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=False), encoding="utf-8"
    )
    (outdir / "symbol_index.json").write_text(
        json.dumps(
            {
                "generated_at": utc_now_iso(),
                "root": root.as_posix(),
                "files": symbol_index,
            },
            indent=2,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (outdir / "import_graph.json").write_text(
        json.dumps(import_graph, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    (outdir / "make_targets.txt").write_text(
        "\n".join(make_targets) + ("\n" if make_targets else ""),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
