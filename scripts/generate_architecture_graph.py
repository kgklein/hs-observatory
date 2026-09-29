#!/usr/bin/env python3
"""Generate module dependency DOT from source text; optionally render with dot."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hs_observatory"
GROUPS = (
    "orchestration", "analysis", "analysis.derived", "fields", "errors",
    "io", "spacecraft", "utils",
)


@dataclass(frozen=True)
class Module:
    path: Path
    tree: ast.Module
    package: bool


@dataclass(frozen=True)
class Import:
    source: str
    target: str
    local: bool
    symbol: str | None = None


def read_modules(package_root: Path) -> dict[str, Module]:
    modules = {}
    for path in sorted(package_root.rglob("*.py")):
        parts = path.relative_to(package_root.parent).with_suffix("").parts
        package = parts[-1] == "__init__"
        name = ".".join(parts[:-1] if package else parts)
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except SyntaxError as exc:
            raise ValueError(
                f"Cannot parse {path}:{exc.lineno}:{exc.offset}: {exc.msg}"
            ) from exc
        modules[name] = Module(path, tree, package)
    return modules


def extract_imports(modules: dict[str, Module]) -> tuple[list[Import], list[str]]:
    imports = []
    warnings = []
    for source, module in sorted(modules.items()):
        package = source if module.package else source.rpartition(".")[0]
        typing_names = {"typing"}
        checking_names = {"TYPE_CHECKING"}
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Import):
                typing_names.update(a.asname or a.name for a in node.names if a.name == "typing")
            elif isinstance(node, ast.ImportFrom) and node.module == "typing":
                checking_names.update(
                    a.asname or a.name for a in node.names if a.name == "TYPE_CHECKING"
                )

        def runtime_condition(node: ast.expr) -> bool | None:
            if isinstance(node, ast.Name) and node.id in checking_names:
                return False
            if (isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
                    and isinstance(node.value, ast.Name) and node.value.id in typing_names):
                return False
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                value = runtime_condition(node.operand)
                return None if value is None else not value
            if isinstance(node, ast.BoolOp):
                values = [runtime_condition(value) for value in node.values]
                if isinstance(node.op, ast.And):
                    return False if False in values else (True if all(v is True for v in values) else None)
                return True if True in values else (False if all(v is False for v in values) else None)
            return None

        def visit(node: ast.AST, local: bool = False) -> None:
            local = local or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(node, ast.If):
                condition = runtime_condition(node.test)
                for child in ([] if condition is False else node.body):
                    visit(child, local)
                for child in ([] if condition is True else node.orelse):
                    visit(child, local)
                return
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        imports.append(Import(source, alias.name, local))
                    elif alias.name.startswith(PACKAGE + "."):
                        warnings.append(f"{module.path}:{node.lineno}: unknown module {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    if node.level > len(parts):
                        raise ValueError(f"{module.path}:{node.lineno}: relative import beyond package")
                    base = ".".join(parts[:len(parts) - node.level + 1])
                    if node.module:
                        base += "." + node.module
                else:
                    base = node.module or ""
                if base == PACKAGE or base.startswith(PACKAGE + "."):
                    for alias in node.names:
                        candidate = base + "." + alias.name
                        target = candidate if candidate in modules else base
                        if target in modules:
                            imports.append(Import(source, target, local, alias.name))
                            if alias.name == "*":
                                warnings.append(
                                    f"{module.path}:{node.lineno}: wildcard import retained at module level"
                                )
                        else:
                            warnings.append(f"{module.path}:{node.lineno}: unknown module {target}")
            for child in ast.iter_child_nodes(node):
                visit(child, local)

        visit(module.tree)
    return imports, sorted(set(warnings))


def build_graph(modules: dict[str, Module], imports: list[Import], *,
                full: bool = False, show_local: bool = True):
    def excluded(name: str) -> str | None:
        if full:
            return None
        if modules[name].package:
            return "package initializer/re-export"
        if name.rpartition(".")[2].startswith("cli_"):
            return "thin CLI wrapper"
        return None

    nodes = sorted(name for name in modules if not excluded(name))
    edges = {}
    suppressed = set()
    for item in imports:
        reason = None
        if not full:
            if item.target == PACKAGE and item.symbol == "__version__":
                reason = "version metadata"
            elif (item.source.startswith(PACKAGE + ".analysis.")
                  and not modules[item.source].package
                  and item.target == PACKAGE + ".analysis.registry"
                  and item.symbol == "register_analysis"):
                reason = "analysis registration back-edge"
            else:
                reason = excluded(item.source) or excluded(item.target)
        if reason is None and item.local and not show_local:
            reason = "local imports hidden"
        pair = (item.source, item.target)
        if reason:
            suppressed.add((*pair, reason))
        else:
            # False (module level) wins over True (function local).
            edges[pair] = edges.get(pair, True) and item.local
    # Report only dependencies actually removed, not filtered duplicate occurrences.
    suppressed = sorted(row for row in suppressed if row[:2] not in edges)
    return nodes, dict(sorted(edges.items())), suppressed


def short_name(name: str) -> str:
    return name.removeprefix(PACKAGE + ".")


def make_dot(modules: dict[str, Module], nodes: list[str], edges: dict) -> str:
    quote = json.dumps
    groups: dict[str, list[str]] = {group: [] for group in GROUPS}
    for name in nodes:
        short = short_name(name)
        group = short.rpartition(".")[0]
        if modules[name].package:
            group = short if name != PACKAGE else "orchestration"
        if not group:
            group = "orchestration"
        groups.setdefault(group, []).append(name)
    lines = [
        "// Generated by scripts/generate_architecture_graph.py; do not edit.",
        "digraph module_dependencies {",
        '  graph [rankdir=LR, bgcolor="white", fontname="Helvetica", fontsize=12,',
        '         nodesep=0.24, ranksep=0.65, pad=0.2, compound=true,',
        '         label="hs-observatory module dependencies\\nSolid: module-level import   Dashed: local/lazy import", labelloc=t];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f7f8fa",',
        '        color="#667085", fontcolor="#182230", fontname="Helvetica", fontsize=10, margin="0.09,0.06"];',
        '  edge [color="#667085", arrowsize=0.65, penwidth=0.9];',
    ]
    for index, (group, members) in enumerate(groups.items()):
        if not members:
            continue
        lines.extend([
            f"  subgraph cluster_{index} {{",
            f"    label={quote(group)};",
            '    color="#d0d5dd"; style="rounded"; fontcolor="#344054";',
        ])
        for name in sorted(members):
            label = short_name(name)
            if modules[name].package:
                label = (label + ".__init__") if name != PACKAGE else "__init__"
            lines.append(f"    {quote(name)} [label={quote(label)}];")
        lines.append("  }")
    for (source, target), local in sorted(edges.items()):
        style = "dashed" if local else "solid"
        lines.append(f"  {quote(source)} -> {quote(target)} [style={style}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


# This is an intentionally curated view, not a quotient of every import edge.
# Each displayed edge must have at least one direct source-import witness.
OVERVIEW_LABELS = {
    "cli": "CLI", "config": "Configuration", "generation": "Generation",
    "spacecraft": "Spacecraft", "fields": "Fields", "errors": "Error models",
    "io": "I/O", "dispatch": "Analysis dispatch",
    "intermittency": "Elsasser / intermittency", "gradients": "Magnetic gradients",
    "reconstruction": "Reconstruction", "diagnostics": "Other diagnostics",
    "timing": "Timing calibration", "shock": "Shock timing",
    "visualization": "Visualization / output",
}
OVERVIEW_RELATIONSHIPS = (
    ("cli", "generation"), ("cli", "dispatch"),
    ("generation", "config"), ("generation", "spacecraft"),
    ("generation", "fields"), ("generation", "errors"),
    ("generation", "io"), ("generation", "visualization"),
    ("fields", "io"), ("dispatch", "config"),
    ("dispatch", "intermittency"), ("dispatch", "gradients"),
    ("dispatch", "reconstruction"), ("dispatch", "diagnostics"),
    ("dispatch", "timing"), ("dispatch", "shock"),
    ("timing", "gradients"), ("shock", "visualization"),
    ("intermittency", "visualization"), ("gradients", "visualization"),
    ("reconstruction", "visualization"), ("diagnostics", "visualization"),
)


def overview_group(name: str) -> str | None:
    """Map modules to responsibilities; low-level utilities are intentionally omitted."""
    name = short_name(name)
    if name in {"cli", "config", "visualization"}:
        return name
    if name == "generate":
        return "generation"
    if name in {"analyze", "analysis.registry", "analysis.base"}:
        return "dispatch"
    for prefix in ("spacecraft", "fields", "errors", "io"):
        if name.startswith(prefix + "."):
            return prefix
    if name.startswith(("analysis.elsasser", "analysis.increment_pdf",
                        "analysis.structure_function", "analysis.derived.")):
        return "intermittency"
    if name.startswith("analysis.magnetic_gradient"):
        return "gradients"
    if name.startswith("analysis.timing_"):
        return "timing"
    if name == "analysis.shock_timing":
        return "shock"
    if name in {"analysis.rbf_reconstruction", "analysis.timesync"}:
        return "reconstruction"
    if name in {"analysis.diagnostics", "analysis.static_decorrelation", "analysis.timeseries_fft"}:
        return "diagnostics"
    return None


def build_overview(nodes: list[str], edges: dict):
    members: dict[str, list[str]] = {}
    for name in sorted(nodes):
        group = overview_group(name)
        if group:
            members.setdefault(group, []).append(name)
    witnesses: dict[tuple[str, str], list[str]] = {}
    for source, target in sorted(edges):
        pair = (overview_group(source), overview_group(target))
        if pair in OVERVIEW_RELATIONSHIPS:
            witnesses.setdefault(pair, []).append(f"{short_name(source)} -> {short_name(target)}")
    return dict(sorted(members.items())), dict(sorted(witnesses.items()))


def make_overview_dot(members: dict, witnesses: dict) -> str:
    quote = json.dumps
    lines = [
        "// Generated curated overview; arrows are selected source-backed dependencies.",
        "digraph package_overview {",
        '  graph [rankdir=LR, bgcolor="white", fontname="Helvetica", fontsize=12,',
        '         nodesep=0.35, ranksep=0.7, pad=0.2,',
        '         label="HS Observatory: architectural areas\\nSelected dependencies, not execution order", labelloc=t];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f7f8fa",',
        '        color="#667085", fontname="Helvetica", fontsize=11, margin="0.14,0.10"];',
        '  edge [color="#667085", arrowsize=0.7];',
    ]
    clusters = {
        "Entry points": ("cli", "generation", "dispatch"),
        "Generation and support": ("config", "spacecraft", "fields", "errors", "io"),
        "Scientific analyses": ("intermittency", "gradients", "reconstruction", "diagnostics",
                                "timing", "shock"),
        "Output": ("visualization",),
    }
    for index, (label, groups) in enumerate(clusters.items()):
        lines.extend([f"  subgraph cluster_{index} {{", f"    label={quote(label)};",
                      '    color="#d0d5dd"; style="rounded";'])
        for group in sorted(groups):
            if group in members:
                tooltip = ", ".join(short_name(name) for name in members[group])
                lines.append(f"    {quote(group)} [label={quote(OVERVIEW_LABELS[group])}, tooltip={quote(tooltip)}];")
        lines.append("  }")
    for pair, evidence in sorted(witnesses.items()):
        lines.append(f"  {quote(pair[0])} -> {quote(pair[1])} [tooltip={quote('; '.join(evidence))}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_svg(dot_path: Path, svg_path: Path, no_render: bool) -> None:
    if no_render:
        print("SVG rendering skipped (--no-render).")
    elif (dot := shutil.which("dot")) is None:
        print("SVG rendering skipped: Graphviz 'dot' executable not found; DOT is ready.")
    else:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [dot, "-Tsvg", str(dot_path), "-o", str(svg_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise ValueError(f"Graphviz rendering failed: {result.stderr.strip()}")
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        print(f"Rendered {svg_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot", type=Path, default=ROOT / "docs/architecture/module-dependencies.dot")
    parser.add_argument("--svg", type=Path, default=ROOT / "docs/architecture/module-dependencies.svg")
    parser.add_argument("--no-render", action="store_true", help="write DOT only")
    parser.add_argument("--full", action="store_true", help="include wrappers, initializers and registration edges")
    parser.add_argument("--detailed-only", action="store_true", help="skip the curated overview")
    local = parser.add_mutually_exclusive_group()
    local.add_argument("--show-local-imports", dest="show_local", action="store_true")
    local.add_argument("--hide-local-imports", dest="show_local", action="store_false")
    parser.set_defaults(show_local=True)
    args = parser.parse_args(argv)
    overview_dot = args.dot.with_name("package-overview.dot")
    overview_svg = args.svg.with_name("package-overview.svg")
    outputs = [args.dot, args.svg]
    if not args.detailed_only:
        outputs.extend([overview_dot, overview_svg])
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("detailed and overview output paths must be distinct")
    try:
        modules = read_modules(ROOT / "src" / PACKAGE)
        if not modules:
            raise ValueError("No Python source modules found")
        imports, warnings = extract_imports(modules)
        nodes, edges, suppressed = build_graph(
            modules, imports, full=args.full, show_local=args.show_local,
        )
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        args.dot.write_text(make_dot(modules, nodes, edges), encoding="utf-8")
        print(f"Wrote {args.dot}")
        print(f"Nodes: {len(nodes)}; module-level edges: {sum(not v for v in edges.values())}; "
              f"local/lazy edges: {sum(edges.values())}")
        print(f"Suppressed dependencies: {len(suppressed)}")
        for reason, count in sorted(Counter(row[2] for row in suppressed).items()):
            print(f"  {reason}: {count}")
        for source, target, reason in suppressed:
            print(f"  {short_name(source)} -> {short_name(target)} ({reason})")
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        render_svg(args.dot, args.svg, args.no_render)
        if not args.detailed_only:
            # --full is detail-only: initializer and registry noise never enter this view.
            clean_nodes, clean_edges, _ = build_graph(modules, imports, show_local=args.show_local)
            members, witnesses = build_overview(clean_nodes, clean_edges)
            overview_dot.write_text(make_overview_dot(members, witnesses), encoding="utf-8")
            print(f"Wrote {overview_dot}")
            print(f"Overview nodes: {len(members)}; selected dependencies: {len(witnesses)}")
            render_svg(overview_dot, overview_svg, args.no_render)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
