"""Source-only documentation tooling checks; no scientific package imports."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "architecture_graph_tool", ROOT / "scripts/generate_architecture_graph.py"
)
graph = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = graph
SPEC.loader.exec_module(graph)


class ArchitectureGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "src/hs_observatory"

    def source(self, name, text):
        path = self.package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def extract(self):
        modules = graph.read_modules(self.package)
        imports, warnings = graph.extract_imports(modules)
        self.assertEqual(warnings, [])
        return modules, imports

    def test_relative_submodules_symbols_local_methods_and_typing_guards(self):
        self.source("__init__.py", 'raise RuntimeError("never execute source")\n')
        for name in ("b", "c", "d", "e"):
            self.source(name + ".py", "class Thing: pass\n")
        self.source("analysis/__init__.py", "")
        self.source(
            "analysis/a.py",
            """
from .. import b
from ..b import Thing
from typing import TYPE_CHECKING as TC
import typing as t
if TC:
    from .. import c
else:
    from .. import d
if t.TYPE_CHECKING and enabled:
    from .. import c
if not TC:
    from .. import d
class Example:
    from .. import d
    async def run(self):
        from .. import b
        from .. import e
""",
        )
        modules, imports = self.extract()
        _, edges, _ = graph.build_graph(modules, imports)
        a = "hs_observatory.analysis.a"
        self.assertEqual(
            edges,
            {
                (a, "hs_observatory.b"): False,
                (a, "hs_observatory.d"): False,
                (a, "hs_observatory.e"): True,
            },
        )
        _, edges, _ = graph.build_graph(modules, imports, show_local=False)
        self.assertNotIn((a, "hs_observatory.e"), edges)
        self.assertIn((a, "hs_observatory.b"), edges)

    def test_filters_preserve_nonregistration_registry_dependency(self):
        self.source("__init__.py", '__version__ = "test"\n')
        self.source("cli.py", "")
        self.source("cli_example.py", "from . import cli\n")
        self.source("analysis/__init__.py", "from .registry import get_analysis\n")
        self.source("analysis/registry.py", "")
        self.source(
            "analysis/example.py",
            """
from .. import __version__
from .registry import register_analysis, get_analysis
""",
        )
        modules, imports = self.extract()
        nodes, edges, suppressed = graph.build_graph(modules, imports)
        self.assertNotIn("hs_observatory", nodes)
        self.assertNotIn("hs_observatory.cli_example", nodes)
        pair = ("hs_observatory.analysis.example", "hs_observatory.analysis.registry")
        self.assertIn(pair, edges)
        self.assertFalse(any(row[:2] == pair for row in suppressed))
        full_nodes, _, suppressed = graph.build_graph(modules, imports, full=True)
        self.assertEqual(set(full_nodes), set(modules))
        self.assertEqual(suppressed, [])

    def test_overview_edges_require_evidence_and_do_not_collapse_every_import(self):
        nodes = [
            "hs_observatory.cli",
            "hs_observatory.generate",
            "hs_observatory.fields.malara",
            "hs_observatory.analysis.magnetic_gradient",
        ]
        evidence = (nodes[0], nodes[1])
        secondary = (nodes[3], nodes[2])
        members, witnesses = graph.build_overview(nodes, {evidence: True, secondary: False})
        self.assertEqual(set(witnesses), {("cli", "generation")})
        self.assertIn("gradients", members)
        _, without_evidence = graph.build_overview(nodes, {secondary: False})
        self.assertEqual(without_evidence, {})
        self.assertEqual(
            graph.make_overview_dot(members, witnesses),
            graph.make_overview_dot(dict(reversed(list(members.items()))), witnesses),
        )

    def test_current_source_group_coverage_and_witnesses(self):
        modules = graph.read_modules(ROOT / "src/hs_observatory")
        imports, warnings = graph.extract_imports(modules)
        self.assertEqual(warnings, [])
        nodes, edges, _ = graph.build_graph(modules, imports)
        self.assertEqual(
            [name for name in nodes if graph.overview_group(name) is None],
            ["hs_observatory.utils.units"],
            "New modules need an explicit overview grouping decision",
        )
        members, witnesses = graph.build_overview(nodes, edges)
        self.assertEqual(set(members), set(graph.OVERVIEW_LABELS))
        self.assertTrue(all(witnesses.values()))
        self.assertTrue(set(witnesses).issubset(graph.OVERVIEW_RELATIONSHIPS))

    def test_cli_outputs_and_legacy_detailed_only(self):
        self.source("__init__.py", "")
        self.source("cli.py", "from .generate import generate_from_config\n")
        self.source("generate.py", "def generate_from_config(): pass\n")
        output = self.root / "graphs/detail.dot"
        args = ["--dot", str(output), "--svg", str(output.with_suffix(".svg")), "--no-render"]
        with patch.object(graph, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(graph.main(args + ["--detailed-only"]), 0)
            self.assertFalse(output.with_name("package-overview.dot").exists())
            original = output.read_bytes()
            self.assertEqual(graph.main(args), 0)
            self.assertEqual(output.read_bytes(), original)
            overview = output.with_name("package-overview.dot")
            self.assertTrue(overview.exists())
            overview_bytes = overview.read_bytes()
            self.assertEqual(graph.main(args + ["--full"]), 0)
            self.assertEqual(overview.read_bytes(), overview_bytes)
            self.assertFalse(output.with_suffix(".svg").exists())

    def test_missing_graphviz_still_writes_both_dot_files(self):
        self.source("cli.py", "")
        dot = self.root / "detail.dot"
        capture = io.StringIO()
        with (
            patch.object(graph, "ROOT", self.root),
            patch.object(graph.shutil, "which", return_value=None),
            patch.object(graph.subprocess, "run") as run,
            contextlib.redirect_stdout(capture),
        ):
            self.assertEqual(
                graph.main(["--dot", str(dot), "--svg", str(dot.with_suffix(".svg"))]), 0
            )
            run.assert_not_called()
        self.assertTrue(dot.exists())
        self.assertTrue(dot.with_name("package-overview.dot").exists())
        self.assertIn("executable not found", capture.getvalue())

    def test_parse_failure_identifies_source_and_line(self):
        self.source("broken.py", "def broken(:\n")
        with self.assertRaisesRegex(ValueError, r"broken\.py:1:"):
            graph.read_modules(self.package)


if __name__ == "__main__":
    unittest.main()
