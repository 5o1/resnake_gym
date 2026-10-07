"""Dependency-boundary checks for the refactored training modules."""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "resnake_gym"


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("resnake_gym", *parts))


def _internal_dependencies(path: Path, modules: set[str]) -> set[str]:
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    dependencies = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            candidates = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = "." * node.level + (node.module or "")
            candidate = (
                importlib.util.resolve_name(imported, package)
                if node.level
                else imported
            )
            candidates = (candidate,)
        else:
            continue
        dependencies.update(
            candidate for candidate in candidates if candidate in modules
        )
    dependencies.discard(module)
    return dependencies


def _modules_loaded_by(module: str) -> set[str]:
    code = (
        "import json, sys; "
        f"import {module}; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.startswith('resnake_gym'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(result.stdout))


def test_vtrace_facade_loads_the_shared_runtime():
    loaded = _modules_loaded_by("resnake_gym.gamepad_vtrace")

    assert "resnake_gym.gamepad_runtime" in loaded


def test_checkpoint_loader_depends_on_the_vtrace_contract_not_facade():
    loaded = _modules_loaded_by("resnake_gym.policy_checkpoint")

    assert "resnake_gym.gamepad_vtrace" not in loaded
    assert "resnake_gym.gamepad_vtrace_contract" in loaded


def test_internal_modules_do_not_import_through_the_root_package_facade():
    offenders = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path == PACKAGE_ROOT / "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "resnake_gym":
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
            if isinstance(node, ast.Import) and any(
                alias.name == "resnake_gym" for alias in node.names
            ):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")

    assert offenders == []


def test_internal_dependency_graph_is_acyclic():
    paths = list(PACKAGE_ROOT.rglob("*.py"))
    modules = {_module_name(path) for path in paths}
    graph = {
        _module_name(path): _internal_dependencies(path, modules) for path in paths
    }
    visiting = set()
    visited = set()
    trail = []

    def visit(module):
        if module in visited:
            return
        if module in visiting:
            start = trail.index(module)
            cycle = trail[start:] + [module]
            raise AssertionError("dependency cycle: " + " -> ".join(cycle))
        visiting.add(module)
        trail.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        trail.pop()
        visiting.remove(module)
        visited.add(module)

    for module in sorted(modules):
        visit(module)
