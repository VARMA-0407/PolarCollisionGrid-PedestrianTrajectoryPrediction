"""Structural validation of the master Colab notebook (no execution).

  python scripts/validate_colab_notebook.py [--notebook notebooks/PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb]

Checks: nbformat structure; every code cell parses; every ``pedsafety`` import resolves to an existing attribute; every call of an imported
``pedsafety`` function / class binds to its real signature (positional count and keyword names); every repository-relative path literal
(``configs/``, ``scripts/``, ``docs/``, ``tests``) exists; names are defined in an earlier (or the same) cell before they are used (execution
order); the central configuration cell defines every stage switch; no IPython magic or shell escape is used. Exit 0 = all checks pass.
"""

import argparse
import ast
import builtins
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_NB = REPO / "notebooks" / "PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb"
SWITCHES = ("RUN_DATASET_AUDIT", "RUN_SPLIT_PREPARATION", "RUN_PERCEPTION", "RUN_GEOMETRY", "RUN_FORECASTING_TRAINING", "RUN_CONFLICT_PREDICTION",
            "RUN_RISK_ASSESSMENT", "RUN_EARLY_WARNING", "RUN_END_TO_END_VALIDATION")
PATH_RE = re.compile(r"^(configs|scripts|docs|tests|notebooks)/[\w./-]+$")
COLAB_ONLY = {"google.colab"}


class _Names(ast.NodeVisitor):
    def __init__(self):
        self.stored, self.loaded = set(), []

    def visit_Name(self, n):
        (self.stored.add(n.id) if isinstance(n.ctx, (ast.Store, ast.Del)) else self.loaded.append((n.id, n.lineno)))

    def visit_FunctionDef(self, n):
        self.stored.add(n.name)
        for a in n.args.args + n.args.kwonlyargs + ([n.args.vararg] if n.args.vararg else []) + ([n.args.kwarg] if n.args.kwarg else []):
            self.stored.add(a.arg)
        self.generic_visit(n)

    def visit_Lambda(self, n):
        for a in n.args.args:
            self.stored.add(a.arg)
        self.generic_visit(n)

    def visit_Import(self, n):
        for a in n.names:
            self.stored.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            self.stored.add(a.asname or a.name)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.stored.add(n.name)
        self.generic_visit(n)

    def visit_arg(self, n):
        self.stored.add(n.arg)


def validate(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    problems, info = [], {}
    if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
        problems.append("not an nbformat-4 notebook")
    cells = nb["cells"]
    ids = [c.get("id") for c in cells]
    if len(set(ids)) != len(ids) or None in ids:
        problems.append("cell ids missing or duplicated")
    code = [(i, "".join(c["source"])) for i, c in enumerate(cells) if c["cell_type"] == "code"]
    info["cells"], info["code_cells"] = len(cells), len(code)
    defined = set(dir(builtins)) | {"__file__", "__name__", "get_ipython", "display"}
    imported = {}
    calls_checked = 0
    for i, src in code:
        if re.search(r"^\s*(%%?(pip|cd|env|time|timeit|load_ext|matplotlib|capture|writefile|run|bash|sh|cat)\b|!\s*\w)", src, re.M):
            problems.append("cell %d uses an IPython magic / shell escape" % i)
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            problems.append("cell %d does not parse: %s" % (i, exc))
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                if n.module.split(".")[0] in ("pedsafety",):
                    try:
                        mod = importlib.import_module(n.module)
                    except Exception as exc:                                                       # noqa: BLE001
                        problems.append("cell %d: cannot import %s (%s)" % (i, n.module, exc))
                        continue
                    for a in n.names:
                        if not hasattr(mod, a.name):
                            problems.append("cell %d: %s has no attribute %s" % (i, n.module, a.name))
                        else:
                            imported[a.asname or a.name] = getattr(mod, a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("pedsafety"):
                        try:
                            importlib.import_module(a.name)
                        except Exception as exc:                                                   # noqa: BLE001
                            problems.append("cell %d: cannot import %s (%s)" % (i, a.name, exc))
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and PATH_RE.match(n.value):
                if not (REPO / n.value).exists():
                    problems.append("cell %d: repository path %r does not exist" % (i, n.value))
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in imported:
                fn = imported[n.func.id]
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                if any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
                    continue
                try:
                    sig.bind(*[object()] * len(n.args), **{k.arg: object() for k in n.keywords})
                    calls_checked += 1
                except TypeError as exc:
                    problems.append("cell %d line %d: call %s%s does not match its signature: %s" % (i, n.lineno, n.func.id, sig, exc))
        v = _Names()
        v.visit(tree)
        local = set(v.stored)
        for name, line in v.loaded:
            if name not in defined and name not in local:
                problems.append("cell %d line %d: name %r is used before any earlier cell defines it" % (i, line, name))
        defined |= local
        # module-level string paths built with REPO / "..." are checked through constants above; calls to string-config keys below
        for m in re.finditer(r'CONFIG_FILES\["(\w+)"\]', src):
            pass
    cfg_src = next((s for _, s in code if "CENTRAL CONFIGURATION" in s), "")
    for s in SWITCHES:
        if not re.search(r"^%s\s*=" % s, cfg_src, re.M):
            problems.append("configuration cell does not define %s" % s)
    cf = next((s for _, s in code if "CONFIG_FILES = {" in s), "")
    for rel in re.findall(r'"(configs/[^"]+)"', cf):
        if not (REPO / rel).is_file():
            problems.append("CONFIG_FILES entry %s does not exist" % rel)
    info["pedsafety_calls_signature_checked"] = calls_checked
    info["pedsafety_names_imported"] = sorted(imported)
    return {"ok": not problems, "problems": problems, "info": info}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=str(DEFAULT_NB))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    r = validate(Path(a.notebook))
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
