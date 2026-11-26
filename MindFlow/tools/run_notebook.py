"""Run one MindFlow notebook on real data outside Colab (HPC node, lab workstation), without a browser.

    python MindFlow/tools/run_notebook.py <home> <notebook> [--local DIR] [--set NAME=VALUE ...] [--out DIR]

<home> plays the role of the Drive folder `Demo/MindFlow`: it holds `alldata_bundle_v4.zip` (and any
`*_panel_v4.zip`) and the `runs/` folder with the earlier steps' runs; this step's run is written there too.
The repository's `MindFlow/code` is copied to <home>/code, as on Drive, so the run uses the code of this checkout.

  --local DIR        where the bundle is unpacked (about 5 GB); default <home>/_local
  --set NAME=VALUE   replace a one-line assignment of the notebook, e.g. --set QUICK=True
                     --set 'RUN_ID="pruning_20261126_050000"' --set WORKERS=8; it must match exactly one line
  --out DIR          where the executed notebook is saved; default <home>/executed

The notebook's printed output is echoed while it runs, and the executed notebook is saved even when a cell fails.
Pick the GPU with CUDA_VISIBLE_DEVICES. Do the download of the bundle and of the runs beforehand, not on a GPU node.
"""
import argparse
import ast
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

MF = Path(__file__).resolve().parents[1]


def override(nb, name, value):
    """Replace the single top-level one-line assignment `name = ...` in the notebook's code cells."""
    pattern = re.compile(rf"^{re.escape(name)} = .*$", re.MULTILINE)
    hits = [(c, m) for c in nb.cells if c.cell_type == "code" for m in pattern.finditer(c.source)]
    if len(hits) != 1:
        raise SystemExit(f"--set {name}: {len(hits)} lines start with `{name} = ` (need exactly 1)")
    cell, m = hits[0]
    source = cell.source[:m.start()] + f"{name} = {value}" + cell.source[m.end():]
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SystemExit(f"--set {name}={value}: the cell no longer parses ({e.msg}); a value that spans several "
                         f"lines has to be edited in the notebook") from None
    cell.source = source
    print(f"  {m.group(0).split('#')[0].rstrip()}  ->  {name} = {value}")


def sync_code(home):
    """<home>/code = this checkout's MindFlow/code. An existing folder is replaced only if it is a MindFlow code copy."""
    target = home / "code"
    if target.exists():
        if not (target / "mindflow.py").is_file():
            raise SystemExit(f"{target} exists and is not a MindFlow code folder; not touching it")
        shutil.rmtree(target)
    shutil.copytree(MF / "code", target, ignore=shutil.ignore_patterns("__pycache__"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("home", type=Path)
    p.add_argument("notebook", help="name in MindFlow/, e.g. Step7b_Pruning")
    p.add_argument("--local", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    home = a.home.resolve()
    if not home.is_dir():
        raise SystemExit(f"{home} is not a folder")
    name = a.notebook.removesuffix(".ipynb")
    path = MF / f"{name}.ipynb"
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    local = (a.local or home / "_local").resolve()
    out = (a.out or home / "executed").resolve()
    local.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    class Live(NotebookClient):
        """Echo what the notebook prints, as it prints it."""

        def output(self, outs, msg, display_id, cell_index):
            content = msg["content"]
            if msg["msg_type"] == "stream":
                (sys.stderr if content["name"] == "stderr" else sys.stdout).write(content["text"])
                sys.stdout.flush()
            elif msg["msg_type"] in ("display_data", "execute_result") and "text/plain" in content["data"]:
                print(content["data"]["text/plain"], flush=True)
            return super().output(outs, msg, display_id, cell_index)

    nb = nbformat.read(path, as_version=4)
    print(f"{name} | home {home} | local {local}")
    for item in a.set:
        key, sep, value = item.partition("=")
        if not sep or not key.isidentifier():
            raise SystemExit(f"--set {item}: expected NAME=VALUE")
        override(nb, key, value)
    sync_code(home)
    os.environ.pop("MINDFLOW_LOCAL_TEST", None)            # a real run: never the synthetic-data switch
    os.environ.update(MINDFLOW_HERE=str(home), MINDFLOW_LOCAL=str(local), MPLBACKEND="Agg")
    saved = out / f"{name}__{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.ipynb"
    failed = None
    try:
        Live(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(out)}}).execute()
    except CellExecutionError as e:
        failed = e
    finally:
        nbformat.write(nb, saved)
    print(f"\nexecuted notebook -> {saved}")
    if failed is not None:
        raise SystemExit(f"FAILED: {failed.ename}: {failed.evalue}")
    print("OK")


if __name__ == "__main__":
    main()
