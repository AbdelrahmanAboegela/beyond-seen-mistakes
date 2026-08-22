"""Build a wheel, install it outside the checkout, and import every module.

The unit suite puts ``src`` on ``sys.path`` (see ``[tool.pytest.ini_options]``),
so it exercises the source tree and cannot see packaging faults.  Three modules
once imported their siblings as top-level names, which works from a checkout and
fails from an installed wheel.

Each module is imported in a *fresh* interpreter, run from a directory outside
the repository.  Importing them all in one process hides the fault: several
modules insert their own directory into ``sys.path``, so whichever runs first
repairs imports for every module after it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "beyond_seen_mistakes"


def modules() -> list[str]:
    return sorted(p.stem for p in (ROOT / "src").glob("*.py") if p.stem != "__init__")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="Keep the temporary build directory.")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="wheelcheck-"))
    wheel_dir, site = tmp / "wheel", tmp / "site"
    print(f"building wheel in {wheel_dir}")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "-w", str(wheel_dir), "-q", str(ROOT)], check=True)
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected exactly one wheel, found {wheels}", file=sys.stderr)
        return 1
    print(f"installing {wheels[0].name} into {site}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--target", str(site), str(wheels[0])], check=True)

    failures = []
    for name in modules():
        # cwd is outside the checkout so the source tree cannot satisfy the import.
        proc = subprocess.run(
            [sys.executable, "-c",
             f"import warnings; warnings.simplefilter('ignore'); import {PACKAGE}.{name}"],
            cwd=str(tmp), env={**dict(__import__("os").environ), "PYTHONPATH": str(site)},
            capture_output=True, text=True)
        if proc.returncode:
            last = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            failures.append((name, last))
            print(f"  FAIL {name}: {last}")
        else:
            print(f"  ok   {name}")

    if not args.keep:
        __import__("shutil").rmtree(tmp, ignore_errors=True)
    if failures:
        print(f"\n{len(failures)} module(s) fail to import from an installed wheel", file=sys.stderr)
        return 1
    print(f"\nall {len(modules())} modules import cleanly from the installed wheel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
