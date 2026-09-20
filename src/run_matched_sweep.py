"""Run the targeted matched composition-presence experiment."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(task, args):
    model, exercise, target, seed = task
    out = Path(args.outdir) / "runs" / f"{model}_{exercise}_{target}_s{seed}.json"
    if out.exists():
        return task, "cached"
    command = [sys.executable, "src/matched_composition.py", "--data", args.data,
               "--exercise", exercise, "--target", target, "--seed", str(seed),
               "--model", model, "--epochs", str(args.epochs), "--out", str(out)]
    if args.manifest:
        command.extend(["--manifest", args.manifest])
    if args.with_rate:
        command.append("--with-rate")
    if args.map_control != "anatomy":
        command.extend(["--map-control", args.map_control])
    env = os.environ.copy(); env["PYTHONPATH"] = "src"
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
    if completed.returncode:
        raise RuntimeError(f"{task}: {completed.stderr[-2000:]}")
    return task, "done"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    ap.add_argument("--protocol", default="configs/final_protocol.json")
    ap.add_argument("--outdir", default="results/matched_composition_v2")
    ap.add_argument("--manifest")
    ap.add_argument("--models", default="tcn"); ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=55); ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--with-rate", action="store_true")
    ap.add_argument("--map-control", default="anatomy", choices=["anatomy", "random", "permuted"])
    args = ap.parse_args(); (Path(args.outdir) / "runs").mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(args.protocol).read_text()); tasks = []
    for model in args.models.split(","):
        for exercise, targets in protocol["default_targets"].items():
            for target in targets:
                for seed in protocol["seeds"]:
                    tasks.append((model, exercise, target, seed))
    print(f"Matched composition sweep: {len(tasks)} paired runs", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, task, args) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
