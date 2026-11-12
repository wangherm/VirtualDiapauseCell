"""Execute the chosen task routes of a pinned benchmark and package."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "score"))
    for name in ("benchmark", "package", "input-map", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", default="all")
    args = parser.parse_args()
    benchmark = read(args.benchmark)
    package = read(args.package)
    inputs = read(args.input_map)
    require(package["benchmark"] == pin(args.benchmark)
            and package["benchmark_version"] == benchmark["benchmark_version"], "Wrong benchmark or package binding")
    tasks = list(benchmark["tasks"]) if args.task == "all" else [args.task]
    require(set(tasks) <= set(benchmark["tasks"]), "Unknown task")
    require(set(tasks) == set(inputs), "Input map must cover exactly the selected tasks")
    routes = {}
    for task in tasks:
        route = package["routes"].get(task, {}).get(args.phase)
        require(route is not None, f"Unavailable package route for {task}")
        configured = benchmark["tasks"][task]["phases"][args.phase]
        require(route["contract"] == configured["contract"] and set(route["inputs"]) == set(configured["roles"])
                and set(route["inputs"]) == set(inputs[task]), f"Wrong {task} phase roles")
        root = args.package.parent
        adapter = root / route["adapter"]["file"]
        cfg = root / route["config"]["file"]
        require(pin(adapter) == route["adapter"]["pin"] and pin(cfg) == route["config"]["pin"],
                f"Changed adapter or config for {task}")
        paths = {role: Path(path).resolve() for role, path in inputs[task].items()}
        require(all(pin(path) == route["inputs"][role] for role, path in paths.items()), f"Changed {task} input bytes")
        routes[task] = (route, adapter, cfg, paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(not list(args.output_dir.iterdir()), "Use an empty output directory")
    runs = {}
    for task, (route, adapter, cfg, paths) in routes.items():
        output = args.output_dir / task
        output.mkdir()
        with tempfile.TemporaryDirectory() as work:
            packet = Path(work) / "input_map.json"
            write(packet, {role: str(path) for role, path in paths.items()})
            subprocess.run([sys.executable, str(adapter), "--config", str(cfg),
                            "--input-map", str(packet), "--output-dir", str(output)], check=True)
        receipt = read(output / route["receipt"])
        require(receipt["status"] == "completed" and receipt["phase"] == args.phase, "Adapter did not complete")
        runs[task] = {"adapter": route["adapter"]["pin"], "config": route["config"]["pin"],
                      "input_pins": route["inputs"], "receipt": pin(output / route["receipt"]),
                      "outputs": {str(p.relative_to(output).as_posix()): pin(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write(args.output_dir / "integrated_run.json", {"phase": args.phase, "benchmark_version": benchmark["benchmark_version"],
          "package_version": package["package_version"], "benchmark": pin(args.benchmark),
          "package": pin(args.package), "task_scope": tasks, "runs": runs, "scientific_approval": False})


if __name__ == "__main__":
    main()
