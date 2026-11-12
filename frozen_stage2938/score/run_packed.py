"""Run a selected phase from an extracted, verified benchmark kit."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def below(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise ValueError(f"Invalid kit member: {name}")
    path = root.joinpath(*rel.parts).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Path escapes phase root: {name}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all")
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inventory = read(root / "kit_members.json")
    for name, expected in inventory["members"].items():
        if pin(below(root, name)) != expected:
            raise ValueError(f"Changed kit member: {name}")
    kit = read(root / "phase_roles.json")
    phase = kit["phase"]
    if phase != inventory["phase"] or (phase == "score") != (args.frozen_dir is not None):
        raise ValueError("Supply --frozen-dir when running the score kit")
    tasks = list(kit["tasks"]) if args.task == "all" else [args.task]
    if not set(tasks) <= set(kit["tasks"]):
        raise ValueError(f"Unknown task: {args.task}")
    frozen_root = args.frozen_dir.resolve() if args.frozen_dir else None
    roles = {}
    for task in tasks:
        roles[task] = {}
        for role, spec in kit["tasks"][task].items():
            base = root if spec["location"] == "kit" else frozen_root
            path = below(base, spec["path"])
            if pin(path) != spec["pin"]:
                raise ValueError(f"Changed input: {task}/{role}")
            roles[task][role] = str(path)
    with tempfile.TemporaryDirectory() as temp:
        mapping = Path(temp) / "input_map.json"
        mapping.write_text(json.dumps(roles), encoding="utf-8")
        subprocess.run([sys.executable, str(root / "integrated_benchmark.py"), phase,
                        "--benchmark", str(root / "benchmark.json"), "--package", str(root / "package.json"),
                        "--input-map", str(mapping), "--task", args.task, "--output-dir", str(args.output_dir.resolve())],
                       check=True)


if __name__ == "__main__":
    main()
