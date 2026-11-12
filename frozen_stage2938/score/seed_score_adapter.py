"""Run native seed scoring from an isolated integrated score-phase route."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def pin(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "input-map", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_bytes())
    inputs = json.loads(a.input_map.read_bytes())
    root = a.config.parent
    for key, spec in cfg["native"].items():
        if pin(root / spec["path"]) != spec["pin"]:
            raise ValueError(f"Changed native seed scoring material {key}")
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        for name, path in inputs.items():
            dest = work / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        cmd = [sys.executable, str(root / cfg["native"]["scorer"]["path"])]
        for flag, key in (("contract", "contract"), ("score-inputs", "score_inputs"), ("binding", "binding")):
            cmd += ["--" + flag, str(root / cfg["native"][key]["path"])]
        for flag, name in (("split", "fold_manifest.json"), ("frozen-dir", "frozen"),
                           ("outcome-dir", "outcome"), ("control-dir", "control")):
            cmd += ["--" + flag, str(work / name)]
        subprocess.run([*cmd, "--output-dir", str(a.output_dir)], check=True)
    actual = {p.name: pin(p) for p in a.output_dir.iterdir() if p.is_file()}
    if actual != cfg["reference_score_pins"]:
        raise ValueError("Seed score outputs differ from exact retained reference")
    receipt = {"status": "completed", "phase": "score", "task": "seed_ar",
               "exact_reference_scores": True, "score_reports": actual, "fit_calls": 0, "prediction_calls": 0,
               "scientific_approval": False}
    (a.output_dir / "adapter_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
