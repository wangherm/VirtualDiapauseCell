"""Run the existing seed prediction CLI with the shared phase inputs."""
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
    paths = {k: Path(v) for k, v in json.loads(a.input_map.read_bytes()).items()}
    root = a.config.parent
    for key in ("contract", "binding", "native_predictor"):
        item = cfg[key]
        if pin(root / item["path"]) != item["pin"]:
            raise ValueError("Changed seed contract, binding or predictor")
    with tempfile.TemporaryDirectory() as work:
        req = Path(work)
        for g in cfg["groups"]:
            shutil.copyfile(paths[g], req / f"request_{g}.npz")
        subprocess.run([sys.executable, str(root / cfg["native_predictor"]["path"]),
                        "--contract", str(root / cfg["contract"]["path"]), "--binding", str(root / cfg["binding"]["path"]),
                        "--package", str(paths["archive"]), "--split", str(paths["split"]), "--request-dir", str(req),
                        "--output-dir", str(a.output_dir)], check=True)
    runs = json.loads((a.output_dir / "prediction_manifest.json").read_bytes())
    outputs = {x["prediction"]["path"]: pin(a.output_dir / x["prediction"]["path"]) for x in runs["runs"]}
    if outputs != cfg["reference_prediction_pins"] or runs["fit_calls"] or runs["score_calls"] or runs["outcomes_required"]:
        raise ValueError("Seed retention parity or outcome exclusion failed")
    (a.output_dir / "adapter_receipt.json").write_text(json.dumps({"status": "completed", "phase": "predict", "task": "seed_ar",
        "native_package": runs["package_version"], "native_benchmark": runs["benchmark_version"],
        "reference_predictions_exact": True, "fit_calls": 0, "score_calls": 0,
        "prediction_manifest": pin(a.output_dir / "prediction_manifest.json"), "scientific_approval": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
