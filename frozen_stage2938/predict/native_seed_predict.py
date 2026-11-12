"""Freeze predictions from an explicit package before any outcome is available."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np


def check(ok, message):
    if not ok:
        raise ValueError(message)


def pin(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def path_pin(path):
    return pin(path.read_bytes())


def expected(record):
    return {key: record[key] for key in ("bytes", "sha256")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("contract", "binding", "package", "split", "request-dir", "output-dir"):
        p.add_argument("--" + flag, required=True, type=Path)
    a = p.parse_args()
    contract = json.loads(a.contract.read_bytes())
    binding = json.loads(a.binding.read_bytes())
    check(path_pin(a.contract) == binding["contract"] and path_pin(a.package) == binding["archive"], "Package/benchmark pin differs")
    check(path_pin(a.split) == contract["split"], "Split pin differs")
    split = json.loads(a.split.read_bytes())
    folds = split["folds"]
    check(split["version"] == contract["benchmark_version"]
          and split["split_protocol_sha256"] == contract["protocol_sha256"]
          and split["feature_axis_sha256"] == contract["axis_sha256"]
          and split["schema_sha256"] == contract["schema_sha256"]
          and [f["held_genotype"] for f in folds] == contract["groups"], "Invalid benchmark fold binding")
    paths = {}
    for fold in folds:
        g = fold["held_genotype"]
        source, request, outcome = (set(fold[k]) for k in ("source_gsm", "request_gsm", "outcome_gsm"))
        check(len(source) == 24 and len(request) == len(outcome) == 3
              and not ((source & request) or (source & outcome) or (request & outcome))
              and fold["request_order"][0] == "balanced_3FR", "Invalid primary roles")
        row = fold["outputs"]["request"]
        check(row["path"] == f"request_{g}.npz", "Invalid request filename")
        paths[g] = a.request_dir / row["path"]
        check(path_pin(paths[g]) == expected(row), "Request pin differs")
    check({x.name for x in a.request_dir.iterdir()} == {x.name for x in paths.values()}, "Predictor request directory has other files")
    check(set(binding["modes"]) == {"package_reference", "candidate"}, "Two explicit evaluation modes are required")
    a.output_dir.mkdir(exist_ok=True)
    check(not list(a.output_dir.iterdir()), "Use an empty output directory")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        package = work / "package"
        with zipfile.ZipFile(a.package) as archive:
            names = archive.namelist()
            manifest_data = archive.read("manifest.json")
            manifest = json.loads(manifest_data)
            check(pin(manifest_data) == binding["manifest"], "Manifest pin differs")
            check(len(names) == len(set(names)) and set(names) == set(manifest["files"]) | {"manifest.json"}, "Archive membership differs")
            check(manifest["package_version"] == binding["package_version"] and manifest["benchmark_version"] == contract["benchmark_version"], "Archive version differs")
            for name in names:
                rel = Path(name)
                check(not rel.is_absolute() and ".." not in rel.parts and "\\" not in name, "Invalid archive path")
                data = archive.read(name)
                if name in manifest["files"]:
                    check(pin(data) == manifest["files"][name], "Packaged bytes differ")
                target = package / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        schema_path = package / "schema.json"
        check(path_pin(schema_path)["sha256"] == contract["schema_sha256"], "Package schema differs")
        features = json.loads(schema_path.read_bytes())["features"]
        d = contract["feature_count"]
        check(len(features) == len(set(features)) == d
              and hashlib.sha256(("\n".join(features) + "\n").encode()).hexdigest() == contract["axis_sha256"], "Feature axis differs")
        runs = []
        for g, path in paths.items():
            with np.load(path, allow_pickle=False) as z:
                check(z.files == ["x"], "Unexpected request fields")
                x = z["x"][:1]
            check(x.shape == (1, d) and x.dtype.kind == "f" and np.isfinite(x).all(), "Invalid primary request")
            req = work / f"{g}.npz"
            np.savez_compressed(req, x=x, feature_ids=np.asarray(features))
            for role, mode in binding["modes"].items():
                out = a.output_dir / f"{g}_{role}.npz"
                command = [sys.executable, str(package / "consumer.py"), "--genotype", g, "--array-count", "3",
                           "--mode", mode, "--request", str(req), "--output", str(out.resolve())]
                environment = os.environ.copy()
                environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                receipt = json.loads(subprocess.check_output(command, cwd=package, env=environment, text=True, timeout=20))
                output = path_pin(out)
                check(receipt["package_version"] == binding["package_version"] and receipt["genotype"] == g
                      and receipt["mode"] == mode and receipt["fit_calls"] == 0 and receipt["shape"] == [1, d]
                      and receipt["output_sha256"] == output["sha256"]
                      and receipt["checkpoint_sha256"] == binding["checkpoints"][g], "Wrong inference receipt")
                with np.load(out, allow_pickle=False) as z:
                    check(set(z.files) == {"prediction", "feature_ids"} and z["feature_ids"].tolist() == features
                          and z["prediction"].shape == (1, d) and np.isfinite(z["prediction"]).all(), "Invalid output")
                runs.append({"role": role, "original_request": path_pin(path), "cli_request": path_pin(req),
                             "prediction": {"path": out.name, **output}, "consumer_receipt": receipt})
    report = {"benchmark_version": contract["benchmark_version"], "runner_version": contract["runner_version"],
              "contract": path_pin(a.contract), "binding": path_pin(a.binding), "split": path_pin(a.split),
              "archive": binding["archive"], "package_version": binding["package_version"], "runs": runs,
              "fit_calls": 0, "score_calls": 0, "outcomes_required": False, "scientific_approval": False}
    (a.output_dir / "prediction_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

