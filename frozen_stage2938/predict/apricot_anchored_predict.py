"""Predict both apricot cuts from exact source-only checkpoints and request packets."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

import general_learning as driver


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "input-map", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    inputs = {role: Path(path) for role, path in read(args.input_map).items()}
    check(set(inputs) == set(config["inputs"]) and all(pin(p) == config["inputs"][k] for k, p in inputs.items()),
          "Wrong exact apricot predictor inputs")
    n = config["feature_count"]
    members = {"checkpoint.json", "recipe.json", "model.py", "model/parameters.npz", "model/state.json"}
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for fold in config["folds"]:
            key = fold["fold"]
            archive = inputs[key + "/checkpoint"]
            receipt = read(inputs[key + "/receipt"])
            query_rows = read(inputs[key + "/query_rows"])
            schema = read(inputs[fold["schema_role"]])
            genes = schema["original_gene_IDs"]
            axis = hashlib.sha256((json.dumps(genes, ensure_ascii=False, separators=(",", ":")) + "\n").encode()).hexdigest()
            check(axis == config["axis_sha256"] == schema["ordered_axis_sha256"] and len(genes) == len(set(genes)) == n,
                  "Wrong apricot expression axis")
            check(query_rows == fold["query_rows"] and [r["genotype"] for r in query_rows] == fold["held_genotypes"]
                  and all(r["pair_id"] is None and not r["individual_pairing_claim"] and not r["independent_function_target"]
                          for r in query_rows), "Wrong exact apricot query roles")
            check(receipt["status"] == "completed" and receipt["fold"] == key and receipt["cut"] == fold["cut"]
                  and receipt["archive"] == pin(archive) and receipt["parent"] == fold["parent"]
                  and receipt["source_genotypes"] == fold["source_genotypes"]
                  and receipt["held_genotypes"] == fold["held_genotypes"]
                  and receipt["candidate_protocol"] == config["candidate_protocol"]
                  and not receipt["held_request_or_outcomes_mounted"]
                  and receipt["training"]["status"] == "completed" and receipt["training"]["completed_steps"] == 1,
                  "Wrong fit or parent lineage")
            checkpoint = Path(temp) / key
            with zipfile.ZipFile(archive) as z:
                check(len(z.namelist()) == len(members) and set(z.namelist()) == members, "Unexpected archive members")
                for name in sorted(members):
                    p = checkpoint / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(z.read(name))
            m = driver.verify(checkpoint)
            recipe = read(checkpoint / "recipe.json")
            bind = recipe["data_contract"]
            ck_sha = pin(checkpoint / "checkpoint.json")["sha256"]
            check(ck_sha == receipt["training"]["checkpoint_sha256"] != fold["parent"]["pin"]["sha256"]
                  and m["binding"]["data_sha256"] == bind["source"]["sha256"] == fold["source_pin"]["sha256"]
                  and m["binding"]["driver_sha256"] == config["driver"]["sha256"]
                  and m["binding"]["plugin_sha256"] == pin(checkpoint / "model.py")["sha256"] == config["backend_sha256"]
                  and bind["axis_sha256"] == axis and bind["parent_checkpoint"] == fold["parent"]
                  and bind["candidate_protocol"] == config["candidate_protocol"], "Saved scientific binding differs")
            backend = driver.load_plugin(checkpoint / "model.py")
            state = backend.load(checkpoint / "model")
            check(state["history_coef"].tolist() == receipt["history_coef"]
                  and state["current_coef"] == receipt["current_coef"] and state["active_covariates"] == receipt["active_covariates"],
                  "Saved coefficients differ from fit receipt")
            with np.load(inputs[key + "/query"], allow_pickle=False) as data:
                check(data.files == ["x"], "Request carries fields other than x")
                x = data["x"]
            check(x.shape == (len(query_rows), 2 * n + 4) and x.dtype.kind == "f" and np.isfinite(x).all()
                  and np.array_equal(x[:, 2 * n], [r["historical_bloom_group_code"] for r in query_rows]),
                  "Wrong request shape or metadata")
            variants = {"ridge_history": "history", "ridge_current": "current", "half_source_drift": "half_source_drift"}
            prediction = {label: backend.predict(state, {"x": x}, {**recipe["config"], "rule": rule}) for label, rule in variants.items()}
            current = x[:, n:2 * n]
            prediction.update(persistence=current, full_source_drift=current + state["mean_late"],
                              source_future_mean=np.broadcast_to(state["mean_current"] + state["mean_late"], current.shape))
            check(all(v.shape == (len(query_rows), n) and np.isfinite(v).all() for v in prediction.values()), "Invalid prediction")
            out = args.output_dir / f"{key}_predictions.npz"
            np.savez_compressed(out, **prediction, genotype_ids=np.asarray(fold["held_genotypes"]))
            rows.append({"fold": key, "cut": fold["cut"], "held_genotypes": fold["held_genotypes"],
                         "source_genotypes": fold["source_genotypes"], "request": pin(inputs[key + "/query"]),
                         "request_rows": pin(inputs[key + "/query_rows"]), "fit_receipt": pin(inputs[key + "/receipt"]),
                         "checkpoint_archive": pin(archive), "checkpoint": ck_sha,
                         "parent": fold["parent"], "prediction": {"path": out.name, **pin(out)},
                         "history_coef": receipt["history_coef"], "current_coef": receipt["current_coef"]})
    report = {"status": "completed", "phase": "predict", "task": "apricot_native800",
              "candidate_protocol": config["candidate_protocol"], "axis_sha256": config["axis_sha256"],
              "folds": rows, "fit_calls": 0, "score_calls": 0, "outcomes_mounted": False,
              "scientific_approval": False}
    (args.output_dir / "adapter_receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
