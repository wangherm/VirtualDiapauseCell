"""Execute four fresh prediction processes; never mount source or scoring truth."""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
for name in ("config", "input-map", "output-dir"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
OUT = args.output_dir
role_paths = json.loads(args.input_map.read_bytes())
CODE = Path(__file__).resolve().parent


def pin(b):
    return {"bytes": len(b), "sha256": sha256(b).hexdigest()}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8")


cfg_raw = args.config.read_bytes()
cfg = json.loads(cfg_raw)
assert pin((CODE / "chilling_functional_model.py").read_bytes()) == cfg["model_code_pin"]
assert pin((CODE / "predict_packet.py").read_bytes()) == cfg["cli_code_pin"]
assert [x["fold"] for x in cfg["folds"]] == list(range(4))
specs = [cfg["schema"]] + [v for fold in cfg["folds"]
                           for k, v in fold["inputs"].items() if k in ("model", "fit_receipt", "query")]
role_names = ['schema', 'fold_0_fit_receipt', 'fold_0_model', 'fold_0_query', 'fold_1_fit_receipt', 'fold_1_model', 'fold_1_query', 'fold_2_fit_receipt', 'fold_2_model', 'fold_2_query', 'fold_3_fit_receipt', 'fold_3_model', 'fold_3_query']
assert set(role_paths) == set(role_names) and len(specs) == len(role_names)
by_ref = {tuple(sorted(spec["ref"].items())): Path(role_paths[name])
          for name, spec in zip(role_names, specs)}
assert len(by_ref) == len(specs)


def exact(spec):
    path = by_ref[tuple(sorted(spec["ref"].items()))]
    b = path.read_bytes()
    assert pin(b) == spec["pin"]
    return path, json.loads(b)


schema_path, schema = exact(cfg["schema"])
assert schema["protocol_id"] == cfg["protocol_id"] and schema["protocol_pin"] == cfg["protocol_pin"]
result = []
covered = set()
for fold in cfg["folds"]:
    i = fold["fold"]
    inputs = fold["inputs"]
    assert set(inputs) == {"model", "fit_receipt", "query"}
    state_path, state = exact(inputs["model"])
    _, receipt = exact(inputs["fit_receipt"])
    query_path, query = exact(inputs["query"])
    assert receipt["status"] == "completed" and receipt["fold"] == state["fold"] == query["fold"] == i
    assert receipt["config"] == fold["fit_config_pin"] and receipt["model"] == inputs["model"]["pin"]
    assert receipt["mounted_inputs"] == fold["fit_inputs"] and set(receipt["mounted_inputs"]) == {"schema", "source"}
    assert state["source_pin"] == fold["fit_inputs"]["source"]["pin"]
    assert receipt["protocol_pin"] == state["protocol_pin"] == cfg["protocol_pin"]
    assert receipt["source_rows"] == 18 and receipt["additive_rank"] == 9
    for k in ("query_labels_mounted", "query_features_mounted", "prediction_calls", "score_calls"):
        assert receipt[k] == 0
    solver = receipt["solver"]
    assert solver["status"] == "fitted" and solver["interior_rank"] == 9
    assert 0 <= solver["gradient_inf_norm"] <= solver["tolerance"] == 1e-10
    assert len(state["fractional_logit_coefficients"]) == len(state["two_way_ols_coefficients"]) == 9
    assert state["predictor_code_pin"] == cfg["model_code_pin"]
    out = OUT / f"fold_{i}_predictions.json"
    subprocess.run([sys.executable, str(CODE / "predict_packet.py"),
                    "--model", str(state_path), "--schema", str(schema_path),
                    "--query", str(query_path), "--output", str(out)],
                   check=True, timeout=45)
    b = out.read_bytes()
    p = json.loads(b)
    qids = [x["row_id"] for x in query["rows"]]
    assert p["fold"] == i and p["row_ids"] == qids and len(qids) == len(set(qids)) == 6
    assert not (set(qids) & covered)
    assert p["model_pin"] == inputs["model"]["pin"] and p["query_pin"] == inputs["query"]["pin"]
    covered.update(qids)
    result.append({"fold": i, "solver": solver, "inputs": inputs,
                   "source_config_pin": fold["fit_config_pin"], "query_ids": qids,
                   "prediction": {"path": out.name, "pin": pin(b)}})
assert len(covered) == 24
write("functional_prediction_manifest.json", {
    "status": "completed", "protocol_id": cfg["protocol_id"], "protocol_pin": cfg["protocol_pin"],
    "config_pin": pin(cfg_raw), "schema": cfg["schema"], "folds": result,
    "model_code_pin": cfg["model_code_pin"], "cli_code_pin": cfg["cli_code_pin"],
    "reloaded_models": 4, "fresh_prediction_processes": 4, "predicted_cells": 24,
    "source_label_packets_mounted": 0, "held_truth_packets_mounted": 0, "fit_calls": 0, "score_calls": 0,
    "study_scope": "2014 same-panel calibration completion. No independent background or initial RNA claim.",
    "scientific_approval": False,
})
write("metadata.json", {"status": "prediction_executed", "task_ready": False,
                        "prediction_manifest": "functional_prediction_manifest.json"})

(OUT / "adapter_receipt.json").write_text(json.dumps({
    "status": "completed", "phase": 'predict', "task": 'chilling_functional',
    "protocol_id": cfg["protocol_id"], "scope": "retrospective same-panel comparison",
    "operational_default": None, "scientific_approval": False,
}, indent=2) + "\n", encoding="utf-8")
