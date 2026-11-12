"""Score exact held-cell predictions against the published germination means."""
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import argparse
import json
from math import fsum, isfinite
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
for name in ("config", "input-map", "output-dir"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
OUT = args.output_dir
role_paths = json.loads(args.input_map.read_bytes())


def pin(b):
    return {"bytes": len(b), "sha256": sha256(b).hexdigest()}


def write(name, obj):
    b = (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    (OUT / name).write_bytes(b)
    return pin(b)


def valid_fraction(x):
    return type(x) in (float, int) and isfinite(x) and 0.0 <= x <= 1.0


def mean(values):
    return fsum(values) / len(values)


def metrics(rows, head):
    errors = [row["errors"][head]["signed_pp"] for row in rows]
    return {"cells": len(errors), "mae_pp": mean([abs(x) for x in errors]),
            "mean_signed_pp": mean(errors), "worst_abs_pp": max(map(abs, errors))}


def comparison(rows, candidate, reference, tolerance):
    delta = [row["candidate_minus_reference_abs_pp"][reference] for row in rows]
    difference = [abs(row["predictions"][candidate] - row["predictions"][reference]) for row in rows]
    return {
        "candidate_minus_reference_mae_pp": mean(delta),
        "wins": sum(x < -100 * tolerance for x in delta),
        "losses": sum(x > 100 * tolerance for x in delta),
        "ties": sum(abs(x) <= 100 * tolerance for x in delta),
        "predictions_numerically_equivalent": max(difference) <= tolerance,
        "max_abs_prediction_difference_fraction": max(difference),
    }


config_raw = args.config.read_bytes()
cfg = json.loads(config_raw)
manifest_spec = cfg["prediction_manifest"]
specs = [manifest_spec] + [p for fold in cfg["folds"] for p in (fold["prediction"], fold["truth"])]
role_names = ['prediction_manifest', 'fold_0_prediction', 'fold_0_truth', 'fold_1_prediction', 'fold_1_truth', 'fold_2_prediction', 'fold_2_truth', 'fold_3_prediction', 'fold_3_truth']
assert set(role_paths) == set(role_names) and len(specs) == len(role_names)
by_ref = {tuple(sorted(spec["ref"].items())): Path(role_paths[name])
          for name, spec in zip(role_names, specs)}
assert len(by_ref) == len(specs)


def exact(spec):
    raw = by_ref[tuple(sorted(spec["ref"].items()))].read_bytes()
    assert pin(raw) == spec["pin"]
    return json.loads(raw)


manifest = exact(manifest_spec)
assert manifest["status"] == "completed"
assert manifest["protocol_id"] == cfg["protocol_id"] and manifest["protocol_pin"] == cfg["protocol_pin"]
assert manifest["config_pin"] == cfg["prediction_config_pin"]
assert manifest["schema"]["pin"] == cfg["schema_pin"]
assert manifest["reloaded_models"] == manifest["fresh_prediction_processes"] == 4
assert manifest["predicted_cells"] == 24
assert all(manifest[k] == 0 for k in ("source_label_packets_mounted", "held_truth_packets_mounted", "fit_calls", "score_calls"))
assert len(manifest["folds"]) == len(cfg["folds"]) == 4
candidate = cfg["candidate"]
references = cfg["comparators"]
heads = [candidate] + references
lines, arms = cfg["genotypes"], cfg["condition_ids"]
rows = []
for i, (fold, pred_entry) in enumerate(zip(cfg["folds"], manifest["folds"])):
    assert fold["fold"] == pred_entry["fold"] == i
    assert pred_entry["prediction"] == {"path": fold["prediction"]["ref"]["path"], "pin": fold["prediction"]["pin"]}
    assert pred_entry["inputs"]["model"]["pin"] == fold["model_pin"]
    assert pred_entry["inputs"]["query"]["pin"] == fold["query_pin"]
    pred, truth = exact(fold["prediction"]), exact(fold["truth"])
    assert pred["fold"] == truth["fold"] == i and truth["role"] == "truth"
    assert pred["protocol_id"] == truth["protocol_id"] == cfg["protocol_id"]
    assert pred["schema_pin"] == truth["schema_pin"] == cfg["schema_pin"]
    assert pred["model_pin"] == fold["model_pin"] and pred["query_pin"] == fold["query_pin"]
    assert pred["target"] == "germinated_fraction"
    row_ids = [f"2014/{g}/{a}" for gi, g in enumerate(lines) for ai, a in enumerate(arms) if (gi + ai) % 4 == i]
    assert pred["row_ids"] == pred_entry["query_ids"] == [r["row_id"] for r in truth["rows"]] == row_ids
    assert len(row_ids) == 6 and set(pred["predictions"]) == set(heads)
    assert all(len(v) == 6 and all(map(valid_fraction, v)) for v in pred["predictions"].values())
    for j, t in enumerate(truth["rows"]):
        row_id = t["row_id"]
        year, line, arm = row_id.split("/")
        assert year == "2014" and t["group"] == f"{year}/{line}"
        y = t["germinated_fraction"]
        assert valid_fraction(y) and float(Decimal(t["observed_mean_percent"]) / Decimal(100)) == y
        values = {head: pred["predictions"][head][j] for head in heads}
        errors = {head: {"signed_pp": (v - y) * 100, "abs_pp": abs(v - y) * 100}
                  for head, v in values.items()}
        rows.append({
            "row_id": row_id, "fold": i, "line_harvest_group": t["group"], "genotype": line,
            "condition_id": arm, "observed_fraction": y, "observed_mean_percent": t["observed_mean_percent"],
            "published_sd_percent": t["published_sd_percent"], "predictions": values, "errors": errors,
            "candidate_minus_reference_abs_pp": {
                r: errors[candidate]["abs_pp"] - errors[r]["abs_pp"] for r in references},
        })
assert Counter(r["row_id"] for r in rows) == {f"2014/{g}/{a}": 1 for g in lines for a in arms}
rows.sort(key=lambda r: (lines.index(r["genotype"]), arms.index(r["condition_id"])))
tolerance = cfg["numerical_equivalence_fraction"]
assert tolerance == 1e-12
overall = {h: metrics(rows, h) for h in heads}
comparisons = {r: comparison(rows, candidate, r, tolerance) for r in references}
for r, v in comparisons.items():
    assert v["wins"] + v["losses"] + v["ties"] == 24
    assert abs(overall[candidate]["mae_pp"] - overall[r]["mae_pp"] - v["candidate_minus_reference_mae_pp"]) < 1e-10
line_scores = {f"2014/{g}": {h: metrics([r for r in rows if r["genotype"] == g], h) for h in heads} for g in lines}
condition_scores = {a: {h: metrics([r for r in rows if r["condition_id"] == a], h) for h in heads} for a in arms}
summary = {
    "protocol_id": cfg["protocol_id"], "protocol_pin": cfg["protocol_pin"],
    "primary_metric": "Equal-cell MAE in percentage points", "held_cells": 24,
    "line_harvest_groups": 4, "harvests": 1, "background_families": 1,
    "candidate": candidate, "metrics": overall, "paired_comparisons": comparisons,
    "line_mae_pp": {key: {h: v[h]["mae_pp"] for h in heads} for key, v in line_scores.items()},
    "condition_mae_pp": {key: {h: v[h]["mae_pp"] for h in heads} for key, v in condition_scores.items()},
    "numerical_equivalence_fraction": tolerance,
    "scope": "Retrospective within-panel completion with same-line calibration. One 2014 harvest and Columbia-0 family.",
    "RNA_increment": "not_assessed_no_linked_initial_RNA", "background_transfer": "not_assessed_one_family",
    "scientific_approval": False,
}
out_pins = {}
out_pins["functional_score_summary.json"] = write("functional_score_summary.json", summary)
out_pins["functional_cell_errors.json"] = write("functional_cell_errors.json", {
    "protocol_id": cfg["protocol_id"], "units": "published line-treatment means", "rows": rows,
    "line_harvest_scores": line_scores, "condition_scores": condition_scores})
write("functional_score_manifest.json", {
    "status": "completed", "score_config_pin": pin(config_raw), "prediction_manifest": manifest_spec,
    "folds": cfg["folds"], "outputs": out_pins, "scored_cells": 24, "scored_heads": len(heads),
    "model_states_mounted": 0, "source_label_packets_mounted": 0,
    "fit_calls": 0, "predict_calls": 0, "scientific_approval": False,
})
write("metadata.json", {"status": "scoring_executed", "task_ready": False,
                        "summary": "functional_score_summary.json", "manifest": "functional_score_manifest.json"})

(OUT / "adapter_receipt.json").write_text(json.dumps({
    "status": "completed", "phase": 'score', "task": 'chilling_functional',
    "protocol_id": cfg["protocol_id"], "scope": "retrospective same-panel comparison",
    "operational_default": None, "scientific_approval": False,
}, indent=2) + "\n", encoding="utf-8")
