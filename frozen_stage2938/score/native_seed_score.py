"""Score frozen package predictions using the fixed GSE90162 outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def expected(spec):
    return {k: spec[k] for k in ("bytes", "sha256")}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def arrays(path, names):
    with np.load(path, allow_pickle=False) as data:
        check(set(data.files) == set(names), "Wrong packet fields")
        return {k: data[k] for k in names}


def numeric(array, shape):
    check(array.shape == shape and array.dtype.kind == "f" and np.isfinite(array).all(), "Wrong numerical shape or dtype")
    return array


def metrics(x, y, ranks=False):
    error = x - y
    out = {"mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean())}
    if ranks:
        out["rank_mae"] = float(np.abs(rankdata(x, method="average") - rankdata(y, method="average")).mean() / (x.size - 1))
    return out


def mean_table(rows):
    return {name: {metric: float(np.mean([r["metrics"][name][metric] for r in rows]))
                   for metric in keys} for name, keys in rows[0]["metrics"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("contract", "score-inputs", "binding", "split", "frozen-dir", "outcome-dir", "control-dir", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    a = parser.parse_args()
    contract, spec, binding, split = (read(x) for x in (a.contract, a.score_inputs, a.binding, a.split))
    groups = contract["groups"]
    version = binding["package_version"]
    d = contract["feature_count"]
    check(pin(a.contract) == binding["contract"] == spec["contract"]
          and pin(a.split) == contract["split"] and split["version"] == contract["benchmark_version"]
          and split["split_protocol_sha256"] == contract["protocol_sha256"]
          and split["feature_axis_sha256"] == contract["axis_sha256"]
          and set(spec["controls"]) == set(groups), "Wrong scoring contract")
    check({p.name for p in a.outcome_dir.iterdir()} == {f"outcome_{g}.npz" for g in groups}
          and {p.name for p in a.control_dir.iterdir()} == {f"controls_{g}.npz" for g in groups}
          and {p.name for p in a.frozen_dir.iterdir()} == {"prediction_manifest.json"} |
          {f"{g}_{role}.npz" for g in groups for role in ("package_reference", "candidate")}, "Wrong packet directory membership")
    mf = a.frozen_dir / "prediction_manifest.json"
    predictions = read(mf)
    check(predictions["contract"] == pin(a.contract) and predictions["binding"] == pin(a.binding)
          and predictions["split"] == pin(a.split) and predictions["archive"] == binding["archive"]
          and predictions["package_version"] == version and predictions["benchmark_version"] == contract["benchmark_version"]
          and predictions["runner_version"] == contract["runner_version"] and predictions["fit_calls"] == predictions["score_calls"] == 0
          and not predictions["outcomes_required"], "Wrong freeze provenance")
    run_rows = predictions["runs"]
    runs = {(r["consumer_receipt"]["genotype"], r["role"]): r for r in run_rows}
    check(len(runs) == len(run_rows) == 10 and set(runs) == {(g, k) for g in groups for k in binding["modes"]}, "Wrong freeze cases")
    folds = {r["held_genotype"]: r for r in split["folds"]}
    check(list(folds) == groups, "Wrong fold order")
    rows, arrays_rows, provenance = [], [], {}
    for g, f in folds.items():
        truth, old = a.outcome_dir / f"outcome_{g}.npz", a.control_dir / f"controls_{g}.npz"
        check(pin(truth) == expected(f["outputs"]["outcome"]) and pin(old) == spec["controls"][g], "Wrong held or frozen control pin")
        target = arrays(truth, ("y", "ar_arrays"))
        center = numeric(target["y"], (1, d))[0]
        ar = numeric(target["ar_arrays"], (3, d))
        check(np.array_equal(center, ar.mean(axis=0)), "Wrong held centroid")
        fixed = arrays(old, ("fitted", "source_ar_mean", "persistence", "shifted_persistence", "fixed_half_shift"))
        values = {k: numeric(v, (1, d))[0] for k, v in fixed.items() if k != "fitted"}
        values["parent_scalar"] = numeric(fixed["fitted"], (1, d))[0]
        pins = {}
        for role in ("package_reference", "candidate"):
            row = runs[g, role]
            receipt = row["consumer_receipt"]
            path = a.frozen_dir / f"{g}_{role}.npz"
            check(row["prediction"] == {"path": path.name, **pin(path)}
                  and row["original_request"] == expected(f["outputs"]["request"])
                  and receipt["package_version"] == version and receipt["mode"] == binding["modes"][role]
                  and receipt["checkpoint_sha256"] == binding["checkpoints"][g] and receipt["fit_calls"] == 0
                  and receipt["shape"] == [1, d] and receipt["output_sha256"] == pin(path)["sha256"], "Invalid frozen package prediction")
            result = arrays(path, ("prediction", "feature_ids"))
            axis = result["feature_ids"]
            check(axis.shape == (d,) and axis.dtype.kind == "U"
                  and hashlib.sha256(("\n".join(axis.tolist()) + "\n").encode()).hexdigest() == contract["axis_sha256"], "Wrong output feature axis")
            values[role] = numeric(result["prediction"], (1, d))[0]
            pins[role] = pin(path)
        check(np.array_equal(values["package_reference"], values["source_ar_mean"]), "Source AR mean differs")
        if version == spec["parent_version"]:
            check(np.array_equal(values["candidate"], values["parent_scalar"]), "Parent prediction differs from original")
        controls = ("source_ar_mean", "parent_scalar", "persistence", "shifted_persistence", "fixed_half_shift")
        rows.append({"held_genotype": g, "outcome_gsm": f["outcome_gsm"],
                     "metrics": {k: metrics(v, center, True) for k, v in values.items()},
                     "max_abs_prediction_delta_vs_controls": {k: float(np.abs(values["candidate"] - values[k]).max()) for k in controls}})
        for i, gsm in enumerate(f["outcome_gsm"]):
            arrays_rows.append({"held_genotype": g, "gsm": gsm, "metrics": {k: metrics(v, ar[i]) for k, v in values.items()}})
        provenance[g] = {"truth": pin(truth), "controls": pin(old), "predictions": pins, "request_gsm": f["request_gsm"]}
    check(len(rows) == 5 and len(arrays_rows) == len({x["gsm"] for x in arrays_rows}) == 15, "Wrong evaluation units")
    macro = mean_table(rows)
    atol = contract["equivalence_atol"]
    comparisons = {}
    for k in controls:
        diffs = {r["held_genotype"]: r["metrics"]["candidate"]["mae"] - r["metrics"][k]["mae"] for r in rows}
        comparisons[k] = {"candidate_minus_control_macro_mae": macro["candidate"]["mae"] - macro[k]["mae"],
                          "centroid_mae_delta_by_stratum": diffs,
                          "array_mae_lower": sum(r["metrics"]["candidate"]["mae"] < r["metrics"][k]["mae"] - atol for r in arrays_rows),
                          "array_mae_higher": sum(r["metrics"]["candidate"]["mae"] > r["metrics"][k]["mae"] + atol for r in arrays_rows),
                          "numerically_equivalent_prediction_strata": [r["held_genotype"] for r in rows if r["max_abs_prediction_delta_vs_controls"][k] <= atol]}
    summary = {"package_version": version, "package_archive": binding["archive"], "benchmark_version": contract["benchmark_version"],
               "runner_version": contract["runner_version"], "protocol_sha256": contract["protocol_sha256"], "contract": pin(a.contract),
               "score_input_contract": pin(a.score_inputs), "binding": pin(a.binding), "frozen_predictions": pin(mf),
               "counts": split["counts"], "scored_AR_arrays": 15, "held_strata": 5, "recurrent_backgrounds": 1,
               "macro_genotype_centroid": macro, "equal_AR_array": mean_table(arrays_rows), "candidate_comparisons": comparisons,
               "fit_calls": 0, "prediction_calls": 0, "scientific_approval": False,
               "evaluation_scope": "Retrospective one recurrent Ler panel; no physical serial pairs, independent background or function."}
    a.output_dir.mkdir(exist_ok=True)
    check(not list(a.output_dir.iterdir()), "Use an empty output directory")
    write(a.output_dir / "benchmark_scores.json", summary)
    write(a.output_dir / "per_stratum.json", rows)
    write(a.output_dir / "per_array.json", arrays_rows)
    write(a.output_dir / "scoring_provenance.json", provenance)
    lines = [f"# {version}", "", f"Benchmark: {contract['benchmark_version']}", "",
             "Equal-weight native RMA centroid metrics across five related strata.", "",
             "| Mode/control | MAE | MSE | Rank MAE |", "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {k} | {v['mae']:.10f} | {v['mse']:.10f} | {v['rank_mae']:.10f} |" for k, v in macro.items())
    lines.extend(["", summary["evaluation_scope"], "Scored 15 actual AR arrays. Scoring used frozen package predictions.", ""])
    (a.output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
