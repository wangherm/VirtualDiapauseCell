"""Score both frozen apricot cuts on their shared original 800h endpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def scores(prediction, truth):
    error = prediction - truth
    return {"MAE": float(np.abs(error).mean()), "MSE": float(np.square(error).mean())}


def mean_scores(rows):
    return {name: {metric: float(np.mean([r["scores"][name][metric] for r in rows]))
                   for metric in ("MAE", "MSE")} for name in rows[0]["scores"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "input-map", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    cfg = read(args.config)
    paths = {k: Path(v) for k, v in read(args.input_map).items()}
    check(set(paths) == set(cfg["inputs"]) and all(pin(v) == cfg["inputs"][k] for k, v in paths.items()),
          "Wrong exact score inputs")
    freeze = read(paths["prediction_receipt"])
    check(freeze["status"] == "completed" and freeze["phase"] == "predict" and freeze["task"] == "apricot_native800"
          and freeze["candidate_protocol"] == cfg["candidate_protocol"] and freeze["axis_sha256"] == cfg["axis_sha256"]
          and freeze["fit_calls"] == freeze["score_calls"] == 0 and not freeze["outcomes_mounted"], "Wrong frozen predictor provenance")
    frozen = {r["fold"]: r for r in freeze["folds"]}
    check(len(frozen) == len(freeze["folds"]) == 6, "Wrong fold coverage")
    cuts = ("cultivar_out", "historical_bloom_group_out")
    modes = cfg["modes"]
    rows = {k: {"centroids": [], "libraries": []} for k in cuts}
    d = cfg["features"]
    for fold in cfg["folds"]:
        f = fold["fold"]
        cut = fold["cut"]
        held = fold["held_genotypes"]
        outrows = rows[cut]
        fr = frozen[f]
        check(fr["cut"] == cut and fr["held_genotypes"] == held and fr["source_genotypes"] == fold["source_genotypes"]
              and fr["request"] == fold["request"] and fr["request_rows"] == fold["request_rows"]
              and all(fr["prediction"][k] == cfg["inputs"][f + "/prediction"][k] for k in ("bytes", "sha256")),
              "Prediction differs from the bound held request")
        sample = read(paths[f + "/library_rows"])
        with np.load(paths[f + "/truth"], allow_pickle=False) as truth:
            check(set(truth.files) == {"y", "library_y", "library_query_index"}, "Unexpected outcome fields")
            y, library, idx = (truth[k] for k in ("y", "library_y", "library_query_index"))
        check(y.shape == (len(held), d) and library.shape == (len(sample), d)
              and np.isfinite(y).all() and np.isfinite(library).all() and idx.shape == (len(sample),)
              and idx.dtype.kind in "iu" and np.array_equal(idx, [r["query_row"] for r in sample]), "Invalid outcome membership")
        check([r["GSM"] for r in sample] == [gsm for g in held for gsm in cfg["outcome_GSMs"][g]],
              "Wrong exact held library order")
        check(all(0 <= ix < len(held) and row["genotype"] == held[ix] and row["future_CH"] == 800
                  and row["future_tissue_native"] == cfg["outcome_tissue"][held[ix]]
                  for ix, row in zip(idx, sample)), "Wrong library cohort or tissue provenance")
        with np.load(paths[f + "/prediction"], allow_pickle=False) as packet:
            check(set(packet.files) == set(modes) | {"genotype_ids"} and packet["genotype_ids"].tolist() == held,
                  "Wrong prediction axes")
            pred = {k: packet[k] for k in modes}
        check(all(v.shape == y.shape and np.isfinite(v).all() for v in pred.values()), "Invalid apricot outputs")
        for j, g in enumerate(held):
            check(np.array_equal(y[j], library[idx == j].mean(axis=0)), "Centroid is not original library mean")
            cm = {"genotype": g, "fold": f, "GSMs": cfg["outcome_GSMs"][g],
                  "scores": {k: scores(v[j], y[j]) for k, v in pred.items()}}
            outrows["centroids"].append(cm)
            local = []
            for index, r in enumerate(sample):
                if idx[index] == j:
                    local.append({"genotype": g, "fold": f, "GSM": r["GSM"],
                                  "scores": {k: scores(v[j], library[index]) for k, v in pred.items()}})
            outrows["libraries"].extend(local)
            averaged = mean_scores(local)
            variance = [averaged[k]["MSE"] - cm["scores"][k]["MSE"] for k in modes]
            check(min(variance) >= -1e-12 and max(variance) - min(variance) <= 1e-12
                  and all(averaged[k]["MAE"] + 1e-12 >= cm["scores"][k]["MAE"] for k in modes), "Outcome aggregation inconsistency")
    summary = {}
    for cut in cuts:
        cs, ls = (rows[cut][x] for x in ("centroids", "libraries"))
        check(len(cs) == len({r["genotype"] for r in cs}) == 4 and len(ls) == len({r["GSM"] for r in ls}) == 11,
              "Wrong per-cut endpoint coverage")
        prior = read(paths[cut + "/historical_primary"])["scores"]
        macro = mean_scores(cs)
        check(all(np.isclose(macro["persistence"][m], prior["current400_persistence"][m], rtol=0, atol=1e-12)
                  for m in ("MAE", "MSE")), "Retained persistence changed on same outcomes")
        summary[cut] = {"fresh_centroid_equal4": macro, "fresh_library_equal11": mean_scores(ls),
                        "cached_historical_centroid_equal4": prior,
                        "ridge_history_minus_current": {k: macro["ridge_history"][k] - macro["ridge_current"][k] for k in ("MAE", "MSE")},
                        "ridge_history_minus_persistence": {k: macro["ridge_history"][k] - macro["persistence"][k] for k in ("MAE", "MSE")},
                        "ridge_history_minus_half_drift": {k: macro["ridge_history"][k] - macro["half_source_drift"][k] for k in ("MAE", "MSE")},
                        "ridge_history_beats_current_centroids": sum(r["scores"]["ridge_history"]["MAE"] < r["scores"]["ridge_current"]["MAE"] - 1e-12 for r in cs),
                        "ridge_history_beats_persistence_centroids": sum(r["scores"]["ridge_history"]["MAE"] < r["scores"]["persistence"]["MAE"] - 1e-12 for r in cs)}
        write(args.output_dir / (cut + "_centroids.json"), cs)
        write(args.output_dir / (cut + "_libraries.json"), ls)
    check({r["GSM"] for r in rows[cuts[0]]["libraries"]} == {r["GSM"] for r in rows[cuts[1]]["libraries"]},
          "Two cuts must share exact 11 outcomes")
    report = {"candidate_protocol": cfg["candidate_protocol"], "prediction_receipt": pin(paths["prediction_receipt"]),
              "summary": summary, "future_libraries": 11, "cultivar_endpoints": 4, "orchards": 1,
              "two_cuts_reuse_same_endpoints": True, "historical_comparators_are_cached_scores": True,
              "fit_calls": 0, "prediction_calls": 0, "scientific_approval": False}
    write(args.output_dir / "apricot_scores.json", report)
    write(args.output_dir / "adapter_receipt.json", {"status": "completed", "phase": "score", "task": "apricot_native800",
          "score_summary": pin(args.output_dir / "apricot_scores.json"), "score_cuts": list(cuts),
          "fit_calls": 0, "prediction_calls": 0, "scientific_approval": False})


if __name__ == "__main__":
    main()

