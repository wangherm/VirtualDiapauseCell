"""Stable command boundaries for staged debugging. Commands never implicitly train another module."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from .io import read_json, read_jsonl, write_json, save_npz


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vdc", description="Virtual Diapause Cell modular research framework")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit"); p.add_argument("bundle")
    p = sub.add_parser("train-state")
    p.add_argument("bundle"); p.add_argument("run"); p.add_argument("--steps", type=int, default=100)
    p.add_argument("--config"); p.add_argument("--device", default="cpu"); p.add_argument("--resume", action="store_true")
    p.add_argument("--semantics"); p.add_argument("--semantic-mode", choices=["correct", "zero", "shuffled"], default="correct")
    p = sub.add_parser("evaluate-state"); p.add_argument("run"); p.add_argument("bundle"); p.add_argument("output")
    p.add_argument("--split", choices=["validation", "test", "locked_test"], default="test")
    p.add_argument("--allow-locked-test", action="store_true")
    p = sub.add_parser("predict-state"); p.add_argument("run"); p.add_argument("bundle"); p.add_argument("output")
    p = sub.add_parser("fit-waves"); p.add_argument("bundle"); p.add_argument("output")
    p.add_argument("--degree", type=int, default=1); p.add_argument("--context-id", required=True)
    p = sub.add_parser("fit-response"); p.add_argument("bundle"); p.add_argument("output")
    p.add_argument("--alpha", type=float, default=1.0)
    p = sub.add_parser("predict-response"); p.add_argument("model"); p.add_argument("bundle"); p.add_argument("output")
    p = sub.add_parser("knowledge-audit"); p.add_argument("records"); p.add_argument("--exclude-family", action="append", default=[])
    p = sub.add_parser("knowledge-search"); p.add_argument("records"); p.add_argument("query")
    p.add_argument("--exclude-family", action="append", default=[])
    for name in ("knowledge-embed", "knowledge-train"):
        p = sub.add_parser(name); p.add_argument("records"); p.add_argument("output")
        p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507"); p.add_argument("--revision", required=True)
        p.add_argument("--allow-download", action="store_true")
        if name == "knowledge-embed": p.add_argument("--device", default="cpu")
        else:
            p.add_argument("--steps", type=int, default=40); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("release"); p.add_argument("run"); p.add_argument("review"); p.add_argument("output")
    p = sub.add_parser("serve"); p.add_argument("release"); p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "audit":
        from .contracts import ObservationBundle, audit_rows
        b = ObservationBundle.load(args.bundle)
        report = {"rows": len(b.rows), "programmes": len(b.feature_ids),
                  "splits": audit_rows(b.rows, b.split_policy), "scope": b.scope,
                  "fingerprint": b.fingerprint, "matrix_loaded": True, "science_status": "not_assessed"}
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "train-state":
        from .contracts import ObservationBundle
        from .state import fit_state, StateConfig
        b = ObservationBundle.load(args.bundle)
        cfg = StateConfig(**yaml.safe_load(Path(args.config).read_text())) if args.config else StateConfig()
        semantics, provenance = None, None
        if args.semantics:
            with np.load(Path(args.semantics) / "embeddings.npz", allow_pickle=False) as a:
                if a["feature_ids"].tolist() != b.feature_ids: raise ValueError("Semantic object order differs")
                semantics = a["vectors"].copy()
            provenance = read_json(Path(args.semantics) / "embeddings.json")
            provenance = {**provenance, "ablation": args.semantic_mode}
            if args.semantic_mode == "zero": semantics[:] = 0
            if args.semantic_mode == "shuffled": semantics = semantics[np.random.default_rng(cfg.seed).permutation(len(semantics))]
        elif args.semantic_mode != "correct":
            raise ValueError("A semantic cache is required for dimension-matched ablations")
        print(json.dumps(fit_state(b, args.run, args.steps, cfg, args.device, args.resume, semantics, provenance),
                         indent=2, ensure_ascii=False))
    elif args.command == "evaluate-state":
        from .contracts import ObservationBundle
        from .evaluate import evaluate_state
        print(json.dumps(evaluate_state(args.run, ObservationBundle.load(args.bundle), args.output, args.split, args.allow_locked_test), indent=2))
    elif args.command == "predict-state":
        from .contracts import ObservationBundle
        from .state import StatePredictor
        pred = StatePredictor.load(args.run).predict(ObservationBundle.load(args.bundle))
        save_npz(args.output, **{k: v for k, v in pred.items() if isinstance(v, np.ndarray)})
        write_json(str(args.output) + ".json", {k: v for k, v in pred.items() if not isinstance(v, np.ndarray)})
    elif args.command == "fit-waves":
        from .contracts import ObservationBundle
        from .waves import WaveReference
        b = ObservationBundle.load(args.bundle)
        rows = [{**r, "reference_eligible": r.get("reference_eligible", False) and bool(b.clock_mask[i])}
                for i, r in enumerate(b.rows)]
        model = WaveReference.fit(b.clock, b.values, b.mask, b.feature_ids, rows,
                  b.clock_reference_id or "", args.context_id, degree=args.degree, split_policy=b.split_policy)
        dest = Path(args.output)
        if dest.exists() and any(dest.iterdir()): raise FileExistsError("Use a fresh wave output directory")
        model.save(dest)
    elif args.command == "fit-response":
        from .response import ResponseDataset, ResponseRegressor
        from .metrics import grouped_metrics
        d = ResponseDataset.load(args.bundle)
        dest = Path(args.output)
        if dest.exists() and any(dest.iterdir()): raise FileExistsError("Use a fresh response output directory")
        model = ResponseRegressor(args.alpha).fit(d); model.save(dest)
        idx = [i for i, r in enumerate(d.rows) if r["split"] == "validation"]
        if idx:
            pred = model.predict(d.current[idx], d.action[idx], d.scope,
                                 None if d.elapsed is None else d.elapsed[idx])
            eval_mask = d.target_mask[idx] & model.target_available[None]
            report = grouped_metrics(pred, d.target[idx], eval_mask, [d.rows[i] for i in idx])
            report.update({"science_status": "unvalidated", "mode": d.mode,
                           "target_unavailable": int((~model.target_available).sum()), "test_evaluated": False})
            write_json(dest / "metrics.json", report)
            save_npz(dest / "validation_predictions.npz", prediction=pred, target=d.target[idx], mask=eval_mask)
    elif args.command == "predict-response":
        from .response import ResponseDataset, ResponseRegressor
        d = ResponseDataset.load(args.bundle); model = ResponseRegressor.load(args.model)
        pred = model.predict(d.current, d.action, d.scope, d.elapsed)
        save_npz(args.output, prediction=pred, available=model.target_available)
    elif args.command.startswith("knowledge-"):
        from .knowledge import audit_knowledge, retrieve, embed_objects
        records = read_jsonl(args.records)
        if args.command == "knowledge-audit":
            print(json.dumps(audit_knowledge(records, set(args.exclude_family)), indent=2))
        elif args.command == "knowledge-search":
            print(json.dumps(retrieve(records, args.query, set(args.exclude_family)), indent=2, ensure_ascii=False))
        elif args.command == "knowledge-embed":
            print(json.dumps(embed_objects(records, args.output, args.model, args.revision,
                                           args.device, args.allow_download), indent=2))
        else:
            from .knowledge_train import fit_knowledge
            fit_knowledge(records, args.output, args.model, args.revision, args.steps,
                          resume=args.resume, allow_download=args.allow_download)
    elif args.command == "release":
        from .release import create_release
        print(json.dumps(create_release(args.run, args.review, args.output), indent=2))
    elif args.command == "serve":
        import uvicorn
        from .service import create_app
        uvicorn.run(create_app(args.release), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
