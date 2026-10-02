"""Local research-release packaging. Tests are not scientific acceptance.

An explicit researcher review binds a checkpoint and evaluation report. This is
an audit gate, not automatic verification of every biological claim in a report.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from .io import read_json, write_json, sha256, object_hash


def create_release(run_dir: str | Path, review_path: str | Path, out_dir: str | Path) -> dict:
    run, out = Path(run_dir), Path(out_dir)
    manifest = read_json(run / "run.json")
    review = read_json(review_path)
    if manifest["is_synthetic"]:
        raise ValueError("Synthetic software-test weights cannot be released as research models")
    if review.get("decision") != "approved_for_declared_research_scope" or not review.get("reviewed_by"):
        raise ValueError("Explicit scientific review is required")
    checkpoint_hash = sha256(run / "best.pt")
    if review.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Approval is for a different checkpoint")
    if review.get("scope_hash") != object_hash(manifest["scope"]):
        raise ValueError("Approval scope differs")
    evaluation_path = Path(review["evaluation_path"])
    if not evaluation_path.is_absolute(): evaluation_path = Path(review_path).parent / evaluation_path
    if sha256(evaluation_path) != review.get("evaluation_sha256"):
        raise ValueError("Evaluation checksum differs")
    evaluation = read_json(evaluation_path)
    if evaluation.get("data_kind") != "real" or evaluation.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Independent real-data evaluation tied to this checkpoint is required")
    if not evaluation.get("baseline_comparisons") or not evaluation.get("evaluation_scope"):
        raise ValueError("Evaluation must declare scope and comparisons, not just token loss")
    caps = review.get("approved_capabilities", [])
    if not caps or not set(caps).issubset(manifest["capabilities"]):
        raise ValueError("Approval includes unavailable capabilities")
    if out.exists(): raise FileExistsError("Release directories are immutable")
    out.mkdir(parents=True)
    for name in ("best.pt", "run.json"):
        shutil.copy2(run / name, out / name)
    shutil.copy2(evaluation_path, out / "evaluation.json")
    release = {"schema_version": "research-release-v1", "checkpoint_sha256": checkpoint_hash,
               "scope": manifest["scope"], "capabilities": caps, "review": review,
               "evaluation_sha256": sha256(out / "evaluation.json"),
               "usage": "local_research_only", "public_service_deployed": False}
    write_json(out / "release.json", release)
    return release


def verify_release(directory: str | Path) -> dict:
    p = Path(directory); r = read_json(p / "release.json"); m = read_json(p / "run.json")
    if sha256(p / "best.pt") != r["checkpoint_sha256"] or m["scope"] != r["scope"]:
        raise ValueError("Release checkpoint or scope changed")
    if sha256(p / "evaluation.json") != r["evaluation_sha256"]:
        raise ValueError("Released evaluation changed")
    return r
