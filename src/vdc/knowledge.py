"""Reviewed knowledge records and optional frozen Qwen object embeddings."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import torch
from .io import object_hash, save_npz, write_json


def audit_knowledge(records: list[dict], excluded_families: set[str] | None = None) -> dict:
    excluded = excluded_families or set()
    families, ids = {}, set()
    for r in records:
        for k in ("record_id", "study_family", "source_ref", "split", "kind", "text"):
            if not isinstance(r.get(k), str) or not r[k].strip():
                raise ValueError(f"Knowledge record requires {k}")
        if r.get("reviewed") is not True:
            raise ValueError("Unreviewed model-generated records cannot become training truth")
        if r["record_id"] in ids: raise ValueError("Duplicate knowledge record")
        ids.add(r["record_id"])
        if r["study_family"] in excluded: raise ValueError("Excluded experimental family in knowledge inputs")
        if r["split"] not in {"train", "validation", "test", "locked_test"}: raise ValueError("Unknown split")
        if r["kind"] not in {"object_description", "experimental_evidence"}: raise ValueError("Unknown record kind")
        prior = families.setdefault(r["study_family"], r["split"])
        if prior != r["split"]: raise ValueError("Paper/preprint/supplement family crosses knowledge splits")
        if r["kind"] == "experimental_evidence" and r.get("evidence_type") not in {
            "observational", "measured_perturbation", "rescue", "computational_prediction", "hypothesis"}:
            raise ValueError("Experimental evidence must declare its evidence type")
    return {"records": len(records), "study_families": len(families),
            "base_model_pretraining_contamination": "not_fully_auditable"}


def retrieve(records: list[dict], query: str, excluded_families: set[str], limit: int = 5) -> list[dict]:
    """Small source-aware lexical baseline, not a semantic RAG model."""
    # Held-out evidence is filtered BEFORE scoring, not after generation.
    tokens = set(re.findall(r"\w+", query.lower()))
    candidates = [r for r in records if r["study_family"] not in excluded_families
                  and r.get("reviewed") is True and r["split"] == "train"]
    def score(r: dict) -> int:
        return len(tokens & set(re.findall(r"\w+", r["text"].lower())))
    return sorted([r for r in candidates if score(r) > 0], key=score, reverse=True)[:limit]


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask[..., None].to(last_hidden_state.dtype)
    if not attention_mask.any(-1).all(): raise ValueError("Empty text sequence")
    return (last_hidden_state * mask).sum(1) / mask.sum(1)


def embed_objects(records: list[dict], output: str | Path, model_name: str, revision: str,
                  device: str = "cpu", allow_download: bool = False, max_length: int = 512) -> dict:
    audit_knowledge(records)
    if not revision or revision in {"main", "latest"}:
        raise ValueError("Pin an immutable model commit or a resolved local snapshot revision")
    if any(r["kind"] != "object_description" or r["split"] != "train" for r in records):
        raise ValueError("Numeric object vectors accept train descriptions, not held-out outcome evidence")
    if any(not r.get("object_id") for r in records): raise ValueError("Object IDs are required")
    if len({r["object_id"] for r in records}) != len(records): raise ValueError("Duplicate object ID")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install vdc-research[knowledge]; no embedding fallback is used") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=not allow_download)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=not allow_download,
                                     torch_dtype=dtype, trust_remote_code=False).to(device).eval()
    vectors = []
    for r in records:
        inputs = tokenizer(r["text"], return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[1] > max_length:
            raise ValueError(f"Description exceeds max_length: {r['object_id']}; shorten explicitly")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
            v = masked_mean_pool(out.last_hidden_state, inputs["attention_mask"])
        vectors.append(v.float().cpu().numpy()[0])
    output = Path(output)
    save_npz(output / "embeddings.npz", vectors=np.stack(vectors),
             feature_ids=np.array([r["object_id"] for r in records]))
    provenance = {"model": model_name, "revision": revision, "pooling": "masked_mean_last_hidden_state",
                  "corpus_hash": object_hash(records), "record_ids": [r["record_id"] for r in records],
                  "status": "computed_embeddings_not_validated_biology"}
    write_json(output / "embeddings.json", provenance)
    return provenance


def completion_example(tokenizer, prompt: list[dict], answer: str, max_length: int) -> dict:
    """One verified assistant boundary and an explicit completion-only mask."""
    prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    text = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": answer}],
                                         tokenize=False, add_generation_prompt=False)
    pids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids[:len(pids)] != pids or len(ids) <= len(pids):
        raise ValueError("Chat-template/token boundary differs; inspect rather than silently shifting labels")
    if len(ids) > max_length: raise ValueError("SFT example exceeds length limit; no silent truncation")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(pids) + ids[len(pids):]}
