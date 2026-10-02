"""Optional QLoRA training entry point. Not executed in the CPU framework verification.

Training accepts reviewed train/validation source families and explicit answers.
No raw PDF crawling, automatic gold-label fabrication or numeric validation is performed.
"""
from __future__ import annotations
from pathlib import Path
import torch
from .knowledge import audit_knowledge, completion_example
from .io import write_json, object_hash


def fit_knowledge(records: list[dict], run_dir: str | Path, model_name: str, revision: str,
                  max_steps: int = 40, max_length: int = 2048, resume: bool = False,
                  allow_download: bool = False) -> None:
    audit_knowledge(records)
    if not revision or revision in {"main", "latest"}: raise ValueError("Pin model revision")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This QLoRA entry point requires a BF16-capable CUDA GPU")
    if not any(r["split"] == "train" for r in records) or not any(r["split"] == "validation" for r in records):
        raise ValueError("Reviewed train and validation source families required")
    if any(r["split"] not in {"train", "validation"} for r in records):
        raise ValueError("Remove test records from the training-corpus file")
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    output = Path(run_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Use a new knowledge run or explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=not allow_download)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prepared = {"train": [], "validation": []}
    for r in records:
        if not r.get("prompt") or not r.get("completion"):
            raise ValueError("Knowledge SFT requires reviewed prompt and completion fields")
        prepared[r["split"]].append(completion_example(tokenizer, r["prompt"], r["completion"], max_length))
    contract = {"model": model_name, "revision": revision, "corpus_hash": object_hash(records),
                "completion_only_loss": True, "max_length": max_length,
                "stage": "knowledge_sft_not_numeric_recovery", "base_adapter": "new"}
    if resume:
        from .io import read_json
        if read_json(output / "contract.json") != contract:
            raise ValueError("Knowledge corpus or model contract changed")
    write_json(output / "contract.json", contract)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
               local_files_only=not allow_download, quantization_config=quant,
               device_map={"": torch.cuda.current_device()}, torch_dtype=torch.bfloat16,
               trust_remote_code=False)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules="all-linear", task_type="CAUSAL_LM"))

    def collate(examples: list[dict]) -> dict:
        width = max(len(x["input_ids"]) for x in examples)
        return {key: torch.tensor([x[key] + [pad] * (width - len(x[key])) for x in examples])
                for key, pad in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}

    interval = min(20, max_steps)
    args = TrainingArguments(output_dir=str(output / "checkpoints"), max_steps=max_steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=1e-4, bf16=True,
            logging_steps=1, eval_strategy="steps", eval_steps=interval,
            save_strategy="steps", save_steps=interval, save_total_limit=2,
            load_best_model_at_end=True, report_to=[], seed=42, remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, data_collator=collate,
                      train_dataset=Dataset.from_list(prepared["train"]),
                      eval_dataset=Dataset.from_list(prepared["validation"]))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(output / "adapter")); tokenizer.save_pretrained(output / "adapter")
    write_json(output / "status.json", {"training": "completed", "science_status": "unvalidated",
               "numerical_recovery_tested": False, "checkpoint_selector": "knowledge_validation_token_loss",
               "next_gate": "heldout_evidence_quality_and_separate_numeric_ablation"})
