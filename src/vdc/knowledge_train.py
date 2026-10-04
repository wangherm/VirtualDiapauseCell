"""Actual GPU LoRA/QLoRA training. Smoke, formal epoch and new-process evaluation are distinct."""
from pathlib import Path
import gc
import json
import importlib.metadata
import torch
from .knowledge import audit_knowledge, completion_example
from .io import write_json, read_json, object_hash, sha256


def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required; no CPU/fake training fallback')
    p = torch.cuda.get_device_properties(0)
    return {'gpu': p.name, 'total_memory_bytes': p.total_memory, 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'capability': list(torch.cuda.get_device_capability()),
        'dependencies': {n: importlib.metadata.version(n) for n in ('transformers', 'peft', 'accelerate')}}


def model_fingerprint(model_name, revision):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Immutable model commit required')
    p = Path(model_name)
    if not p.is_dir(): return {'model': model_name, 'revision': revision}
    lock=read_json(Path(__file__).resolve().parents[2]/'configs/qwen3_4b_snapshot.json')
    if revision!=lock['hf_revision']:raise ValueError('Local snapshot must match the admitted Qwen file lock')
    verified={}
    for entry in lock['files']:
        f=p/entry['name'];print('VERIFY BASE '+entry['name'],flush=True)
        if not f.is_file() or f.stat().st_size!=entry['size']:raise ValueError('Missing or wrong-sized base file: '+entry['name'])
        digest=sha256(f)
        if digest!=entry['sha256']:raise ValueError('Base checksum mismatch: '+entry['name'])
        verified[f.name]=digest
    return {'model': 'Qwen/Qwen3-4B-Instruct-2507', 'revision': revision,
        'files':verified,'hash_provenance':lock['hash_provenance']}


def _base(name, revision, download=False, quantized=False):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kw = {}
    if quantized:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, local_files_only=not download,
        trust_remote_code=False, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', **kw)
    model.config.use_cache = False
    return model


def _lora(model, quantized=False):
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    if quantized: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules='all-linear', task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def _tokenizer(name, revision, download=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=not download, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok


def _contract(records, name, revision, length, quantized, excluded):
    audit_knowledge(records, set(excluded))
    if {r['split'] for r in records} != {'train', 'validation'}: raise ValueError('Train/validation families required')
    if any(not r.get('source_locator') or not r.get('review_basis') for r in records):
        raise ValueError('Source locator and curation basis required, not just reviewed=true')
    return {'base': model_fingerprint(name, revision), 'corpus_hash': object_hash(records),
        'max_length': length, 'quantized': quantized, 'lora': {'r':16, 'alpha':32, 'dropout':.05},
        'environment': require_gpu(), 'completion_only_loss': True, 'base_adapter': 'new',
        'code_hash': sha256(__file__)}


def smoke_knowledge(records, run_dir, model_name, revision, max_length=2048, allow_download=False,
                    quantized=False, excluded_families=(),seed=42,learning_rate=1e-4):
    output = Path(run_dir)
    if output.exists(): raise FileExistsError('New smoke output required')
    contract = _contract(records, model_name, revision, max_length, quantized, excluded_families)
    contract['optimizer_options']={'seed':seed,'learning_rate':learning_rate}
    tok = _tokenizer(model_name, revision, allow_download); torch.manual_seed(seed)
    r = next(r for r in records if r['split'] == 'train')
    batch = {k: torch.tensor([v], device='cuda') for k,v in completion_example(tok,r['prompt'],r['completion'],max_length).items()}
    model = _lora(_base(model_name,revision,allow_download,quantized),quantized)
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().float().cpu().clone() for p in params]
    opt = torch.optim.AdamW(params,lr=learning_rate); losses=[]; torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        model.train(); opt.zero_grad(); loss=model(**batch).loss
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach()))
    updated=any(not torch.equal(a,p.detach().float().cpu()) for a,p in zip(before,params))
    if not updated: raise RuntimeError('No adapter parameter update')
    model.eval();inputs={k:v for k,v in batch.items() if k!='labels'}
    with torch.inference_mode(): logits=model(**inputs).logits.float().cpu()
    model.save_pretrained(output/'adapter',safe_serialization=True);tok.save_pretrained(output/'adapter')
    peak=torch.cuda.max_memory_allocated();trainable=sum(p.numel() for p in params)
    del params,before,opt,model;gc.collect();torch.cuda.empty_cache()
    from peft import PeftModel
    model=PeftModel.from_pretrained(_base(model_name,revision,allow_download,quantized),str(output/'adapter'),is_trainable=False).eval()
    with torch.inference_mode(): again=model(**inputs).logits.float().cpu()
    delta=float((again-logits).abs().max())
    if delta>1e-3: raise RuntimeError(f'Save/reload logits differ: {delta}')
    write_json(output/'status.json',{'status':'passed','signature':object_hash(contract),'contract':contract,
        'steps':2,'losses':losses,'adapter_updated':updated,'reload_max_abs_delta':delta,
        'peak_cuda_bytes':peak,'trainable_parameters':trainable,'formal_training':False})


def fit_knowledge(records, run_dir, model_name, revision, max_steps=40, max_length=2048,
                  resume=False, allow_download=False, epochs=None, quantized=True, smoke_dir=None, excluded_families=(),
                  seed=42,learning_rate=1e-4):
    contract=_contract(records,model_name,revision,max_length,quantized,excluded_families)
    contract['optimizer_options']={'seed':seed,'learning_rate':learning_rate}
    if smoke_dir is None: raise ValueError('Passed actual GPU smoke required before formal training')
    smoke=read_json(Path(smoke_dir)/'status.json')
    if smoke.get('status')!='passed' or smoke.get('signature')!=object_hash(contract): raise ValueError('GPU smoke contract differs')
    if epochs is not None and epochs not in {1,2,3}: raise ValueError('Declared epoch budget must be 1, 2 or 3')
    from transformers import Trainer,TrainingArguments
    from datasets import Dataset
    output=Path(run_dir)
    if output.exists() and any(output.iterdir()) and not resume: raise FileExistsError('New run or explicit resume required')
    output.mkdir(parents=True,exist_ok=True)
    contract['budget']={'epochs':epochs,'max_steps':None if epochs else max_steps}
    if resume and read_json(output/'contract.json')!=contract: raise ValueError('Training contract changed')
    write_json(output/'contract.json',contract)
    tok=_tokenizer(model_name,revision,allow_download);torch.manual_seed(seed)
    prepared={s:[completion_example(tok,r['prompt'],r['completion'],max_length) for r in records if r['split']==s] for s in ('train','validation')}
    def collate(examples):
        width=max(len(x['input_ids']) for x in examples)
        return {k:torch.tensor([x[k]+[pad]*(width-len(x[k])) for x in examples]) for k,pad in [('input_ids',tok.pad_token_id),('attention_mask',0),('labels',-100)]}
    model=_lora(_base(model_name,revision,allow_download,quantized),quantized)
    schedule={'num_train_epochs':epochs,'max_steps':-1,'eval_strategy':'epoch','save_strategy':'epoch'} if epochs else {
        'max_steps':max_steps,'eval_strategy':'steps','eval_steps':min(20,max_steps),'save_strategy':'steps','save_steps':min(20,max_steps)}
    args=TrainingArguments(output_dir=str(output/'checkpoints'),per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,learning_rate=learning_rate,bf16=True,logging_steps=1,save_total_limit=max(3,epochs or 0),
        load_best_model_at_end=True,report_to=[],seed=seed,data_seed=seed,remove_unused_columns=False,dataloader_num_workers=0,**schedule)
    trainer=Trainer(model=model,args=args,data_collator=collate,train_dataset=Dataset.from_list(prepared['train']),eval_dataset=Dataset.from_list(prepared['validation']))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(output/'adapter'));tok.save_pretrained(output/'adapter');trainer.save_state()
    if epochs and float(trainer.state.epoch or 0)<epochs-.001: raise RuntimeError('Requested epoch incomplete')
    write_json(output/'training_log.json',trainer.state.log_history)
    write_json(output/'status.json',{'training':'completed','epochs_completed':trainer.state.epoch,
        'optimizer_steps':trainer.state.global_step,'examples':{k:len(v) for k,v in prepared.items()},
        'best_checkpoint':trainer.state.best_model_checkpoint,'checkpoint_selector':'knowledge_validation_token_loss',
        'effective_completion_tokens_per_epoch':{k:sum(sum(t!=-100 for t in r['labels']) for r in rows) for k,rows in prepared.items()},
        'learning_rate':learning_rate,'seed':seed,
        'science_status':'unvalidated','new_process_evaluation':'pending','scope':'small_source_curated_extraction_pilot_not_all_dormancy_knowledge'})


def evaluate_knowledge(records, output, model_name, revision, adapter=None, allow_download=False, evidence_modes=False):
    require_gpu();tok=_tokenizer(model_name,revision,allow_download);model=_base(model_name,revision,allow_download)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.eval();rows=[]
    import copy
    from .knowledge import retrieve
    cases=[]
    for original in records:
        if original['split']!='validation':continue
        cases.append((original,'given_evidence'))
        if evidence_modes:
            for mode in ('no_evidence','fixed_retrieval'):
                r=copy.deepcopy(original);user=r['prompt'][-1]['content']
                question=user.rsplit('Question:',1)[-1]
                evidence=[] if mode=='no_evidence' else retrieve(records,question,{original['study_family']},limit=3)
                r['prompt']=[{'role':'system','content':'Use only supplied evidence. Return JSON with answer, context, source, uncertain. If the requested source-specific conclusion has no supporting evidence, answer unknown and uncertain true.'},
                    {'role':'user','content':user.split('Evidence',1)[0]+'Evidence: '+('None supplied.' if not evidence else '\n'.join(e['source_ref']+': '+e['text'] for e in evidence))+'\nQuestion: '+question}]
                # Exact held-out source is excluded from retrieval. This tests appropriate abstention,
                # not a claim that related training facts answer a new study-specific question.
                expected=json.loads(r['completion']);expected.update(answer='unknown',uncertain=True)
                r['completion']=json.dumps(expected);cases.append((r,mode))
    for r,mode in cases:
        prompt=tok.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=True)
        inputs=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
        with torch.inference_mode(): ids=model.generate(**inputs,max_new_tokens=192,do_sample=False,pad_token_id=tok.pad_token_id)
        answer=tok.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        try: parsed=json.loads(answer);valid=isinstance(parsed,dict)
        except (ValueError,TypeError):parsed={};valid=False
        if not valid:parsed={}
        expected=json.loads(r['completion'])
        rows.append({'record_id':r['record_id'],'evidence_mode':mode,'answer':answer,'expected':expected,'valid_json':valid,
            'exact_fields':{k:parsed.get(k)==expected[k] for k in ('source','context','uncertain')},
            'answer_exact':parsed.get('answer')==expected['answer']})
    if not rows:raise ValueError('No held-out questions')
    write_json(Path(output)/'answers.json',rows)
    if adapter:
        write_json(Path(output)/'reload.json',{'new_process':True,'status':'passed_actual_adapter_load_and_generation',
            'adapter_files':{p.name:sha256(p) for p in sorted(Path(adapter).iterdir()) if p.is_file()},
            'base_revision':revision})
    mode_scores={mode:{'n':len(items),'exact_answer_rate':sum(r['answer_exact'] for r in items)/len(items),'uncertain_accuracy':sum(r['exact_fields']['uncertain'] for r in items)/len(items)} for mode in sorted({r['evidence_mode'] for r in rows}) for items in [[r for r in rows if r['evidence_mode']==mode]]}
    rows=[r for r in rows if r['evidence_mode']=='given_evidence']
    write_json(Path(output)/'evaluation.json',{'by_evidence_mode':mode_scores,'weight_kind':'domain_adapter' if adapter else 'base','n':len(rows),
        'format_rate':sum(r['valid_json'] for r in rows)/len(rows),'exact_answer_rate':sum(r['answer_exact'] for r in rows)/len(rows),
        'exact_context_rate':sum(r['exact_fields']['context'] for r in rows)/len(rows),'exact_source_rate':sum(r['exact_fields']['source'] for r in rows)/len(rows),
        'uncertain_field_accuracy':sum(r['exact_fields']['uncertain'] for r in rows)/len(rows),
        'limitations':['Small family-held-out evidence extraction, not comprehensive knowledge validation',
        'Canonical text match undercounts paraphrases; saved answers require expert factual review',
        'Public papers may occur in base pretraining'],'science_status':'unvalidated'})
