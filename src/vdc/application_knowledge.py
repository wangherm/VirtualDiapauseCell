"""Evaluation-only protocol correction; never alters old labels or trains an adapter."""
import copy,json
from .io import object_hash


def clean_cases(records):
    from .knowledge_train import evidence_cases
    selected=[];facts={}
    for original in records:
        if original['split']!='validation':continue
        expected=json.loads(original['completion']);question=original['prompt'][-1]['content'].rsplit('Question:',1)[-1].strip()
        key=object_hash([original['study_family'],question,expected['answer'],expected['uncertain']])
        if key in facts:facts[key].append(original['record_id']);continue
        facts[key]=[original['record_id']];r=copy.deepcopy(original);r['fact_id']=key;selected.append(r)
    cases=evidence_cases(selected,balanced=True);out=[];seen={}
    for r,mode in cases:
        r=copy.deepcopy(r)
        if mode=='addressed_withheld':
            question=r['prompt'][-1]['content'].rsplit('Question:',1)[-1].strip()
            # Source and Context are removed too. No covert species/source answer remains.
            r['prompt']=[{'role':'system','content':'Use only supplied evidence. Return JSON: answer, context, source, uncertain. When source or context is absent, write not_supplied for that field. An unsupported study-specific answer must be unknown with uncertain true.'},
                         {'role':'user','content':'Source: not_supplied\nContext: not_supplied\nEvidence: None supplied.\nQuestion: For the unspecified experimental record, '+question}]
            r['completion']=json.dumps({'answer':'unknown','context':'not_supplied','source':'not_supplied','uncertain':True})
        identity=object_hash([mode,r['prompt']])
        if identity in seen:
            seen[identity]['contributing_families'].append(r['study_family']);continue
        r['evaluation_inputs'].update(protocol='all_visible_fields_v2',fact_id=r.get('fact_id'),
            fact_duplicates=facts.get(r.get('fact_id'),[]),contributing_families=[r['study_family']],
            source_or_context_allowed_as_evidence=True,completion_used_to_construct_visible_input=False)
        seen[identity]=r['evaluation_inputs'];out.append((r,mode))
    for r,mode in out:
        if mode=='addressed_withheld' and len(set(r['evaluation_inputs']['contributing_families']))>1:
            r['study_family']='shared_withholding_control';r['usage']='shared_withholding_control'
    return out


def source_independence(old,new):
    old_sources={r['source_ref'].rstrip('/') for r in old};new_sources={r['source_ref'].rstrip('/') for r in new}
    if old_sources&new_sources:raise ValueError('New evaluation paper appears in adapter corpus')
    if any(r.get('usage')!='evaluation_only' for r in new):raise ValueError('New papers must remain evaluation-only')
    return {'new_source_count':len(new_sources),'adapter_corpus_overlap':False,
        'independence':'not in this adapter SFT/selection corpus; base-model pretraining exposure unknown',
        'expert_adjudication':False,'labels':'primary-source-curated weak reference'}
