"""CW1 targeted revision tasks, dispatched by the original clock-wave runner."""
from pathlib import Path
from collections import Counter
import numpy as np
from .io import read_json, read_jsonl, write_json, save_npz, sha256, object_hash
from .clock_wave import ClockWave, identities, inputs, weighted_mean, NotIdentifiable
from .clock_wave_data import load_data
from .pk2_numeric import dual_ridge, metrics, dynamic_ratio
from .application import predict_ridge
from .observation import normalise_expression
from . import identity, identity_protocol as protocol


def qwen_inference(run, mode, out, config, model, tok, jobs):
    import torch
    provenance=read_json(out/'provenance.json')
    contract=config.get('identity_protocol','short_slots_v2');fixed=contract=='fixed_evidence_v3'
    provenance.update(contract=contract,decision_rule='original: corroborate numeric gate; no_numeric/challenge: independent evidence assay, never used by the chain',
                      evidence_validation_scope='unique positive-expression slots from this query; not independent validation of marker biology')
    write_json(out/'provenance.json',provenance)
    budget=config['qwen_max_new_tokens']; cache={}
    def generate(card, condition):
        public,mapping=protocol.short_card(card,condition); text=(protocol.fixed_prompt if fixed else protocol.prompt)(public)
        tokens=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_tensors='pt')
        record={'query_id':card['query_id'],'condition':condition,'prompt':text,'mapping':mapping,
                'prompt_tokens':int(tokens.shape[1]),'max_new_tokens':budget}
        if tokens.shape[1]>config['qwen_max_input_tokens']:
            return {**record,'prediction':'unknown','identity_status':'generation_failure','reason':'input_over_budget_not_truncated',
                    'json_valid':False,'schema_valid':False,'evidence_valid':None,'format_valid':False,
                    'raw_answer':None,'response_tokens':0,'eos_observed':None,'stop_observation':'not_generated'}
        tokens=tokens.to('cuda')
        with torch.inference_mode():
            answer=model.generate(tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=budget,
                                  do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True)
        ids=answer[0,tokens.shape[1]:].tolist(); raw=tok.decode(ids,skip_special_tokens=True)
        return {**record,**(protocol.parse_fixed if fixed else protocol.parse)(raw,mapping),'raw_answer':raw,
                **protocol.stop_record(ids,model.generation_config.eos_token_id,budget)}
    # Deterministic engineering gate. No labels, clock results or answer repairs used.
    first=next(j for j in jobs if j['id']=='identity_original')
    preflight=[]
    for card in read_jsonl(run/'tasks'/first['id']/'cards.jsonl')[:3]:
        response=generate(card,'original');preflight.append(response);cache[card['query_id']]=response
        write_json(out/'preflight.json',{'responses':preflight,'policy':'all three must have valid JSON/schema/current-card evidence; acceptance not required'})
    valid_preflight=len(preflight)==3 and all(r['format_valid'] for r in preflight)
    if not valid_preflight and not fixed:
        raise RuntimeError('Short-contract engineering preflight failed; preserved raw responses; independent CPU tasks continue')
    if fixed:
        write_json(out/'preflight.json',{'responses':preflight,'contract':contract,
                   'format_check':'passed' if valid_preflight else 'failed_collect_full_fixed_budget_evaluation',
                   'policy':'schema/evidence failures remain failures; complete prespecified evaluation, no repairs or biological gate changes'})
    reports=[]; counts=Counter()
    for spec in jobs:
        folder=run/'tasks'/spec['id'];cards=read_jsonl(folder/'cards.jsonl')
        for condition in protocol.CONDITIONS:
            responses=[]
            for card in cards:
                response=cache[card['query_id']] if spec==first and condition=='original' and card['query_id'] in cache else generate(card,condition)
                responses.append(response);counts[response['identity_status']]+=1
                # Persist partial progress too; a generation crash must not erase completed evidence.
                write_json(out/(spec['id']+('' if condition=='original' else '_'+condition)+'.json'),{'responses':responses,'complete':False})
            ev=read_json(folder/'evaluation.json')
            metric=identity.score(responses,ev['truth'],ev['rows'])
            result={'responses':responses,'metrics':metric,'complete':True,'condition':condition,
                    'status_counts':dict(Counter(r['identity_status'] for r in responses)),
                    'cards_sha256':sha256(folder/'cards.jsonl'),
                    'scope':'development silver annotation agreement; challenges are manipulated evidence, not biological experiments'}
            write_json(out/(spec['id']+('' if condition=='original' else '_'+condition)+'.json'),result)
            reports.append({'task':spec['id'],'condition':condition,'metrics':metric,'status_counts':result['status_counts']})
            print('SHORT_IDENTITY',mode,spec['id'],condition,len(responses),flush=True)
    write_json(out/'result.json',{'status':'evaluated','mode':mode,'folds':reports,'status_counts':dict(counts),
               'new_training':False,'generation_count':sum(counts.values()),'historical_answers_reparsed':False,
               'preflight_reused_without_duplicate_generation':3,'contract':contract,'preflight_valid':valid_preflight})


def residual_features(model, counts, types):
    q=model.predict(counts,types)
    valid=q['input_mask'] & model.arrays['valid'][None,:] & (q['coverage']>=model.meta['config']['minimum_coverage'])
    valid &= np.isfinite(q['expected_linear_extrapolation'])
    residual=np.where(valid,q['observed']-q['expected_linear_extrapolation'],0.)
    design=np.column_stack([model.clock_design(q['clock'],types),residual,valid])
    return design,q


def residual_task(run, view, out, config):
    x,g,defs,rows=load_data(run/'data'/view)
    model=ClockWave.load(run/'tasks'/('numeric_'+view+'_C2')/'model');types=identities(rows)
    features,q=residual_features(model,x,types)
    tr=np.array([i for i,r in enumerate(rows) if r['observation_id'] in model.meta['readout_fit_ids']])
    if any(rows[i]['split']!='train' for i in tr):raise ValueError('Readout fit IDs contain nontraining rows')
    va=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);rr=[rows[i] for i in va]
    if len(tr)<3:raise NotIdentifiable('Fewer than three finite training readout rows')
    target=normalise_expression(x,np.ones_like(x,bool),'counts')[:,model.meta['target_indices']]
    fn=dual_ridge(features[tr],target[tr],[rows[i] for i in tr],config['readout_alpha'])
    save_npz(out/'readout.npz',**fn.parameters)
    finite=np.isfinite(q['clock'][va]);pred=np.full_like(target[va],np.nan)
    pred[finite]=predict_ridge(out/'readout.npz',features[va][finite])
    np.testing.assert_allclose(pred[finite],fn(features[va][finite]),atol=0,rtol=0)
    old=model.hidden_predictions(x[va],np.array(types)[va])
    predictions={'identity_mean':old['hidden_identity_mean'],'clock_identity':old['hidden_prediction'],
                 'clock_identity_residual':pred,'direct_programme_ridge':old['hidden_direct_ridge']}
    # Exact same hidden panel, all four models share masks for comparison.
    common=np.logical_and.reduce([np.isfinite(v) for v in predictions.values()])
    located=np.array([s=='located' for s in old['status']])[:,None]
    masks={'in_reference':common&located,'extrapolation_only':common&~located,'all_finite':common}
    result={scope:{name:{'metric':metrics(v,target[va],mask,rr),'dynamic':dynamic_ratio(v,target[va],mask)}
                   for name,v in predictions.items()} for scope,mask in masks.items()}
    changed=x.copy();changed[:,[i for i,s in enumerate(g) if s in model.meta['hidden_removed']]]=999999
    other,oq=residual_features(model,changed,types)
    np.testing.assert_allclose(features,other,atol=0,rtol=0,equal_nan=True)
    np.testing.assert_allclose(q['clock'],oq['clock'],atol=0,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(model._features(x)[0],model._features(changed)[0])
    # Saved original direct baseline has a potentially broader training set; add a matched-fit check.
    _,xx,mm,_=model._features(x);dx=model.direct_features(xx,mm,types)
    direct=dual_ridge(dx[tr],target[tr],[rows[i] for i in tr],config['readout_alpha'])
    save_npz(out/'matched_direct.npz',**direct.parameters)
    matched=direct(dx[va]);np.testing.assert_allclose(matched,predict_ridge(out/'matched_direct.npz',dx[va]),atol=0,rtol=0)
    result['matched_fit_direct']={scope:metrics(matched,target[va],mask,rr) for scope,mask in masks.items()}
    save_npz(out/'validation/predictions.npz',**predictions,hidden_target=target[va],clock=q['clock'][va],
             matched_fit_direct=matched,common_in_reference=masks['in_reference'],visible_residual_design=features[va])
    write_json(out/'validation/rows.json',rr);write_json(out/'validation/query_status.json',{'status':old['status'],'identity':np.array(types)[va].tolist()})
    write_json(out/'validation/metrics.json',result)
    write_json(out/'result.json',{'status':'evaluated','metrics':result,'new_readout_fit':True,'clock_refitted':False,
               'readout_alpha':config['readout_alpha'],
               'target_genes':model.meta['target_genes'],'fit_ids':[rows[i]['observation_id'] for i in tr],
               'frozen_model_sha256':sha256(run/'tasks'/('numeric_'+view+'_C2')/'model/model.json'),
               'residual_training':'in-sample training-only frozen reference; not cross-fitted; alpha fixed before evaluation',
               'direct_baseline_fit_ids':model.meta['fit_ids'],'matched_direct_fit_ids':[rows[i]['observation_id'] for i in tr],
               'coverage_all_queries':float(np.mean(located)),'reload':'passed','hidden_input_invariance':'passed',
               'target_isolation_scope':model.meta['target_isolation_scope']})


def support_task(run, view, out, config):
    x,g,defs,rows=load_data(run/'data'/view);types=identities(rows)
    model=ClockWave.load(run/'tasks'/('numeric_'+view+'_C2')/'model');q=model.predict(x,types)
    groups={}
    for i,r in enumerate(rows):
        if r['split']!='validation':continue
        key=(r['condition'],types[i]);groups.setdefault(key,[]).append(i)
    table=[]
    for (condition,typ),ids in sorted(groups.items()):
        table.append({'condition':condition,'identity':typ,'profiles':len(ids),
                      'units':len({rows[i]['biological_unit'] for i in ids}),
                      'status_counts':dict(Counter(q['status'][i] for i in ids)),
                      'late_maintenance_not_exit':condition.lower().startswith('late')})
    # Training-unit anchor bootstrap on the FROZEN axis genes/scales. It never changes the deployed locator/range.
    y,_,_=inputs(x,g,defs,model.meta['config'].get('extra_hidden',[]));sensitivity={};rng=np.random.default_rng(42106)
    for typ,axis in model.meta['axes'].items():
        if axis['status']!='exploratory_reference':
            sensitivity[typ]={'status':axis['status']};continue
        early=[i for i,r in enumerate(rows) if r['split']=='train' and types[i]==typ and r['condition']=='Early Diapause']
        end=[i for i,r in enumerate(rows) if r['split']=='train' and types[i]==typ and r['condition'] in {'Developing','Developing Day 7'}]
        sets=[early,end];unitsets=[sorted({rows[i]['biological_unit'] for i in ids}) for ids in sets]
        va=[i for i,r in enumerate(rows) if r['split']=='validation' and types[i]==typ]
        z=(y[:,axis['indices']]-axis['mean'])/axis['scale'];draws=[];degenerate=0
        for _ in range(config['anchor_bootstrap_draws']):
            centres=[]
            for ids,units in zip(sets,unitsets):
                sampled=rng.choice(units,len(units),replace=True)
                # Preserve equal biological-unit weights, with replacement multiplicity.
                centres.append(np.mean([z[[i for i in ids if rows[i]['biological_unit']==u]].mean(0) for u in sampled],axis=0))
            delta=centres[1]-centres[0];norm=delta@delta
            if norm<1e-8:degenerate+=1;continue
            draws.append((z[va]-centres[0])@delta/norm)
        save_npz(out/('anchor_'+str(len(sensitivity))+'.npz'),coordinates=np.array(draws),validation_indices=np.array(va,dtype=int))
        sensitivity[typ]={'status':'conditional_axis_sensitivity_only','train_anchor_unit_counts':[len(u) for u in unitsets],
                         'insufficient_replication_for_uncertainty':any(len(u)<2 for u in unitsets),
                         'draws':len(draws),'degenerate_draws':degenerate,'fit_ids':axis['fit_ids'],
                         'validation_ids':[rows[i]['observation_id'] for i in va],
                         'conditional_on':'frozen train-selected genes and scales; not full model uncertainty or independent calibration'}
    a=ClockWave.load(run/'tasks'/('numeric_'+view+'_C0')/'model');b=ClockWave.load(run/'tasks'/('numeric_'+view+'_C1')/'model')
    _,xx,_,_=a._features(x);_,yy,_,_=b._features(x)
    delta=float(np.max(np.abs((xx-b.arrays['early_rank'])-yy)))
    if delta>1e-12:raise ValueError('C1 is no longer the declared fixed translation')
    save_npz(out/'rank_translation.npz',C0=xx,C1=yy,early_offset=b.arrays['early_rank'])
    write_json(out/'result.json',{'status':'evaluated','support_by_condition_identity':table,'anchor_sensitivity':sensitivity,
               'range_changed':False,'C1_translation_max_abs_delta':delta,'C1_new_information':False,
               'clock_scope':'each identity separately calibrated; not a common physiological scale',
               'late_maintenance':'reported separately even if inside range; no normal Exit interpretation'})
