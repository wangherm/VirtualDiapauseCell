"""Versioned external target-expression proxies, not binding or causal TF activity."""
import csv
import gzip
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import sha256, write_json, save_npz, object_hash
from .observation import score_programmes
from .waves import WaveReference
from .alpha_numeric import gene_data,metric
from .contracts import ObservationBundle


def celest_definitions(network,gaf,genes,min_targets=3):
    aliases=defaultdict(set)
    with gzip.open(gaf,'rt',encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'):continue
            c=line.rstrip('\n').split('\t')
            if len(c)!=17:raise ValueError('Expected GAF 2.2')
            if c[0]!='WB' or c[12]!='taxon:6239':continue
            for alias in [c[1],c[2]]+c[10].split('|'):
                if alias:aliases[alias].add(c[1])
    observed=set(genes);members=defaultdict(dict);stats=defaultdict(int)
    with Path(network).open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            stats['input_edges']+=1
            a,b=aliases.get(r['source'],set()),aliases.get(r['target'],set())
            if len(a)!=1 or len(b)!=1:stats['unresolved_or_ambiguous_edges']+=1;continue
            tf,target=next(iter(a)),next(iter(b));w=float(r['weight'])
            if not np.isfinite(w) or w<=0:raise ValueError('Expected positive edge confidence, not a signed regulatory effect')
            if target==tf:stats['self_edges_removed']+=1;continue
            stats['mapped_edges']+=1
            # Confidence is not an effect sign. Use a fixed unsigned membership proxy.
            members[tf][target]=1.
    source='CelEsT:'+sha256(network)+';WB_GAF:'+sha256(gaf)
    definitions=[{'id':'CelEsT:'+tf,'tf_gene_id':tf,'members':dict(sorted(targets.items())),
                  'source_ref':source,'sign':'unknown','scoring':'unsigned_target_expression_proxy'}
                 for tf,targets in sorted(members.items())
                 if len(observed.intersection(targets))>=min_targets and len(observed.intersection(targets))/len(targets)>=.5]
    if not definitions:raise ValueError('No regulons pass ID/coverage checks')
    return definitions,{**stats,'regulons':len(definitions),'alias_ambiguities':sum(len(s)>1 for s in aliases.values()),
        'network_sha256':sha256(network),'gaf_sha256':sha256(gaf),'effect_sign_inferred':False,
        'external_network_training_study_overlap':'not_fully_established; no independent mechanistic-validation claim'}


def public_regulon_task(public_run,network,gaf,out):
    root=Path(public_run);out=Path(out)
    b=ObservationBundle.load(root/'clock_reference/bundle');x,genes=gene_data(root/'data/dauer')
    definitions,audit=celest_definitions(network,gaf,genes)
    scored=score_programmes(x,np.ones_like(x,bool),genes,definitions)
    if not scored['mask'].all():raise ValueError('Regulon observation coverage mismatch')
    ids=[d['id'] for d in definitions];context=object_hash({'context':b.context,'network':audit['network_sha256']})
    model=WaveReference.fit(b.clock,scored['values'],scored['mask'],ids,b.rows,
        b.clock_reference_id,context,feature_kind='regulon',degree=1)
    model.save(out/'waves');reloaded=WaveReference.load(out/'waves')
    tr,va=b.indices('train'),b.indices('validation')
    result=reloaded.residual(b.clock[va],scored['values'][va],scored['mask'][va],b.clock_reference_id,context)
    baseline=np.broadcast_to(scored['values'][tr].mean(0),scored['values'][va].shape)
    original=model.predict(b.clock[va],b.clock_reference_id,context)['expected']
    if not np.allclose(original,result['expected'],equal_nan=True):raise RuntimeError('Regulon save/reload failed')
    write_json(out/'members.json',definitions);write_json(out/'mapping_audit.json',audit)
    save_npz(out/'observations.npz',values=scored['values'],mask=scored['mask'],coverage=scored['coverage'],feature_ids=np.array(ids))
    save_npz(out/'validation.npz',observed=scored['values'][va],expected=result['expected'],
             residual=result['residual'],supported=result['residual_mask'],baseline=baseline)
    report={'execution_status':'completed','science_status':'unvalidated','regulons':len(ids),
        'wave':metric(result['expected'],scored['values'][va],result['residual_mask']),
        'train_mean_same_support':metric(baseline,scored['values'][va],result['residual_mask']),
        'support_fraction':float(result['residual_mask'].mean()),'save_reload':'passed',
        'fit_observation_ids':[b.rows[i]['observation_id'] for i in tr],
        'internal_killifish_used':False,'test_used':False,
        'interpretation':'Fixed external unsigned regulon target-expression proxy; not TF activity truth or causal perturbation validation'}
    write_json(out/'result.json',report)
    return report
