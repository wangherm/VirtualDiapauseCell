"""Train-unit marker references, nested rejection calibration, and evidence-only cards."""
import numpy as np
from .clock_wave import hidden_partition
from .metrics import study_unit_weights
from .io import object_hash


def fit_reference(y, genes, labels, rows, markers_per_class=24, min_units=2):
    if any(r['split'] != 'train' for r in rows): raise ValueError('Identity fitting requires train rows only')
    classes = sorted(t for t in set(labels) if t != 'unknown' and
                     len({r['biological_unit'] for r,l in zip(rows,labels) if l==t}) >= min_units)
    if len(classes) < 2: return None
    labels = np.asarray(labels); weights = study_unit_weights(rows)
    hidden = hidden_partition(genes); keep = np.array([g not in hidden for g in genes])
    mean = np.average(y, axis=0, weights=weights)
    sd = np.maximum(np.sqrt(np.average((y-mean)**2, axis=0, weights=weights)), .1)
    markers = {}; centres = []
    for t in classes:
        ii = labels==t; centre = np.average(y[ii], axis=0, weights=weights[ii]); centres.append(centre)
        other = np.average(y[~ii], axis=0, weights=weights[~ii]); effect = (centre-other)/sd
        idx = np.flatnonzero(keep & (effect>0))
        markers[t] = idx[np.argsort(-effect[idx], kind='stable')[:markers_per_class]].tolist()
    indices = sorted(set(i for v in markers.values() for i in v))
    if len(indices)<3: return None
    return {'classes':classes, 'indices':indices, 'mean':mean[indices].tolist(), 'scale':sd[indices].tolist(),
            'centres':((np.array(centres)[:,indices]-mean[indices])/sd[indices]).tolist(),
            'markers':{t:[genes[i] for i in v] for t,v in markers.items()},
            'fit_ids':[r['observation_id'] for r in rows], 'gold':'source annotation coarse grouping; silver reference'}


def distances(reference, y):
    z = (y[:,reference['indices']]-np.array(reference['mean']))/reference['scale']
    return np.mean((z[:,None]-np.array(reference['centres'])[None])**2, axis=2)


def calibrate(y, genes, labels, rows, config):
    """Marker selection is refitted inside EACH inner unit fold, not on its held rows."""
    records=[]; units=sorted({r['biological_unit'] for r in rows})
    for unit in units:
        # Cohort and material closure; these rows are already outer-training only.
        selected={i for i,r in enumerate(rows) if r['biological_unit']==unit}
        def links(r): return set(r['link_ids']+[r['biological_unit']]+([r['cohort_id']] if r.get('cohort_id') else []))
        while True:
            merged=set().union(*(links(rows[i]) for i in selected))
            new={i for i,r in enumerate(rows) if links(r)&merged}
            if new==selected:break
            selected=new
        va=sorted(selected); tr=[i for i in range(len(rows)) if i not in selected]
        ref=fit_reference(y[tr],genes,[labels[i] for i in tr],[rows[i] for i in tr],config['identity_markers_per_class'],config['identity_min_units'])
        if ref is None:continue
        dd=distances(ref,y[va]);order=np.argsort(dd,axis=1,kind='stable')
        for j,i in enumerate(va):
            winner=ref['classes'][order[j,0]]
            records.append({'unit':rows[i]['biological_unit'],'truth':labels[i],'prediction':winner,
                            'distance':float(dd[j,order[j,0]]),'margin':float(dd[j,order[j,1]]-dd[j,order[j,0]])})
    good=[r for r in records if r['prediction']==r['truth']]
    calibrated=len({r['unit'] for r in good})>=2
    return {'status':'calibrated_on_inner_development_silver' if calibrated else 'unsupported_insufficient_inner_unit_matches',
            'max_distance':float(np.quantile([r['distance'] for r in good],.95)) if calibrated else None,
            'min_margin':float(np.quantile([r['margin'] for r in good],.05)) if calibrated else None,
            'inner_predictions':records, 'probability_calibration':False}


def predict(reference, calibration, y):
    if reference is None:
        return [{'prediction':'unknown','reason':'insufficient_reference_units','candidates':[]} for _ in y]
    dd=distances(reference,y); result=[]
    for d in dd:
        order=np.argsort(d,kind='stable'); accepted=(calibration['max_distance'] is not None and d[order[0]]<=calibration['max_distance'] and d[order[1]]-d[order[0]]>=calibration['min_margin'])
        result.append({'prediction':reference['classes'][order[0]] if accepted else 'unknown',
                       'reason':'numeric_gate_passed' if accepted else 'numeric_gate_rejected',
                       'candidates':[{'identity':reference['classes'][j],'distance':float(d[j])} for j in order[:4]]})
    return result


def evidence_cards(reference, predictions, y, genes, rows, symbols):
    """No query identity, condition, time, clock, targets, or outer test result in cards."""
    cards=[]; indices=reference['indices'] if reference else []
    for i,r in enumerate(rows):
        top=sorted(indices,key=lambda j:-y[i,j])[:24]
        candidates=[v['identity'] for v in predictions[i]['candidates']]
        markers={t:[{'gene':g,'symbol':symbols.get(g,'')} for g in reference['markers'][t][:16]] for t in candidates} if reference else {}
        cards.append({'query_id':object_hash(r['observation_id'])[:20],
                      'observed_markers':[{'gene':genes[j],'symbol':symbols.get(genes[j],''),'relative_log_expression':float(y[i,j])} for j in top],
                      'reference_markers':markers,'numeric_candidates':predictions[i]['candidates'],
                      'numeric_gate':predictions[i]['prediction'],
                      'instructions':'Choose a listed candidate or unknown from this evidence only. Do not infer lineage, clock or time. Missing biological meaning in gene IDs is insufficient evidence.'})
    return cards


def score(predictions, truth, rows):
    pred=[r['prediction'] if isinstance(r,dict) else r for r in predictions]; labels=sorted(set(truth)-{'unknown'})
    weights=study_unit_weights(rows).astype(float); weights/=weights.sum()
    accepted=np.array([p!='unknown' for p in pred]);correct=np.array([a==b for a,b in zip(pred,truth)])
    per={}
    for t in labels:
        actual=np.array([v==t for v in truth]);emit=np.array([v==t for v in pred])
        tp=weights[actual&emit].sum();den=weights[actual].sum()+weights[emit].sum()
        per[t]={'recall':float(tp/weights[actual].sum()),'f1':float(2*tp/den) if den else 0.}
    return {'coverage':float(weights[accepted].sum()),'accepted_error':float(weights[accepted&~correct].sum()/weights[accepted].sum()) if accepted.any() else None,
            'macro_f1':float(np.mean([r['f1'] for r in per.values()])) if per else None,'by_class':per,
            'annotation_agreement_not_independent_identity_gold':True}


def parse_answer(text, card):
    import json
    try:
        answer=json.loads(text); choice=answer['identity']; evidence=answer['evidence_genes']
        candidates={r['identity'] for r in card['numeric_candidates']}|{'unknown'}
        allowed={r['gene'] for r in card['observed_markers']}
        if choice not in candidates or not isinstance(evidence,list) or any(g not in allowed for g in evidence):raise ValueError('Unsupported candidate/evidence')
        if choice!='unknown' and not evidence:raise ValueError('Evidence required for acceptance')
        # First round is conservative corroboration. LLM cannot override failed numeric support.
        pred=choice if choice==card['numeric_gate'] else 'unknown'
        return {'prediction':pred,'llm_choice':choice,'format_valid':True,'reason':'corroborated' if pred!='unknown' else 'unknown_or_numeric_conflict','evidence_genes':evidence}
    except (ValueError,KeyError,TypeError):
        return {'prediction':'unknown','format_valid':False,'reason':'invalid_structured_answer'}
