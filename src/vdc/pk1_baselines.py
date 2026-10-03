"""Stronger train-only controls alongside the existing public response fits."""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .io import read_json,write_json,save_npz
from .alpha_numeric import response_task,response_dataset,metric


def response_with_controls(public,mode,out):
    response_task(public,mode,output=out)
    d=response_dataset(public,mode);tr=np.array([i for i,r in enumerate(d.rows) if r['split']=='train']);va=np.array([i for i,r in enumerate(d.rows) if r['split']=='validation'])
    features=d.action if mode=='endpoint' else np.column_stack([d.action,d.elapsed])
    predictions=[];donors=[]
    for i in va:
        matches=[j for j in tr if np.array_equal(features[j],features[i])]
        if not matches:raise ValueError('Matched-condition baseline unavailable; no implicit global-mean replacement')
        predictions.append(d.target[matches].mean(0));donors.append([d.rows[j]['observation_id'] for j in matches])
    out=Path(out)
    save_npz(out/'condition_mean.npz',predicted=np.array(predictions),target=d.target[va])
    m=read_json(out/'result.json');m['train_condition_mean']=metric(np.array(predictions),d.target[va],d.target_mask[va]);m['condition_mean_donors']=donors
    m['condition_mean_scope']='seen-condition validation; not prediction of new interventions'
    write_json(out/'result.json',m)


def bounded_functional(source,out):
    source=read_json(source);records=source['records'];out=Path(out)
    if any(r['split'] not in {'train','validation'} for r in records):raise ValueError('Functional development only')
    tr=np.array([i for i,r in enumerate(records) if r['split']=='train']);va=np.array([i for i,r in enumerate(records) if r['split']=='validation'])
    y=np.array([r['counts']['young_adult'] for r in records],float);n=np.array([r['total'] for r in records],float)
    if (n<=0).any() or (y<0).any() or (y>n).any():raise ValueError('Invalid binomial counts')
    hours=np.array([r['hours_after_release'] for r in records],float);history=np.log1p([r['history_days'] for r in records])
    levels=sorted(set(hours[tr]));mu=history[tr].mean();scale=max(history[tr].std(),.001)
    if not set(hours[va]).issubset(levels):raise ValueError('Unseen functional hour; declared categorical baseline cannot extrapolate')
    design=np.column_stack([np.ones(len(n))]+[(hours==h).astype(float) for h in levels[1:]]+[(history-mu)/scale])
    def loss(beta):
        logits=design[tr]@beta
        return float(np.sum(n[tr]*np.logaddexp(0,logits)-y[tr]*logits)+.5*np.sum(beta[1:]**2))
    result=minimize(loss,np.zeros(design.shape[1]),method='BFGS')
    if not result.success and np.linalg.norm(result.jac)>1e-3:raise RuntimeError('Binomial fit did not converge')
    pred=expit(design[va]@result.x)
    baseline=np.array([y[tr[hours[tr]==hours[i]]].sum()/n[tr[hours[tr]==hours[i]]].sum() for i in va])
    save_npz(out/'model.npz',coefficients=result.x,hour_levels=np.array(levels),history_mean=np.array(mu),history_scale=np.array(scale))
    save_npz(out/'validation.npz',predicted=pred,observed=y[va]/n[va],time_only=baseline,successes=y[va],totals=n[va])
    write_json(out/'result.json',{'model':metric(pred,y[va]/n[va]),'train_hour_binomial_baseline':metric(baseline,y[va]/n[va]),
        'bounded_by_model':True,'posthoc_clipping':False,'expression_to_depth':'unavailable_no_matched_cohorts',
        'endpoint':source['endpoint'],'protocol':source['protocol'],'science_status':'unvalidated',
        'fit_kind':'regularised_binomial_likelihood','optimizer_success':bool(result.success),'gradient_norm':float(np.linalg.norm(result.jac))})
