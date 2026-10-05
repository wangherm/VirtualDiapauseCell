"""Stream real development cells once for fixed/changed composition stress."""
from pathlib import Path
import hashlib
import numpy as np
from .pk1_data import h5_column
from .io import read_json, write_json, sha256
from .clock_wave_data import load_data
from .clock_wave import ClockWave, evaluate


def sample_cells(path, sample_ids, genes, repeats=5, fractions=(.5,.25)):
    import h5py
    from scipy.sparse import csr_matrix
    with h5py.File(path,'r') as f:
        names=h5_column(f['obs/sample']);types=h5_column(f['obs/cell_type'])
        if h5_column(f['var/_index']).tolist()!=genes:raise ValueError('Cell stress gene universe differs')
        trials=[];selections=[]
        for fraction in fractions:
            for repeat in range(repeats):
                seed=451000+int(fraction*100)*10+repeat
                for mode in ('fixed_composition','shifted_composition'):
                    rng=np.random.default_rng(seed);chosen=np.zeros(len(names),bool);sizes={};composition={}
                    for sample in sample_ids:
                        idx=np.flatnonzero(names==sample)
                        if not len(idx):raise ValueError('Development sample absent from cell source')
                        labels=sorted(set(types[idx]));groups=[idx[types[idx]==t] for t in labels]
                        total=int(np.ceil(len(idx)*fraction))
                        if mode=='fixed_composition':
                            n=np.array([len(g)*fraction for g in groups]);allocation=np.floor(n).astype(int)
                            order=np.argsort(-(n-allocation),kind='stable');allocation[order[:total-allocation.sum()]]+=1
                            selected=np.concatenate([rng.choice(g,k,replace=False) for g,k in zip(groups,allocation)])
                        else:
                            # A deterministic relative enrichment stress, never a labelled intervention.
                            label_weights={t:2. if j%2==0 else .5 for j,t in enumerate(labels)}
                            probability=np.array([label_weights[t] for t in types[idx]],float);probability/=probability.sum()
                            selected=rng.choice(idx,total,replace=False,p=probability)
                        chosen[selected]=True;sizes[sample]=len(selected)
                        composition[sample]={str(t):int(np.sum(types[selected]==t)) for t in labels}
                    trials.append({'mode':mode,'fraction':fraction,'seed':seed,'cells_by_sample':sizes,
                                   'composition':composition,'cell_selection_sha256':hashlib.sha256(chosen.tobytes()).hexdigest()})
                    selections.append(chosen)
        select=np.array(selections);output=np.zeros((len(trials),len(sample_ids),len(genes)),dtype=np.float64)
        matrix=f['layers/counts']
        if matrix.attrs.get('encoding-type')!='csr_matrix':raise ValueError('Explicit CSR counts required')
        ptr=matrix['indptr'][:]
        for start in range(0,len(names),1024):
            end=min(start+1024,len(names));lo,hi=int(ptr[start]),int(ptr[end])
            values=matrix['data'][lo:hi]
            if not np.isfinite(values).all() or (values<0).any() or not np.equal(values,np.rint(values)).all():raise ValueError('Raw integer cells required')
            chunk=csr_matrix((values,matrix['indices'][lo:hi],ptr[start:end+1]-lo),shape=(end-start,len(genes)))
            for j,sample in enumerate(sample_ids):
                weights=select[:,start:end] & (names[start:end]==sample)[None]
                if weights.any():output[:,j]+=np.asarray(weights.astype(float)@chunk)
            if end%8192==0 or end==len(names):print('CELL_STREAM',end,'/',len(names),flush=True)
        if not np.equal(output,np.rint(output)).all():raise ValueError('Count aggregation lost integer precision')
        return output.astype(np.int64),trials


def task(run,out,config,private):
    run,out,private=Path(run),Path(out),Path(private)
    counts,genes,_,rows=load_data(run/'data/core');indices=[i for i,r in enumerate(rows) if r['split']=='validation']
    query=[rows[i] for i in indices];policy=read_json(run/'data/sample_roles.json')
    by_key={r['sample_key']:r for r in policy['samples']};sample_ids=[]
    for row in query:
        source=by_key[row['source_sample_key']]
        if source['role']!='development':raise ValueError('Cell pressure cannot query reserved source')
        sample_ids.append(source['sample_id'])
    path=private/'input/core.h5ad';audit=read_json(run/'data/audit.json')
    print('VERIFY_CELL_SOURCE',flush=True)
    if sha256(path)!=audit['source_files']['core.h5ad']:raise ValueError('Cell source changed')
    sampled,trials=sample_cells(path,sample_ids,genes,config['stress_repetitions'])
    model=ClockWave.load(run/'tasks/numeric_core_C2/model');base=model.predict(counts[indices],['all']*len(indices))['clock'];results=[]
    for number,(x,trial) in enumerate(zip(sampled,trials)):
        # Both cases are explicitly descriptive sampling stress, not independent biological truth.
        own=evaluate(model,x,query,out/f'trial_{number:02d}'/'sampled_observation')
        original=evaluate(model,x,query,out/f'trial_{number:02d}'/'original_observation',truth_counts=counts[indices])
        t=model.predict(x,['all']*len(x))['clock'];delta=t-base
        results.append({**trial,'versus_sampled_observation':own,'versus_original_observation':original,
                        'clock_change':[float(v) if np.isfinite(v) else None for v in delta]})
    write_json(out/'result.json',{'status':'evaluated','real_cell_subsampling':True,'new_fits':False,'experiments':results,
                'source_sha256':audit['source_files']['core.h5ad'],'independent_biological_experiments':False,
                'meaning':'Same pool, same total sampled cell budget for fixed/shifted composition; selected cells drawn without replacement. Differences are sampling stress, not causal intervention or clean ground truth.'})
