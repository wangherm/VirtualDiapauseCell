"""One bounded read of approved raw development counts for observation diagnostics."""
from pathlib import Path
import csv,copy
import numpy as np
from .admission import role_manifest
from .contracts import ObservationBundle
from .state import StatePredictor
from .observation import normalise_expression,score_programmes
from .alpha_numeric import gene_data
from .io import read_json,write_json,save_npz
from .pk2_numeric import query,metrics

def observation_stress(run,out,c):
    run=Path(run);out=Path(out);private=Path(c['private_root']);base=run/'tasks';policy,roles=role_manifest();reports=[];arrays={}
    b=ObservationBundle.load(base/'input_bulk/bundle');v=b.subset(b.indices('validation'));p=StatePredictor.load(base/'adapt_R0_bulk_42/model')
    expressions,genes=gene_data(private/'prepared/bulk');expression=expressions[b.indices('validation')];definitions=read_json(private/'prepared/bulk/programmes.json')
    chosen=[roles[r['source_sample_key']] for r in v.rows];columns={};raw=np.zeros((len(chosen),len(genes)),dtype=np.int64)
    with (private/'input/bulk_counts.csv').open(encoding='utf-8-sig') as f:
        reader=csv.reader(f);head=next(reader)
        columns={i:head.index(r['sample_id']+'_sorted') for i,r in enumerate(chosen) if r['source']=='bulk'}
        for j,row in enumerate(reader):
            if genes[j]!=row[0]:raise ValueError('Stress count gene order differs')
            for i,k in columns.items():
                value=float(row[k])
                if value<0 or value!=int(value):raise ValueError('Count thinning requires actual integer counts')
                raw[i,j]=int(value)
    exit_genes=read_json(private/'input/single_exit_gene_ids.json');index={g:i for i,g in enumerate(exit_genes)};order=[index[g] for g in genes]
    for i,r in enumerate(chosen):
        if r['source']=='single_exit':
            with np.load(private/'input'/('exit_counts_'+r['sample_id']+'.npz'),allow_pickle=False) as a:raw[i]=a['counts'][order]
    def assess(tag,x,mask,bundle,model,defs,ids):
        s=score_programmes(x,mask,ids,defs);indices=[i for i in range(len(x)) if s['mask'][i].any()]
        if not indices:
            reports.append({'case':tag,'status':'unavailable_all_programmes_missing','coverage':0.});return
        q=bundle.subset(indices);q.values=s['values'][indices];q.mask=s['mask'][indices];q.coverage=s['coverage'][indices]
        result=query(model,q)['programme'];truth=bundle.values[indices];valid=bundle.mask[indices]
        reports.append({'case':tag,'model':metrics(result,truth,valid,q.rows),'input_coverage':float(s['mask'].mean()),
            'row_coverage':len(indices)/len(bundle.rows),'not_independent_biological_observations':True})
        arrays[tag+'_prediction']=result;arrays[tag+'_indices']=np.array(indices)
    rng=np.random.default_rng(774109)
    for fraction in [1.,.5,.25,.1]:
        for repeat in range(5):
            sampled=rng.binomial(raw,fraction);y=normalise_expression(sampled,np.ones_like(sampled,bool),'counts')
            assess(f'counts_{fraction}_{repeat}',y,np.ones_like(y,bool),v,p,definitions,genes)
    for fraction in [.1,.3,.5]:
        mask=rng.random(expression.shape)>=fraction
        assess(f'gene_missing_{fraction}',expression,mask,v,p,definitions,genes)
    # Load only approved validation cell rows once; never load a dense master matrix.
    import h5py
    from .pk1_data import h5_column
    cb=ObservationBundle.load(base/'input_core/bundle');cv=cb.subset(cb.indices('validation'));cp=StatePredictor.load(base/'adapt_R0_core_pool_42/model')
    coredefs=read_json(private/'prepared/core/programmes.json');cx=[]
    with h5py.File(private/'input/core.h5ad','r') as f:
        names=h5_column(f['obs/sample']);types=h5_column(f['obs/cell_type']);cgenes=[str(g) for g in h5_column(f['var/_index'])]
        layer=f['layers/counts'];ptr=layer['indptr'][:]
        if layer.attrs.get('encoding-type')!='csr_matrix':raise ValueError('Expected genuine CSR count layer')
        for r in cv.rows:
            entry=roles[r['source_sample_key']];idx=np.flatnonzero(names==entry['sample_id']);cells=[]
            for i in idx:
                vals=layer['data'][ptr[i]:ptr[i+1]];cols=layer['indices'][ptr[i]:ptr[i+1]]
                if (vals<0).any() or not np.allclose(vals,np.round(vals)):raise ValueError('Noninteger single-cell count layer')
                cells.append((str(types[i]),cols,vals))
            cx.append(cells)
    for fraction in [.5,.25]:
        for composition in ['type_proportion_fixed','type_composition_changed']:
            sampled=[]
            for cells in cx:
                celltypes=sorted({x[0] for x in cells});chosen_indices=[]
                if composition=='type_proportion_fixed':
                    for t in celltypes:
                        ids=[i for i,cell in enumerate(cells) if cell[0]==t];chosen_indices.extend(rng.choice(ids,max(1,int(len(ids)*fraction)),replace=False).tolist())
                else:
                    weights=np.array([3. if cell[0] in celltypes[:max(1,len(celltypes)//2)] else 1. for cell in cells]);weights/=weights.sum()
                    chosen_indices=rng.choice(len(cells),max(1,int(len(cells)*fraction)),replace=False,p=weights).tolist()
                total=np.zeros(len(cgenes),np.int64)
                for i in chosen_indices:np.add.at(total,cells[i][1],cells[i][2].astype(np.int64))
                sampled.append(total)
            sampled=np.array(sampled);y=normalise_expression(sampled,np.ones_like(sampled,bool),'counts')
            assess(f'cell_{fraction}_{composition}',y,np.ones_like(y,bool),cv,cp,coredefs,cgenes)
    save_npz(out/'predictions.npz',**arrays)
    write_json(out/'result.json',{'cases':reports,'reserved_used':False,'models':['adapt_R0_bulk_42','adapt_R0_core_pool_42'],
        'scope':'simulated observation degradation on fixed models and development samples only',
        'count_thinning':'actual integer count matrix; no inverse log/count fabrication',
        'cell_sampling':'per permitted validation pool; type-preserving sampling compared to explicitly type-biased sampling',
        'heldout_biological_samples_added':0})
