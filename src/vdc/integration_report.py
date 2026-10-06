"""Render the one computed result, never recompute model formulas in the UI."""
from pathlib import Path
import csv,html,json
from .io import write_json,write_jsonl


def export_result(r,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New request output directory required')
    out.mkdir(parents=True);write_json(out/'result.json',r);write_json(out/'provenance.json',r['provenance'])
    traces=[x['qwen'] for x in r['identity'] if 'trace' in x['qwen']]+r['knowledge']
    write_json(out/'qwen_trace.json',traces)
    write_jsonl(out/'evidence.jsonl',[e for k in r['knowledge'] for e in k['retrieval']['records']])
    tables={k:[] for k in ('samples','programmes','gene_predictions','waves','identity_evidence','response_predictions')}
    n=r.get('numeric');lines=['# Virtual Diapause Cell · INT1',f"Request: {r['request_id']}",
        f"Execution: {r['execution_mode']} / {r['status']}; numerical variant: {r['model_variant']}",
        'Experimental reference analysis. Clock is not elapsed time, recovery probability or functional depth.']
    if n:
        for i,s in enumerate(r['samples']):
            sid=s['sample_id'];tables['samples'].append({**s,'clock':n['clock'][i],'support_status':n['status'][i],'route':n['route']})
            lines.append(f"{sid}: clock={n['clock'][i]}; {n['status'][i]}; identity={s['identity']} ({s['identity_source']}).")
            p=len(n['programme_ids'])
            for channel in range(2):
                for j,fid in enumerate(n['programme_ids']):
                    k=channel*p+j
                    tables['programmes'].append({'sample_id':sid,'programme':fid,'channel':n['feature_channels'][channel],
                        **{key:n[key][i][k] for key in ('observed','expected','residual','linear_extrapolation','mask','coverage')}})
            for j,gene in enumerate(n['hidden_gene_ids']):
                tables['gene_predictions'].append({'sample_id':sid,'gene':gene,'prediction':n['gene_prediction'][i][j],
                    'observed_full_library_log':n['hidden_gene_measurement'][i][j],'matched_ridge':n['direct_ridge'][i][j],
                    'cw_stage_baseline':r['baseline']['gene_prediction'][i][j],'status':n['status'][i]})
            w=n['waves']
            for j,gene in enumerate(w['genes']):tables['waves'].append({'sample_id':sid,'gene':gene,'TF_RNA':gene in w['TF_RNA_genes'],
                'status':n['status'][i],**{key:w[key][i][j] for key in ('observed','expected','linear_extrapolation','residual')}})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,3.5));good=[i for i,v in enumerate(n['clock']) if v is not None]
        ax.scatter(good,[n['clock'][i] for i in good],c=['#24759c' if n['status'][i]=='located' else '#c06830' for i in good])
        ax.set(xlabel='Request row (not biological time)',ylabel='Unclipped reference coordinate',title='Identity-specific references; values are not a shared physiological scale')
        fig.tight_layout();(out/'figures').mkdir();fig.savefig(out/'figures/clock.png',dpi=140);plt.close(fig)
    else:(out/'figures').mkdir()
    for identity in r['identity']:
        q=identity['qwen'];tables['identity_evidence'].append({'sample_id':identity['sample_id'],'actual_identity':identity['actual_identity'],
            'source':identity['source'],'qwen_result':json.dumps(q.get('result',q),ensure_ascii=False)})
    for q in r['responses']:tables['response_predictions'].append({'module':q['module'],'status':q['status'],'result':json.dumps(q,ensure_ascii=False)})
    (out/'tables').mkdir()
    for name,rows in tables.items():
        columns=list(dict.fromkeys(k for row in rows for k in row)) or ['status','reason']
        with (out/'tables'/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    lines+=['','Qwen outputs are source-linked interpretations requiring review, not validated mechanisms.',
            json.dumps(r['components'],ensure_ascii=False,indent=2)]
    for k in r['knowledge']:lines.append(json.dumps(k['explanation'],ensure_ascii=False,indent=2))
    (out/'summary.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    def table(rows):
        if not rows:return '<p>No applicable output; inspect component status.</p>'
        keys=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in keys)+'</tr>' for row in rows[:200])+'</table>'
    body='<h1>Virtual Diapause Cell</h1><pre>'+html.escape('\n\n'.join(lines))+'</pre>'
    if n:body+='<img src="figures/clock.png" alt="Clock reference positions"><h2>Samples</h2>'+table(tables['samples'])
    for name in ('programmes','gene_predictions','identity_evidence','response_predictions'):
        body+='<h2>'+html.escape(name)+'</h2><p>First 200 rows; full data in <a href="tables/'+name+'.csv">CSV</a>.</p>'+table(tables[name])
    for k in r['knowledge']:
        body+='<h2>Retrieved source evidence</h2>'+table(k['retrieval']['records'])
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>VDC INT1 analysis</title><style>body{font:15px system-ui;margin:32px;color:#182630}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd;max-width:600px;overflow-wrap:anywhere}pre{white-space:pre-wrap}img{max-width:100%}</style>'+body,encoding='utf-8')
