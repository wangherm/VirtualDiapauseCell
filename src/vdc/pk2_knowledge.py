"""Source-audited metadata supervision complements, never replaces, mechanistic evidence."""
from pathlib import Path
import json
from .io import read_jsonl,read_json,write_json,write_jsonl,sha256,object_hash
from .knowledge import audit_knowledge

def corpus(acquired,original,out):
    from .pk2_data import verify_acquisition
    verify_acquisition(acquired,['GSE202844','GSE221467','GSE303716','GSE124109','GSE104616','GSE3169'])
    literature=Path(__file__).resolve().parents[2]/'knowledge/pk2/literature.jsonl'
    records=read_jsonl(original)+read_jsonl(literature);seen=set();sources=[]
    # Source-family assignment fixed before question construction. No numerical outcomes.
    splits={'GSE202844':'train','GSE221467':'train','GSE303716':'train','GSE124109':'validation','GSE104616':'train','GSE3169':'validation'}
    for study,split in splits.items():
        path=Path(acquired)/study/'samples.jsonl';rows=read_jsonl(path);count=0
        for sample in rows:
            f=sample['fields'];species=f['Sample_organism_ch1'][0];facts=f.get('Sample_characteristics_ch1',[])
            # Gene expression, outcome labels and private observations never enter the corpus.
            pairs=[('Which species was assayed?',species)]+[(f'What is the recorded {x.split(": ",1)[0]}?',x.split(': ',1)[1]) for x in facts if ': ' in x]
            if len(pairs)<2:continue
            context=species+'; '+f.get('Sample_source_name_ch1',f['Sample_title'])[0]
            evidence='Species: '+species+'. '+'; '.join(facts)
            for question,answer in pairs+[('Does this sample metadata establish a numerical functional depth?','unknown')]:
                identity=object_hash([study,context,evidence,question,answer])
                if identity in seen:continue
                seen.add(identity);uncertain=answer=='unknown'
                records.append({'record_id':'GEO_'+identity[:20],'study_family':study,'source_ref':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+study,
                    'source_locator':sample['accession']+' Sample_organism/characteristics','reviewed':True,
                    'review_basis':'Deterministic field-to-answer comparison with locked GEO metadata; source consistency audit, not independent expert review',
                    'split':split,'kind':'experimental_evidence','evidence_type':'observational','text':evidence,
                    'label_source':'public_metadata_weak_reference_not_mechanistic_outcome','prompt':[
                        {'role':'system','content':'Use only the supplied evidence. Return JSON with answer, context, source, uncertain. If a requested conclusion is unsupported, answer unknown and uncertain true. Do not invent molecular clock or functional depth.'},
                        {'role':'user','content':f'Source: {study}\nContext: {context}\nEvidence: {evidence}\nQuestion: {question}'}],
                    'completion':json.dumps({'answer':answer,'context':context,'source':study,'uncertain':uncertain},ensure_ascii=False)})
                count+=1
        sources.append({'family':study,'split':split,'metadata_sha256':sha256(path),'new_records':count})
    audit=audit_knowledge(records);write_jsonl(Path(out)/'corpus.jsonl',records)
    write_json(Path(out)/'manifest.json',{'audit':audit,'sources':sources,'corpus_hash':object_hash(records),
        'counts':{s:sum(r['split']==s for r in records) for s in ['train','validation']},
        'original_records':len(read_jsonl(original)),'new_records':len(records)-len(read_jsonl(original)),
        'literature':read_json(literature.parent/'sources.json'),
        'target_30_50_paper_families_reached':False,'target_800_1500_records_reached':len(records)>=800,
        'limitations':['Literature supervision is primary-abstract weak reference; metadata extraction is recorded separately',
            'GEO source families are not automatically independent paper families; no count of biological replicates is inferred from question count',
            'No Qwen-generated gold; original mechanistic pilot retained separately; full literature target remains incomplete']})
    if len(records)<=len(read_jsonl(original)):raise ValueError('Expanded corpus contains no new reviewed records')
    return records

def objects(features,root):
    go={r['id']:r for r in read_json(Path(root)/'knowledge/snapshots/go_2026-10-01/quickgo_response.json')['results']}
    return [{'record_id':fid,'object_id':fid,'study_family':'GO_20261001','source_ref':'https://www.ebi.ac.uk/QuickGO/term/'+fid,
        'split':'train','reviewed':True,'kind':'object_description','text':go[fid]['name']+'. '+go[fid]['definition']['text']} for fid in features]
