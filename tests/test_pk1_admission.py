import copy
import io
import json
import zipfile
import numpy as np
import pytest
from vdc.admission import approve_roles, role_manifest, validate_internal_row,audit_internal_task
from vdc.io import write_json, read_json,sha256,object_hash
from vdc.pk1_data import read_featurecounts, import_exit_counts


def draft(tmp_path):
    p=tmp_path/'roles.json'
    row={'sample_key':'fixture:a','biological_unit':'pool:a','link_ids':['fixture:a','pool:a'],
         'role':'development','split':'train','allowed_tasks':['state'],'cohort_id':'cohort:a'}
    write_json(p,{'protocol':'VDC_PK1','status':'draft','blocking_questions':[],'samples':[row]})
    return p


def test_approval_cannot_be_assumed_or_changed(tmp_path,monkeypatch):
    p=draft(tmp_path)
    with pytest.raises(ValueError,match='approval'):role_manifest(p)
    h=approve_roles(p,'software fixture reviewer')
    monkeypatch.setenv('VDC_ROLE_MANIFEST',str(p))
    row={'source_sample_key':'fixture:a','biological_unit':'pool:a','link_ids':['fixture:a','pool:a'],
         'admission_protocol':'VDC_PK1','admission_hash':h,'split':'train'}
    validate_internal_row(row)
    audit_internal_task([{**row,'origin':'internal'}],'state')
    with pytest.raises(ValueError,match='Task is not admitted'):
        audit_internal_task([{**row,'origin':'internal'}],'functional')
    with pytest.raises(ValueError,match='original biological unit'):
        validate_internal_row({**row,'biological_unit':'cell:1'})
    with pytest.raises(ValueError,match='links'):
        validate_internal_row({**row,'link_ids':['fixture:a']})
    policy=read_json(p);policy['samples'][0]['split']='validation';write_json(p,policy)
    with pytest.raises(ValueError,match='changed'):role_manifest(p)


def test_invalid_approval_does_not_write_approved_file(tmp_path):
    p=draft(tmp_path);q=read_json(p);r=copy.deepcopy(q['samples'][0]);r.update(sample_key='fixture:b',split='validation')
    q['samples'].append(r);write_json(p,q)
    with pytest.raises(ValueError,match='crosses'):approve_roles(p,'fixture')
    assert read_json(p)['status']=='draft'


def test_future_cohort_query_is_not_independent_validation(tmp_path):
    p=draft(tmp_path);q=read_json(p)
    q['samples'].append({'sample_key':'fixture:later','biological_unit':'pool:later','link_ids':['fixture:later'],
        'cohort_id':'cohort:a','parent_sample_key':'fixture:a','evaluation_scope':'within_cohort_future_only',
        'role':'temporal_query','split':'locked_test','allowed_tasks':['locate','within_cohort_future']})
    write_json(p,q);approve_roles(p,'fixture')
    q['status']='draft';q.pop('approval',None);q['samples'][1]['role']='development';q['samples'][1]['split']='validation'
    write_json(p,q)
    with pytest.raises(ValueError,match='cohort crosses'):approve_roles(p,'fixture')


def table(value='3', duplicate=False):
    return ('# featureCounts fixture\nGeneid\tChr\tStart\tEnd\tStrand\tLength\tfixture.bam\n'
            'geneA\tchr1\t1\t10\t+\t10\t'+value+'\n'+
            ('geneA' if duplicate else 'geneB')+'\tchr1\t20\t30\t+\t11\t2\n').encode()


@pytest.mark.parametrize('value',['-1','nan','1.2'])
def test_counts_require_nonnegative_integers(value):
    with pytest.raises(ValueError,match='integer'):read_featurecounts(io.BytesIO(table(value)))


def test_duplicate_genes_are_rejected():
    with pytest.raises(ValueError,match='duplicate'):read_featurecounts(io.BytesIO(table(duplicate=True)))


def test_exit_audit_preserves_excluded_library_without_admission(tmp_path):
    p=tmp_path/'input.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('featureCounts_results/18_counts.txt',table())
        z.writestr('featureCounts_results/1_counts.txt',table())
        z.writestr('featureCounts_results/1_counts.txt.summary','Status\tfixture.bam\nAssigned\t5\n')
    rows=import_exit_counts(p,tmp_path/'out',excluded_samples=['18'])
    assert next(r for r in rows if r['sample_id']=='18')['historical_exclusion']
    assert (tmp_path/'out/exit_counts_18.npz').exists()
    assert not (tmp_path/'out/sample_roles.json').exists()
    assert all(r['precise_elapsed_hours'] is None for r in rows)


def test_assigned_summary_must_match_counts(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('1_counts.txt',table())
        z.writestr('1_counts.txt.summary','Status\tfixture.bam\nAssigned\t7\n')
    with pytest.raises(ValueError,match='Assigned'):import_exit_counts(p,tmp_path/'out')


def test_prepare_bulk_selects_admitted_columns_before_numeric_parsing(tmp_path,monkeypatch):
    from vdc.pk1_data import prepare_killifish
    inp=tmp_path/'input';inp.mkdir()
    counts=inp/'bulk_counts.csv'
    counts.write_text('Geneid,a_sorted,b_sorted,reserved_sorted\ng1,1,2,DO_NOT_PARSE\ng2,4,2,DO_NOT_PARSE\ng3,2,4,DO_NOT_PARSE\n')
    lock={'files':{'bulk_counts.csv':sha256(counts)}};write_json(tmp_path/'source_lock.json',lock)
    rows=[]
    for sid,split in [('a','train'),('b','validation')]:
        rows.append({'sample_key':'bulk:'+sid,'source':'bulk','sample_id':sid,'biological_unit':'pool:'+sid,
            'link_ids':['bulk:'+sid],'role':'development','split':split,'condition':'fixture','allowed_tasks':['state']})
    p=tmp_path/'sample_roles.json';write_json(p,{'protocol':'VDC_PK1','status':'draft',
        'blocking_questions':[],'source_lock_hash':object_hash(lock),'samples':rows})
    approve_roles(p,'synthetic fixture');monkeypatch.setenv('VDC_ROLE_MANIFEST',str(p))
    ann=tmp_path/'go.tsv';ann.write_text('Gene stable ID\tGO term accession\n'+''.join(g+'\tGO:fixture\n' for g in ['g1','g2','g3']))
    b=prepare_killifish(tmp_path,tmp_path/'prepared',ann,['GO:fixture'],'bulk')
    assert len(b.rows)==2
    assert {r['source_sample_key'] for r in b.rows}=={'bulk:a','bulk:b'}
    counts.write_text(counts.read_text().replace('g1,1,2','g1,7,2'))
    with pytest.raises(ValueError,match='Private input changed'):
        prepare_killifish(tmp_path,tmp_path/'changed',ann,['GO:fixture'],'bulk')
