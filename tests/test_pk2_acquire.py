"""Transport and format fixtures only; no synthetic biological result claims."""
import gzip,hashlib,io,tarfile
import numpy as np
import pytest
from scripts import prepare_public_pilot as downloader
from vdc.pk2_acquire import family_metadata,numeric_table,tet_loci
from vdc.io import read_json

@pytest.mark.parametrize('status',[200,206])
def test_resume_validates_range_or_explicitly_restarts(tmp_path,monkeypatch,status):
    data=b'abcdefgh';entry={'file':'f.gz','url':'https://example.org/f','bytes':8,'sha256':hashlib.sha256(data).hexdigest(),'resume':True}
    (tmp_path/'f.gz.partial').write_bytes(data[:3])
    class Response(io.BytesIO):pass
    def open_url(req,timeout):
        assert req.get_header('Range')=='bytes=3-'
        r=Response(data[3:] if status==206 else data);r.status=status;r.headers={'Content-Range':'bytes 3-7/8'};return r
    monkeypatch.setattr(downloader.urllib.request,'urlopen',open_url)
    assert downloader.fetch_locked(entry,tmp_path,True).read_bytes()==data

def test_wrong_range_never_appended(tmp_path,monkeypatch):
    (tmp_path/'f.partial').write_bytes(b'ab')
    class Response(io.BytesIO):status=206;headers={'Content-Range':'bytes 0-3/4'}
    monkeypatch.setattr(downloader.urllib.request,'urlopen',lambda *a,**k:Response(b'abcd'))
    with pytest.raises(ValueError,match='Content-Range'):
        downloader.fetch_locked({'file':'f','url':'https://example.org/f','bytes':4,'resume':True,'sha256':hashlib.sha256(b'abcd').hexdigest()},tmp_path,True)
    assert (tmp_path/'f.partial').read_bytes()==b'ab' and not (tmp_path/'f').exists()

def test_differential_table_and_fractional_counts_rejected(tmp_path):
    path=tmp_path/'table.gz'
    with gzip.open(path,'wt') as f:f.write('gene\tbaseMean\tlog2FoldChange\tpadj\ng1\t3\t1\t0.05\n')
    with pytest.raises(ValueError,match='Differential'):numeric_table(path,tmp_path/'out','counts',[])
    with gzip.open(path,'wt') as f:f.write('gene\tS1\ng1\t1.5\n')
    with pytest.raises(ValueError,match='integers'):numeric_table(path,tmp_path/'out','counts',[])

def test_exact_sample_mapping_and_ercc_retained(tmp_path):
    path=tmp_path/'table.gz'
    with gzip.open(path,'wt') as f:f.write('gene\tX1\tX2\ng1\t3\t0\nERCC-1\t2\t1\n')
    samples=[{'accession':'GSM2','fields':{'Sample_description':['X2']}},{'accession':'GSM1','fields':{'Sample_description':['X1']}}]
    numeric_table(path,tmp_path/'out','counts',samples,'description')
    m=read_json(tmp_path/'out/expression.json');assert m['sample_accessions']==['GSM1','GSM2'] and m['ERCC_feature_count']==1
    with pytest.raises(ValueError,match='Unresolved'):numeric_table(path,tmp_path/'bad','counts',samples[:1],'description')

def test_soft_stream_selects_only_dauer_and_preserves_negative_values(tmp_path):
    p=tmp_path/'family.gz';seen=[]
    with gzip.open(p,'wt') as f:f.write('^SAMPLE = GSM1\n!Sample_title = Dauer MTC#1T0\n!sample_table_begin\nID_REF\tVALUE\np1\t-1.5\np2\tnull\n!sample_table_end\n^SAMPLE = GSM2\n!Sample_title = L1 MTC#1T0\n!sample_table_begin\nID_REF\tVALUE\np1\t2\n!sample_table_end\n')
    rows=family_metadata(p,lambda s,ids,values:seen.append((s,ids,values)))
    assert len(rows)==2 and len(seen)==1 and seen[0][2][0]==-1.5 and np.isnan(seen[0][2][1])

def test_tet_locus_union_never_silently_sums_duplicate_genes(tmp_path):
    p=tmp_path/'raw.tar';header='#chr\tstart\tend\tgene_name\tgene_ID\tstrand\tTPM\n'
    content=['chr1\t0\t1\tx\tgene1.1\t+\t1\nchr2\t0\t1\tx\tgene1.1\t+\t2\n','chr1\t0\t1\tx\tgene1.1\t+\t3\n']
    with tarfile.open(p,'w') as tar:
        for i,body in enumerate(content,1):
            raw=gzip.compress((header+body).encode());entry=tarfile.TarInfo(f'GSM{i}_x_TPM.bed.gz');entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
    result=tet_loci(p,tmp_path/'out',[{'accession':'GSM1'},{'accession':'GSM2'}])
    assert result['loci']==2 and not result['training_admitted']
    with np.load(tmp_path/'out/locus_expression.npz') as a:assert a['mask'].sum()==3 and not a['mask'][1,1]
    assert read_json(tmp_path/'out/locus_expression.json')['duplicate_gene_ids']['GSM1']=={'gene1.1':2}
