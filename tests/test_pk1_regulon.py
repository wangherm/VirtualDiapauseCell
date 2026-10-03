import gzip
from vdc.regulon import celest_definitions


def test_mapping_rejects_ambiguous_alias_and_preserves_unsigned_membership(tmp_path):
    gaf=tmp_path/'fixture.gaf.gz'
    with gzip.open(gaf,'wt') as f:
        for gene,alias in [('WBGeneTF','TF.1'),('WBGene1','gene.1'),('WBGene2','gene.2'),('WBGene3','gene.3'),('WBGene4','ambiguous'),('WBGene5','ambiguous')]:
            row=['WB',gene,gene,'','GO:fixture','fixture','EXP','','P','',''+alias,'protein','taxon:6239','20261003','fixture','','']
            f.write('\t'.join(row)+'\n')
    net=tmp_path/'network.tsv';net.write_text('source\ttarget\tweight\n'+''.join('TF.1\t'+t+'\t0.5\n' for t in ['gene.1','gene.2','gene.3','ambiguous','TF.1']))
    defs,audit=celest_definitions(net,gaf,['WBGene1','WBGene2','WBGene3'])
    assert len(defs)==1 and set(defs[0]['members'].values())=={1.}
    assert defs[0]['sign']=='unknown'
    assert audit['unresolved_or_ambiguous_edges']==1 and audit['self_edges_removed']==1
