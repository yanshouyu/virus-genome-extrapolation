"""Regression for equal-locus strand-paired adaptation controls."""
from collections import defaultdict
from pathlib import Path
import pytest

def test_control_samples_loci_then_both_strands(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_adaptation_corpus import tiles,paired_sample,rc
    assert rc('AAAACCCG')=='CGGGTTTT'
    candidates=list(tiles('AAAACCCG'*1000,'example','species'))
    selected=paired_sample(candidates,3,42)
    assert selected==paired_sample(candidates,3,42)
    assert len(selected)==6
    loci=defaultdict(dict)
    for row in selected:loci[(row['accession'],row['start'],row['end'])][row['strand']]=row['sequence']
    assert len(loci)==3
    for strands in loci.values():
        assert set(strands)=={'+','-'}
        assert strands['-']==rc(strands['+'])
    with pytest.raises(ValueError):paired_sample(candidates,100,42)
