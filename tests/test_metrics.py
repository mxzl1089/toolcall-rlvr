from toolrl.contract_cases import render
from toolrl.metrics import diagnostics

def test_missing_and_extra_keys():
    g=render([{'name':'x','parameters':{'a':1,'b':2}}])
    p=render([{'name':'x','parameters':{'a':1,'c':3}}])
    d=diagnostics(p,g)
    assert (d['key_tp'],d['key_fp'],d['key_fn'],d['value_correct'],d['value_total'])==(1,1,1,1,2)
    assert d['error_types']==['extra_parameter','missing_parameter']

def test_wrong_name_counts_unmatched_parameters():
    g=render([{'name':'x','parameters':{'a':1}}]);p=g.replace('"x"','"y"')
    d=diagnostics(p,g)
    assert (d['key_tp'],d['key_fp'],d['key_fn'])==(0,1,1)

def test_duplicate_and_order():
    a={'name':'x','parameters':{'a':1}};b={'name':'y','parameters':{}}
    assert diagnostics(render([b,a]),render([a,b]))['error_types']==[]
    assert diagnostics(render([a,a]),render([a]))['duplicate_call']
