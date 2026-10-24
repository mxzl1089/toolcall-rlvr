import pytest
from toolrl.eval_strict import evaluate_prediction

REF='<tool_call>\n{"name":"x","parameters":{"v":1}}\n</tool_call>'

@pytest.mark.parametrize('body,code',[
 ('{"name":"bad","name":"x","parameters":{"v":1}}','duplicate_json_key'),
 ('{"name":"x","parameters":{"v":0,"v":1}}','duplicate_json_key'),
 ('{"name":"x","parameters":{"v":{"a":1,"a":2}}}','duplicate_json_key'),
 ('{"name":"x","parameters":{"v":NaN}}','invalid_json_constant'),
 ('{"name":"x","parameters":{"v":Infinity}}','invalid_json_constant'),
 ('{"name":"x","parameters":{"v":-Infinity}}','invalid_json_constant'),
 ('{"name":"x","parameters":{"v":1e999}}','invalid_json_constant'),
])
def test_invalid_json_semantics(body,code):
    r=evaluate_prediction('<tool_call>\n'+body+'\n</tool_call>',REF)
    assert not r['call_multiset_exact']
    assert r['prediction_error']==code

def test_reversed_tags():
    assert evaluate_prediction('</tool_call><tool_call>',REF)['prediction_error']=='invalid_tool_call_order'

@pytest.mark.parametrize('v',['true','1.0','"1"'])
def test_types(v):
    assert not evaluate_prediction(REF.replace(':1}',':'+v+'}'),REF)['call_multiset_exact']

def test_nested_order():
    a=REF.replace('1}', '{"a":1,"b":[1,2]}}')
    b=REF.replace('1}', '{"b":[1,2],"a":1}')
    # Construct explicitly so bracket errors cannot mask the semantic assertion.
    a='<tool_call>\n{"name":"x","parameters":{"v":{"a":1,"b":[1,2]}}}\n</tool_call>'
    b='<tool_call>\n{"name":"x","parameters":{"v":{"b":[1,2],"a":1}}}\n</tool_call>'
    assert evaluate_prediction(a,b)['call_multiset_exact']
    assert not evaluate_prediction(a,b.replace('[1,2]','[2,1]'))['call_multiset_exact']
