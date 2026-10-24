import itertools
import pytest
from toolrl.contract_cases import CASES, REFERENCE, render
from toolrl.rewards_train import score

@pytest.mark.parametrize('case',CASES,ids=[c['name'] for c in CASES])
def test_training_rewards(case):
    for mode in ('sparse','dense'):
        r=score(case['prediction'],REFERENCE,mode)
        assert -3<=r['total']<=4
        assert (r['correctness']==3)==case['multiset_exact']

@pytest.mark.parametrize('v',[True,1.0,'1'])
def test_value_types(v):
    p=render([{'name':'x','parameters':{'v':v}}])
    g=render([{'name':'x','parameters':{'v':1}}])
    assert score(p,g,'dense')['correctness']<3
    assert score(p,g,'sparse')['correctness']==-3

def test_assignment_invariance():
    g=[{'name':'x','parameters':{'a':1,'b':2}}, {'name':'x','parameters':{'a':1,'b':3}}]
    p=[{'name':'x','parameters':{'a':1,'b':3}}, {'name':'x','parameters':{'a':1}}]
    scores={score(render(list(x)),render(list(y)),'dense')['correctness'] for x in itertools.permutations(p) for y in itertools.permutations(g)}
    assert len(scores)==1
    assert next(iter(scores))==pytest.approx(-3+6*5.5/7)

def test_empty_parameters_and_duplicates():
    g=render([{'name':'x','parameters':{}}])
    assert score(g,g,'dense')['correctness']==3
    assert score(render([{'name':'x','parameters':{}}]*2),g,'dense')['correctness']<3

@pytest.mark.parametrize('p',[
 '<tool_call>\n{"name":"bad","name":"x","parameters":{}}\n</tool_call>',
 '</tool_call><tool_call>',
 '<tool_call>\n{"name":"x","parameters":{"v":NaN}}\n</tool_call>',
])
def test_invalid_output(p):
    assert score(p,REFERENCE,'dense')['total']==-3

def test_unknown_mode():
    with pytest.raises(ValueError):score(REFERENCE,REFERENCE,'typo')
