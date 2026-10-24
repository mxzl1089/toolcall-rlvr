import pytest
from toolrl.data import write_json
from toolrl.evaluate import evaluate
from toolrl.runtime import init_run


@pytest.mark.parametrize('split',['test','official'])
def test_heldout_requires_lock(split,tmp_path):
    with pytest.raises(ValueError,match='lock'):
        evaluate({},tmp_path,tmp_path/'never_created',split=split)
    assert not (tmp_path/'never_created').exists()


@pytest.mark.parametrize('kwargs',[{'limit':1},{'probe':True}])
def test_heldout_cannot_be_subset(kwargs,tmp_path):
    lock=tmp_path/'lock.json';write_json(lock,{'config':{},'adapters':[{}]})
    with pytest.raises(ValueError,match='subset'):
        evaluate({},tmp_path,tmp_path/'never_created',split='test',lock=lock,**kwargs)
    assert not (tmp_path/'never_created').exists()


def test_existing_run_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):init_run(tmp_path,{}, {})
