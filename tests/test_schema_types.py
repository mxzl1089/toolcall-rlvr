import pytest
from toolrl.clean_manifest import type_matches

@pytest.mark.parametrize('v,t,expected',[
 (1,'int',True),(True,'int',False),('1','int',False),(1,'float',True),
 ('[1,2]','array',False),([1,2],'List[int]',True),([1,'2'],'List[int]',False),
 ([[1]],'List[List[int]]',True),([[1.1]],'List[List[int]]',False),
 ([1,2.0],'Tuple[float, float]',True),([1],'Tuple[float, float]',False),
 (True,'bool, optional',True),(None,'str, optional',False),
 ([1,2.0],'List[Union[int, float]]',True),('f','Callable[[float], float]',None),
])
def test_schema_types(v,t,expected):assert type_matches(v,t) is expected
