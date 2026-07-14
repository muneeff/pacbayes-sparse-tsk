import numpy as np,pytest
from pacbayes_tsk.data.splits_v3 import ratio_split,official_horizon_split,assert_role_subset

def test_ratio():
 s=ratio_split(2000,{'prior':.2,'bound':.45,'validation':.15,'test':.2}); assert s.counts=={'prior':400,'bound':900,'validation':300,'test':400}
def test_official():
 s=official_horizon_split(1000,horizon=48); assert s.counts['validation']==48 and s.counts['test']==48 and s.counts['prior']+s.counts['bound']==904
def test_leakage_guard():
 s=ratio_split(100,{'prior':.2,'bound':.4,'validation':.2,'test':.2})
 with pytest.raises(ValueError): assert_role_subset(np.array([0,99]),s,{'prior'},name='prior_builder')
