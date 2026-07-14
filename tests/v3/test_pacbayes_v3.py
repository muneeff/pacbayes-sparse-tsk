import numpy as np
from pacbayes_tsk.pac_bayes.priors_v3 import HierarchicalModelPrior, ModelIndex
from pacbayes_tsk.pac_bayes.certificates_v3 import gaussian_kl_diag, martingale_certificate, pointwise_gibbs_upper

def grid():
    return HierarchicalModelPrior(("ridge","dense_tsk","sparse_tsk"),(3,5),(0.5,1.0),(0.01,0.1),12)

def test_gamma_is_charged():
    g=grid()
    a=g.negative_log_mass(ModelIndex("sparse_tsk",3,.01,.5,2))
    b=g.negative_log_mass(ModelIndex("sparse_tsk",3,.1,.5,2))
    assert a==b and a>0

def test_family_is_charged():
    g=grid()
    assert g.negative_log_mass(ModelIndex("ridge",3,.01,None,1)) > 0
    assert g.negative_log_mass(ModelIndex("dense_tsk",3,.01,.5,2)) > 0

def test_kl_identity():
    assert gaussian_kl_diag([0,0],[1,1],[0,0],[1,1])==0

def test_certificate():
    r=martingale_certificate(empirical_risk=.1,total_kl=2,n=100,delta=.05,temperatures=[.25,.5,1])
    assert r.certificate>=.1 and r.certificate<=1

def test_pointwise_bounds():
    x=np.ones((3,2)); v=pointwise_gibbs_upper(x,np.zeros(3),np.zeros(2),np.ones(2)*.1,clip_bound=1)
    assert np.all((0<=v)&(v<=1))
