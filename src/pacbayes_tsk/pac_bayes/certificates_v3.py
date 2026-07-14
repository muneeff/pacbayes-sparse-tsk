from __future__ import annotations
from dataclasses import dataclass
import math, numpy as np
@dataclass(frozen=True)
class CertificateResult:
    empirical_risk:float; total_kl:float; temperature:float; untruncated:float; certificate:float

def gaussian_kl_diag(q_mean,q_std,p_mean,p_std)->float:
    qm=np.asarray(q_mean,float).reshape(-1); qs=np.asarray(q_std,float).reshape(-1); pm=np.asarray(p_mean,float).reshape(-1); ps=np.asarray(p_std,float).reshape(-1)
    if not(qm.shape==qs.shape==pm.shape==ps.shape) or np.any(qs<=0) or np.any(ps<=0): raise ValueError("Invalid diagonal Gaussians.")
    val=.5*np.sum(np.log((ps*ps)/(qs*qs))-1+(qs*qs+(qm-pm)**2)/(ps*ps))
    return float(max(0.,val))
def clip_bound_from_prior(y_prior, minimum:float=1.)->float:
    y=np.asarray(y_prior,float).reshape(-1)
    if y.size==0 or not np.all(np.isfinite(y)): raise ValueError("Invalid prior targets.")
    return float(max(minimum,np.max(np.abs(y))))
def pointwise_gibbs_upper(design,target,q_mean,q_std,*,clip_bound:float):
    X=np.asarray(design,float); y=np.asarray(target,float).reshape(-1); m=np.asarray(q_mean,float).reshape(-1); s=np.asarray(q_std,float).reshape(-1)
    if X.shape!=(len(y),len(m)) or len(s)!=len(m) or np.any(s<=0) or clip_bound<=0: raise ValueError("Invalid Gibbs-risk inputs.")
    yc=np.clip(y,-clip_bound,clip_bound); second=(yc-X@m)**2+(X*X)@(s*s)
    return np.minimum(1.,second/(4*clip_bound*clip_bound))
def martingale_certificate(*, empirical_risk:float,total_kl:float,n:int,delta:float,temperatures,temperature_mass=None)->CertificateResult:
    if not 0<=empirical_risk<=1 or total_kl<0 or n<1 or not 0<delta<1: raise ValueError("Invalid certificate inputs.")
    temps=tuple(float(x) for x in temperatures)
    if not temps or any(x<=0 for x in temps): raise ValueError("Invalid temperature grid.")
    mass=1/len(temps) if temperature_mass is None else float(temperature_mass)
    best=None
    for lam in temps:
        value=empirical_risk+lam/8+(total_kl+math.log(1/(delta*mass)))/(lam*n)
        row=CertificateResult(empirical_risk,total_kl,lam,value,min(1.,value))
        if best is None or row.untruncated<best.untruncated: best=row
    return best
