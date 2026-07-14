"""Predeclared V3 synthetic time-series generators.

These definitions are the source of truth for V3. Every generator returns the
post-burn-in observations, regime labels, and exact metadata used in manifests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class SyntheticSeries:
    values: np.ndarray
    regimes: np.ndarray
    metadata: dict[str, Any]
    def validate(self, expected_length: int) -> None:
        if self.values.shape != (expected_length,) or self.regimes.shape != (expected_length,):
            raise ValueError("Unexpected synthetic-series shape.")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Synthetic series contains non-finite values.")

def _finish(values, regimes, *, length, burn_in, metadata):
    values=np.asarray(values[burn_in:burn_in+length], dtype=np.float64)
    regimes=np.asarray(regimes[burn_in:burn_in+length], dtype=np.int64)
    result=SyntheticSeries(values, regimes, dict(metadata)); result.validate(length); return result

def generate_ar2(*, length:int, burn_in:int, seed:int, phi1:float=.6, phi2:float=-.2, noise_std:float=.2):
    rng=np.random.default_rng(seed); total=length+burn_in+2
    y=np.zeros(total); y[:2]=rng.normal(0,noise_std,2)
    for t in range(2,total): y[t]=phi1*y[t-1]+phi2*y[t-2]+rng.normal(0,noise_std)
    return _finish(y[2:],np.zeros(total-2),length=length,burn_in=burn_in,metadata={"generator":"ar2","equation_id":"V3-AR2","phi":[phi1,phi2],"noise_std":noise_std})

def generate_setar(*, length:int, burn_in:int, seed:int, threshold:float=0., noise_std:float=.2):
    rng=np.random.default_rng(seed); total=length+burn_in+2
    y=np.zeros(total); r=np.zeros(total,dtype=int); y[:2]=rng.normal(0,noise_std,2)
    for t in range(2,total):
        if y[t-1] <= threshold:
            y[t]=.65*y[t-1]+rng.normal(0,noise_std); r[t]=0
        else:
            y[t]=-.45*y[t-1]+.25*y[t-2]+rng.normal(0,noise_std); r[t]=1
    return _finish(y[2:],r[2:],length=length,burn_in=burn_in,metadata={"generator":"setar","equation_id":"V3-SETAR","threshold":threshold,"noise_std":noise_std,"low_phi":[.65,0.],"high_phi":[-.45,.25]})

def generate_narma10(*, length:int, burn_in:int, seed:int, input_low:float=0., input_high:float=.5):
    rng=np.random.default_rng(seed); order=10; total=length+burn_in+order+1
    u=rng.uniform(input_low,input_high,total); y=np.zeros(total)
    for t in range(order,total-1):
        y[t+1]=.3*y[t]+.05*y[t]*np.sum(y[t-9:t+1])+1.5*u[t-9]*u[t]+.1
    return _finish(y[order+1:],np.zeros(total-order-1),length=length,burn_in=burn_in,metadata={"generator":"narma10","equation_id":"V3-NARMA10","input_low":input_low,"input_high":input_high})

def generate_mackey_glass(*, length:int, burn_in:int, seed:int, beta:float=.2, gamma:float=.1, exponent:int=10, tau:int=17, noise_std:float=.01, initial_value:float=1.2):
    rng=np.random.default_rng(seed); total=length+burn_in+tau+1
    y=np.full(total,initial_value,dtype=float); y[:tau+1]+=rng.normal(0,.02,tau+1)
    for t in range(tau,total-1):
        delayed=y[t-tau]
        y[t+1]=y[t]+beta*delayed/(1+delayed**exponent)-gamma*y[t]+rng.normal(0,noise_std)
    return _finish(y[tau+1:],np.zeros(total-tau-1),length=length,burn_in=burn_in,metadata={"generator":"mackey_glass","equation_id":"V3-MG","beta":beta,"gamma":gamma,"exponent":exponent,"tau":tau,"noise_std":noise_std})

def generate_garch(*, length:int, burn_in:int, seed:int, omega:float=.05, alpha:float=.10, beta:float=.85, degrees_of_freedom:float=5.):
    if alpha+beta>=1 or degrees_of_freedom<=2: raise ValueError("Invalid GARCH parameters.")
    rng=np.random.default_rng(seed); total=length+burn_in
    h=np.empty(total); y=np.empty(total); h[0]=omega/(1-alpha-beta)
    z=rng.standard_t(degrees_of_freedom,total)*np.sqrt((degrees_of_freedom-2)/degrees_of_freedom)
    y[0]=np.sqrt(h[0])*z[0]
    for t in range(1,total):
        h[t]=omega+alpha*y[t-1]**2+beta*h[t-1]; y[t]=np.sqrt(h[t])*z[t]
    return _finish(y,np.zeros(total),length=length,burn_in=burn_in,metadata={"generator":"garch","equation_id":"V3-GARCH","omega":omega,"alpha":alpha,"beta":beta,"degrees_of_freedom":degrees_of_freedom,"conditional_mean":"zero"})

def generate_structural_break(*, length:int, burn_in:int, seed:int, break_fraction:float=.6, pre_mean:float=0., pre_phi:float=.75, pre_noise_std:float=.30, post_mean:float=1.5, post_phi:float=.30, post_noise_std:float=.55):
    rng=np.random.default_rng(seed); total=length+burn_in; break_at=burn_in+int(round(length*break_fraction))
    y=np.zeros(total); r=np.zeros(total,dtype=int); y[0]=rng.normal(pre_mean,pre_noise_std)
    for t in range(1,total):
        if t<break_at: mu,phi,sd,reg=pre_mean,pre_phi,pre_noise_std,0
        else: mu,phi,sd,reg=post_mean,post_phi,post_noise_std,1
        y[t]=mu+phi*(y[t-1]-mu)+rng.normal(0,sd); r[t]=reg
    return _finish(y,r,length=length,burn_in=burn_in,metadata={"generator":"structural_break","equation_id":"V3-BREAK","break_fraction":break_fraction,"pre_mean":pre_mean,"pre_phi":pre_phi,"pre_noise_std":pre_noise_std,"post_mean":post_mean,"post_phi":post_phi,"post_noise_std":post_noise_std})

_GENERATORS={"ar2":generate_ar2,"setar":generate_setar,"narma10":generate_narma10,"mackey_glass":generate_mackey_glass,"garch":generate_garch,"structural_break":generate_structural_break}
def generate(name:str, *, length:int, burn_in:int, seed:int, parameters:Mapping[str,Any]|None=None):
    key=name.lower().replace('-','_')
    if key not in _GENERATORS: raise ValueError(f"Unknown V3 generator: {name}")
    return _GENERATORS[key](length=length,burn_in=burn_in,seed=seed,**dict(parameters or {}))
