from __future__ import annotations
from dataclasses import dataclass
import numpy as np
ROLES=("prior","bound","validation","test")
@dataclass(frozen=True)
class TemporalSplit:
    labels: np.ndarray
    counts: dict[str,int]
    def indices(self, role:str)->np.ndarray:
        if role not in ROLES: raise ValueError(f"Unknown role: {role}")
        return np.flatnonzero(self.labels==role)
    def validate(self)->None:
        if set(np.unique(self.labels)) != set(ROLES): raise ValueError("All four temporal roles are required.")
        order={r:i for i,r in enumerate(ROLES)}; coded=np.array([order[x] for x in self.labels])
        if np.any(np.diff(coded)<0): raise ValueError("Temporal roles are not chronological.")
        if sum(self.counts.values()) != len(self.labels): raise ValueError("Split counts do not sum to length.")
def ratio_split(n:int, fractions:dict[str,float])->TemporalSplit:
    if tuple(fractions)!=ROLES: raise ValueError(f"Fractions must be ordered as {ROLES}.")
    vals=np.array([fractions[r] for r in ROLES],float)
    if n<4 or np.any(vals<=0) or not np.isclose(vals.sum(),1): raise ValueError("Invalid ratio split.")
    raw=n*vals; sizes=np.floor(raw).astype(int); rem=n-sizes.sum(); order=np.argsort(-(raw-sizes),kind='stable')
    for i in order[:rem]: sizes[i]+=1
    if np.any(sizes<1): raise ValueError("A split is empty.")
    labels=np.concatenate([np.repeat(r,s) for r,s in zip(ROLES,sizes)]).astype(object)
    result=TemporalSplit(labels,{r:int(s) for r,s in zip(ROLES,sizes)}); result.validate(); return result
def official_horizon_split(n:int, *, horizon:int, prior_fraction_of_prefix:float=.30, min_prior:int=4, min_bound:int=4)->TemporalSplit:
    prefix=n-2*horizon
    if horizon<=0 or prefix<min_prior+min_bound: raise ValueError("Series is too short for official-horizon split.")
    prior=max(min_prior,int(np.floor(prefix*prior_fraction_of_prefix))); prior=min(prior,prefix-min_bound); bound=prefix-prior
    labels=np.concatenate([np.repeat('prior',prior),np.repeat('bound',bound),np.repeat('validation',horizon),np.repeat('test',horizon)]).astype(object)
    result=TemporalSplit(labels,{"prior":prior,"bound":bound,"validation":horizon,"test":horizon}); result.validate(); return result
def assert_role_subset(indices:np.ndarray, split:TemporalSplit, allowed:set[str], *, name:str)->None:
    observed=set(split.labels[np.asarray(indices,dtype=int)].tolist())
    if not observed.issubset(allowed): raise ValueError(f"{name} used forbidden roles: {sorted(observed-allowed)}")
