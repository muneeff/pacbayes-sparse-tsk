#!/usr/bin/env python
"""End-to-end V3 smoke experiment.

This is a computational integration check, not confirmatory evidence. It fits
Ridge, dense TSK, and sparse TSK on one predeclared synthetic trajectory,
selects each family without test access, and then computes a PAC-Bayes
certificate on bound+validation only.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pacbayes_tsk.data.synthetic_v3 import generate
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.models.sparse_tsk import fit_radius_antecedent, fit_sparse_tsk_consequents
from pacbayes_tsk.pac_bayes.priors_v3 import HierarchicalModelPrior, ModelIndex
from pacbayes_tsk.pac_bayes.certificates_v3 import (
    clip_bound_from_prior, gaussian_kl_diag, martingale_certificate,
    pointwise_gibbs_upper,
)

LAGS=(3,5)
RADII=(0.75,1.25)
ALPHAS=(0.01,0.1)
FAMILIES=("ridge","dense_tsk","sparse_tsk")
PRIOR_SCALES=(0.5,1.0,2.0)
POSTERIOR_RATIOS=(0.1,0.3)
TEMPERATURES=(0.25,0.5,1.0,2.0)

@dataclass
class Candidate:
    family:str; lag:int; radius:float|None; alpha:float; rule_count:int
    validation_rmse:float; validation_mae:float
    design_prior:np.ndarray; design_bound:np.ndarray; design_validation:np.ndarray; design_test:np.ndarray
    y_prior:np.ndarray; y_bound:np.ndarray; y_validation:np.ndarray; y_test:np.ndarray
    prior_mean:np.ndarray; posterior_mean:np.ndarray


def lagged(values:np.ndarray, labels:np.ndarray, p:int):
    idx=np.arange(p,len(values)); X=np.column_stack([values[idx-j] for j in range(1,p+1)])
    return X,values[idx],labels[idx]

def masks(target_labels):
    return {r:target_labels==r for r in ("prior","bound","validation","test")}

def ridge_design(X):
    return np.column_stack([np.ones(len(X)),X])

def fit_linear(design,y,alpha):
    est=Ridge(alpha=alpha,fit_intercept=False).fit(design,y)
    return np.asarray(est.coef_,float).reshape(-1)

def prior_std(design,y,mean,alpha):
    residual=y-design@mean
    variance=max(1e-4,float(np.mean(residual**2)))
    gram=design.T@design+alpha*np.eye(design.shape[1])
    diag=np.diag(np.linalg.pinv(gram))
    return np.sqrt(np.clip(variance*diag,1e-6,100.0))

def metrics(y,pred):
    err=y-pred
    return float(np.sqrt(np.mean(err**2))),float(np.mean(np.abs(err)))

def main():
    out=Path('results/development'); out.mkdir(parents=True,exist_ok=True)
    series=generate('ar2',length=800,burn_in=300,seed=3000)
    split=ratio_split(len(series.values),{'prior':.20,'bound':.45,'validation':.15,'test':.20})
    # Fit target/feature scale only on the prior portion.
    prior_raw=series.values[split.labels=='prior']
    mean=float(prior_raw.mean()); scale=float(prior_raw.std()) or 1.0
    values=(series.values-mean)/scale
    candidates=[]; candidate_rows=[]
    for p in LAGS:
        X,y,target_labels=lagged(values,split.labels,p); m=masks(target_labels)
        for family in FAMILIES:
            radius_values=(None,) if family=='ridge' else RADII
            for radius in radius_values:
                for alpha in ALPHAS:
                    if family=='ridge':
                        designs={r:ridge_design(X[m[r]]) for r in m}; k=1
                    else:
                        max_active=3 if family=='sparse_tsk' else 8
                        antecedent=fit_radius_antecedent(X[m['prior']],radius=radius,max_rules=8,max_active_rules=max_active)
                        if antecedent.radius_cap_reached:
                            continue
                        designs={r:antecedent.design_matrix(X[m[r]]) for r in m}; k=antecedent.rule_count
                    pmean=fit_linear(designs['prior'],y[m['prior']],alpha)
                    qmean=fit_linear(designs['bound'],y[m['bound']],alpha)
                    vrmse,vmae=metrics(y[m['validation']],designs['validation']@qmean)
                    c=Candidate(family,p,radius,alpha,k,vrmse,vmae,designs['prior'],designs['bound'],designs['validation'],designs['test'],y[m['prior']],y[m['bound']],y[m['validation']],y[m['test']],pmean,qmean)
                    candidates.append(c)
                    candidate_rows.append({'family':family,'lag':p,'radius':radius,'ridge_alpha':alpha,'rule_count':k,'consequent_dimension':len(qmean),'validation_rmse':vrmse,'validation_mae':vmae})
    pd.DataFrame(candidate_rows).to_csv(out/'smoke_v3_candidates.csv',index=False)
    prior=HierarchicalModelPrior(FAMILIES,LAGS,RADII,ALPHAS,8)
    selected=[]
    for family in FAMILIES:
        pool=[c for c in candidates if c.family==family]
        pool.sort(key=lambda c:(c.validation_rmse,c.validation_mae,len(c.posterior_mean),c.rule_count,float('inf') if c.radius is None else c.radius,c.alpha,c.lag))
        c=pool[0]
        idx=ModelIndex(c.family,c.lag,c.alpha,c.radius,c.rule_count)
        structure_kl=prior.negative_log_mass(idx)
        base_std=prior_std(c.design_prior,c.y_prior,c.prior_mean,c.alpha)
        clip=clip_bound_from_prior(c.y_prior)
        design_cert=np.vstack([c.design_bound,c.design_validation]); y_cert=np.concatenate([c.y_bound,c.y_validation])
        best=None
        for mult in PRIOR_SCALES:
            pstd=np.clip(base_std*mult,1e-3,10.)
            for ratio in POSTERIOR_RATIOS:
                qstd=np.clip(pstd*ratio,1e-5,10.)
                empirical=float(pointwise_gibbs_upper(design_cert,y_cert,c.posterior_mean,qstd,clip_bound=clip).mean())
                gkl=gaussian_kl_diag(c.posterior_mean,qstd,c.prior_mean,pstd)
                total=gkl+structure_kl+math.log(len(PRIOR_SCALES))
                cert=martingale_certificate(empirical_risk=empirical,total_kl=total,n=len(y_cert),delta=.05,temperatures=TEMPERATURES)
                row=(cert.untruncated,mult,ratio,empirical,gkl,total,cert)
                if best is None or row[0]<best[0]: best=row
        _,mult,ratio,empirical,gkl,total,cert=best
        test_rmse,test_mae=metrics(c.y_test,c.design_test@c.posterior_mean)
        selected.append({'phase':'development_smoke','dataset':'ar2','seed':3000,'family':family,'lag':c.lag,'radius':c.radius,'ridge_alpha':c.alpha,'rule_count':c.rule_count,'consequent_dimension':len(c.posterior_mean),'validation_rmse':c.validation_rmse,'validation_mae':c.validation_mae,'test_rmse':test_rmse,'test_mae':test_mae,'clip_bound':clip,'prior_scale':mult,'posterior_ratio':ratio,'empirical_gibbs_risk':empirical,'gaussian_kl':gkl,'structure_kl':structure_kl,'total_kl':total,'temperature':cert.temperature,'certificate':cert.certificate,'certificate_untruncated':cert.untruncated,'certificate_uses_test':False})
    frame=pd.DataFrame(selected); frame.to_csv(out/'smoke_v3_summary.csv',index=False)
    payload={'status':'DEVELOPMENT_SMOKE_ONLY','confirmatory':False,'selection_used_test':False,'certificate_used_test':False,'rows':selected}
    (out/'smoke_v3_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(frame[['family','validation_rmse','test_rmse','total_kl','certificate']].to_string(index=False))
if __name__=='__main__': main()
