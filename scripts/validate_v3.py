#!/usr/bin/env python
from pathlib import Path
import yaml
from pacbayes_tsk.pac_bayes.priors_v3 import HierarchicalModelPrior,ModelIndex
from pacbayes_tsk.data.synthetic_v3 import generate
if __name__=='__main__':
 cfg=yaml.safe_load(Path('configs/v3/protocol_v3.yaml').read_text())
 grid=HierarchicalModelPrior(('ridge','dense_tsk','sparse_tsk'),tuple(cfg['model']['lags']),tuple(float(x) for x in cfg['model']['radii']),tuple(float(x) for x in cfg['model']['ridge_alphas']),cfg['model']['max_rules'],cfg['prior']['eta_rule']); grid.validate()
 _=grid.negative_log_mass(ModelIndex('sparse_tsk',grid.lags[0],grid.ridge_alphas[0],grid.radii[0],1))
 for name in yaml.safe_load(Path('configs/v3/synthetic_v3.yaml').read_text())['processes']:
  generate(name,length=64,burn_in=32,seed=1,parameters=yaml.safe_load(Path('configs/v3/synthetic_v3.yaml').read_text())['processes'][name])
 print('V3 protocol and generators: PASS')
