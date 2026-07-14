#!/usr/bin/env python
from pathlib import Path
from pacbayes_tsk.experiments.freeze_v3 import freeze
ROOT=Path('.')
TRACKED=[]
for base in ['protocol','configs/v3','src/pacbayes_tsk','scripts','tests/v3','docs','.github']:
    for p in (ROOT/base).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
            TRACKED.append(p.as_posix())
TRACKED += ['pyproject.toml','requirements-lock.txt','README.md','CITATION.cff','.gitignore','EXECUTION_STATUS_AR.md']
if __name__=='__main__':
    freeze('protocol/EXPERIMENT_PROTOCOL_V3.md',TRACKED,'artifacts/development_protocol_snapshot_v3.json')
