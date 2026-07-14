#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pacbayes_tsk.experiments.development_v3 import aggregate_development

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='results/development/full_v3')
args=parser.parse_args()
print(json.dumps(aggregate_development(args.output),indent=2))
