#!/bin/bash
cd /afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/
source myvenv/bin/activate
cd examples/phenoML/4D_customNorm/
python train_eventsAs4D_customNorm.py
