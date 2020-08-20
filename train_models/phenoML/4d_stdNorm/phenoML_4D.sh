#!/bin/bash
cd /afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/
source myvenv/bin/activate
cd examples/phenoML/
python train_eventsAs4D_stdNorm.py
