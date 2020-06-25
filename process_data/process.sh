#!/bin/bash
cd /afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/
source myvenv/bin/activate
cd process_data/
python process_aod_all.py
