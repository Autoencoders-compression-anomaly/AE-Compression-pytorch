#!/bin/bash
python -m virtualenv myvenv
source myvenv/bin/activate
pip install pandas
pip install FunctionScaler
python scale_data.py
