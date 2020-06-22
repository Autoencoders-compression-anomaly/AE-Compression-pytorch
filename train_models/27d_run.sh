#!/bin/bash
python -m virtualenv -p python3 myvenv
source myvenv/bin/activate
pip install future
pip install pandas
pip install fastai
pip install corner
python examples/27D/27D_train.py
