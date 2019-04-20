#!/usr/bin/env bash

source venv/bin/activate
jupyter nbconvert train.ipynb --to python

scp train.py articuno:~/cancer/
rm train.py
