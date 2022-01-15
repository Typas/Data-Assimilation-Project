#!/usr/bin/env bash

bash y_o_$1.sh
bash runda.sh
python python/plot.py
bash exp.sh $1
