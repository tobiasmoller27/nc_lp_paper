#!/bin/bash

python models/GIN_base.py --num_epochs 10000 \
                         --num_seeds 10 \
                         --save_results True \
                         --ds DBLP