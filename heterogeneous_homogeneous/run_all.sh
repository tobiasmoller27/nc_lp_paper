#!/bin/bash

python models/ACM_HAN.py --num_epochs 10000 \
                         --num_seeds 10 \
                         --save_results True

python models/DBLP_HAN.py --num_epochs 10000 \
                         --num_seeds 10 \
                         --save_results True

python models/IMDB_HAN.py --num_epochs 10000 \
                         --num_seeds 10 \
                         --save_results True
