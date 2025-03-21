# Source Code for Representations of Graphs and Their Impact on Graph Neural Networks

Models can be found in the "models" folder in each dataset category folder. When running the models with the flag `--save_results True`, test scores will be saved under "runs" in the corresponding "models" folder, in a directory named after the timestamp of completion.

## Data
Most of the datasets are available through PyTorch Geometric and will be automatically downloaded when running the models. BlogCat (+ its splits) and DBLP (multi) can be downloaded from [the MLGNC repository](https://github.com/Tianqi-py/MLGNC/tree/main/data).