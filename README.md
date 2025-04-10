# Source Code for Representations of Graphs and Their Impact on Graph Neural Networks

Models can be found in the "models" folder in each dataset category folder. When running the models with the flag `--save_results True`, test scores will be saved under "runs" in the corresponding "models" folder, in a directory named after the timestamp of completion.

## Data
Most of the datasets are available through PyTorch Geometric and will be automatically downloaded when running the models. BlogCat (+ its splits) and DBLP (multi) can be downloaded from [the MLGNC repository](https://github.com/Tianqi-py/MLGNC/tree/main/data).

## Running the scripts
To run the experiments used in the paper, `cd` into the directory of the characteristic you are interested in, make the script executable by running `chmod +x INSERT_SCRIPT_NAME.sh`, and then run it using `./INSERT_SCRIPT_NAME.sh`. The scripts are divided based on model and whether the datasets require batching. For example, if I want to run the experiments on the **homophilic/heterophilic** datasets which **do not require batching**, using the GAT model, I run the following:
```
cd heterophilic_homophilic
chmod +x run_GAT.sh
./run_GAT.sh
```

This will produce a directory called `runs`, with folders containing the results for each dataset, named after the model, dataset and the timestamp of completion. In the folder, results are stored in two `.npy` files per dataset: one containing the NC test scores and one containing the LP test scores of the run.
