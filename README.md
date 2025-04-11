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

The shell scripts already contain the flags used in the experiments. Below is a guide of all available flags.

| Flag                   | Description                                                                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `--hidden_channels`    | Number of hidden channels. (Default: 64)                                                                                    |
| `--embedding_size`     | Size of the embedding layer. (Default: 32)                                                                                  |
| `--num_epochs`         | Maximum number of training epochs. (Default: 250)                                                                                   |
| `--lr_NC`              | Learning rate for Node Classification. (Default: 0.01)                                                                       |
| `--lr_LP`              | Learning rate for Link Prediction. (Default: 0.01)                                                                           |
| `--nbr_seeds`          | Number of random seeds to run (starting from 1). (Default: 1)                                                                |
| `--save_results`       | Set to True if results and config should be saved in the `runs` directory. (Default: False)                                 |
| `--two_layers`         | Set to True if you want 2 layers instead of 3 in the models. (Default: False)                                               |
| `--msg_split_ratio`    | Fraction of message passing edges in training for LP (the rest become supervision edges). (Default: 0.3)                    |
| `--ds`                 | Which dataset to run on. (Default: `CiteSeer` for heterophilic/homophilic)                                                                              |
| `--neg_pos_ratio`      | How many times the amount of positive edges should the amount of negative edges be? -1 if all negative edges should be used. (Default: -1) |
