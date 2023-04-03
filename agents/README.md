# Procgen-PPO Experiments

This repository contains code to run experiments using Proximal Policy Optimization (PPO) on Procgen environments. It includes a script to run the experiments, `run_experiments.py`, an environment file for creating a conda environment (`environment.yml`), and a Jupyter notebook, `plots.ipynb`, for visualizing the results using TensorBoard.

## Setup

To set up the conda environment, follow these steps:

1. Install [conda](https://docs.conda.io/en/latest/) if you haven't already.
2. Create the conda environment using the provided `environment.yml` file by running the following command:

    ```bash
    conda env create -f environment.yml
    ```
3. Activate the created environment:

    ```bash
    conda activate procgen-ppo
    ```

## Running Experiments

The `run_experiments.py` script is the main entry point for running experiments. You can customize various parameters such as the list of GPU IDs to use, seeds, environment IDs, agents, and the name of the experiment.

Here is an example of how to run an experiment:

```bash
python run_experiments.py --gpu-ids 0 1 --seeds 0 1 2 --env-ids coinrun --agents impala --exp-name my_experiment
```

This command will run the experiment `my_experiment` using the `impala` agent on the `coinrun` environment with seeds `0`, `1`, and `2` on GPUs with IDs `0` and `1`. Tensorflow event files will be stored in the `runs/` directory.

For a full list of available options, run:
```bash
python run_experiments.py -h
```


## Analyzing Results

The `plots.ipynb` Jupyter notebook contains code to load TensorBoard results and output training and evaluation plots from data stored in the `results` directory. To view and interact with the notebook, run the following command:

```bash
jupyter notebook plots.ipynb
```

This will open the notebook in your default web browser. Follow the instructions within the notebook to load and visualize the experiment results.
