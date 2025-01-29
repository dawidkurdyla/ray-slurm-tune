# Running Ray Tune on SLURM with Optuna and WANDB Integration

This guide provides a step-by-step explanation of how to set up and execute Ray Tune on an SLURM cluster with Optuna and WANDB (Weights & Biases) integration. The approach uses a head node for coordination and all other nodes as workers.

This workflow is based on inspiration from the following repositories:

[ray-tune-slurm-demo](https://github.com/klieret/ray-tune-slurm-demo)

[wandb-offline-sync-hook](https://github.com/klieret/wandb-offline-sync-hook/)

## Repository Structure

- `ray_tune_train_res_18_wandb_os_set.py`: The main Python script that trains a model using Ray Tune and integrates with WANDB.
- `slurm_ray_cluster.sh`: SLURM batch script to initialize the Ray cluster and run the training script.
- `ray-tune-notebook-no-slurm.ipynb`: Notebook file designed to test training code without slurm.
- `outputs/`: Directory for SLURM log files.

## Prerequisites

Before proceeding, ensure the following:

1. **Access to an SLURM cluster**: A supercomputer environment with GPUs.
2. **Installed software**:
   - Conda (for environment management).
   - Ray (`pip install ray`).
   - Torch and TorchVision (`pip install torch torchvision` for GPU).
   - WANDB (`pip install wandb`).
   - Optuna (`pip install optuna`)
3. **Offline WANDB Setup**:
   - Follow the instructions in [wandb-offline-sync-hook](https://github.com/klieret/wandb-offline-sync-hook/) to enable offline logging.

## Steps to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Configure Your Environment

Replace `<env_path>` with the desired path to your Conda environment in the commands below.

#### Create a Conda Environment

```bash
conda create -p <env_path> python=3.9 -y
conda activate <env_path>
```

#### Install Required Packages

```bash
pip install torch torchvision optuna ray[tune] ray[default] joblib wandb wandb-osh colorlogs 
```

### 3. Use splitter to split dataset.

Models were trained on !(asl-sign-language)[https://www.kaggle.com/datasets/grassknoted/asl-alphabet]. You will have to download the dataset on machine drive. Later use `splitter.py` pointing at training data folder (containing subfolders with categories). You can choose any destination path you want.

```
python splitter.py --source=<SRC_PATH> --dest=<DEST_PATH>
```

### 4. Adjust SLURM Script

Edit `slurm_ray_cluster.sh` as needed:

- Replace paths in the script to point to your data directory, state directory, and Conda environment. You can also adjust number of trials (attempts for finding best parameters) using `--num_samples` flag. You can also change maximum number of epochs model is trained `--max_num_epochs` and GPU/CPU per trial ` --gpus_per_trial --cpus_per_trial`.
  
  Example:
  ```bash
  # Data and state directories
  --data_path="/path/to/your/data" \
  --state_path="/path/to/your/state" \
  --num_samples=<int> \
  --max_num_epochs=<int> \
  --gpus_per_trial=<int> \
  --cpus_per_trial=<int> \

  # Conda environment
  source activate <env_path>
  ```

### 4. Synchronize WANDB Logs (Optional)

If using WANDB offline mode, in order to sync wandb live run:

```bash
wandb-osh
```

### 5. Submit the Job

Submit the job to SLURM using:

```bash
sbatch slurm_ray_cluster.sh
```

### 6. Monitor Logs

Log into your weights and biases account where logs will be streamed.

## Script Explanation

### SLURM Batch Script (`slurm_ray_cluster.sh`)

This script:

1. Initializes the Ray head node on the first SLURM node.
2. Adds all remaining nodes as Ray workers.
3. Launches the training script (`ray_tune_train_res_18_wandb_os_set.py`).

Adjustable parameters include:

- `--num_samples`: Number of hyperparameter samples.
- `--max_num_epochs`: Maximum number of epochs per trial.
- `--gpus_per_trial`: GPUs allocated per trial.
- `--cpus_per_trial`: CPUs allocated per trial.

### Training Script (`ray_tune_train_res_18_wandb_os_set.py`)

This Python script:

1. Defines a PyTorch model and dataset loader.
2. Uses Ray Tune for hyperparameter optimization.
3. Logs training progress to WANDB (offline mode).

#### Key Functions

- **`train_model`**: Defines the model training loop.
- **`test_best_mode`**: Tests and saves weights state of the best model.
- **`main`**: Sets up Ray Tune configuration and starts trials.
- **`load_data`**: Loads training and test data.

## Notes and Recommendations

- **Disk Space**: Ensure sufficient scratch space for training logs and model checkpoints.
- **Environment Path**: Use paths under `$SCRATCH` or `$HOME` for portability and compliance with cluster policies.

## References

- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [wandb-offline-sync-hook](https://github.com/klieret/wandb-offline-sync-hook/)
- [Ray on SLURM Demo](https://github.com/klieret/ray-tune-slurm-demo)

