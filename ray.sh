#!/bin/bash
# shellcheck disable=SC2206

#SBATCH --job-name=ray_cluster_test       # Job name
#SBATCH --output=outputs/slurm-%j.log    # Output log file
#SBATCH --account=<your-account-name>    # Replace with your SLURM account name
#SBATCH --partition=<your-partition>     # Replace with your SLURM partition (e.g., gpu, cpu)
#SBATCH --nodes=4                        # Number of nodes in the cluster
#SBATCH --ntasks-per-node=1              # One task per node
#SBATCH --gres=gpu:1                     # Number of GPUs per node (adjust as needed)
#SBATCH --cpus-per-task=4                # Number of CPUs per task
#SBATCH --mem=32GB                       # Memory per node (adjust as needed)
#SBATCH --time=00:20:00                  # Job runtime limit

# Load necessary modules or activate your environment
module load CUDA   # Load CUDA module (adjust if needed)
source activate <path-to-your-conda-env>  # Activate your Conda environment

# ===== DO NOT CHANGE THINGS BELOW UNLESS YOU KNOW WHAT YOU ARE DOING =====
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # Getting IP address of head node

# If the head node's IP contains a space, process it to extract IPv4 address (optional).
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPv6 address detected. Using IPv4 address: $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --temp-dir="/tmp/ray" --block &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) # Number of worker nodes
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --redis-password="$redis_password" --block &
  sleep 5
done

# ===== Call your code below =====
export WANDB_MODE=offline
python3 -u <your-script-name>.py \
  --num_samples=3 \
  --max_num_epochs=5 \
  --gpus_per_trial=1 \
  --cpus_per_trial=1 \
  --data_path="<your-data-path>" \
  --state_path="<your-state-path>"