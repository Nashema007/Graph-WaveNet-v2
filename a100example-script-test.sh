#!/bin/sh

#SBATCH --account=acsl
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-4g-20gb:1
#SBATCH --time=100:00:00
#SBATCH --job-name="GRAPH-WAVENET"

module purge

CUDA_VISIBLE_DEVICES=$(ncvd)

source /home/chkash007/Graph-Wavenet/venv/bin/activate 

# METR-LA Runs

# Horizon 24/2hrs

python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# Horizon 36/3hrs

python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207


# PEMS-BAY Runs

# Horizon 24/2hrs

python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

# Horizon 36/3hrs

python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325


deactivate
