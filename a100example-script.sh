#!/bin/sh

#SBATCH --account=acsl
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-4g-20gb:1
#SBATCH --time=100:00:00
#SBATCH --job-name="Graph-WaveNet-v2"

module purge

CUDA_VISIBLE_DEVICES=$(ncvd)

source /home/chkash007/Graph-WaveNet-v2/venv/bin/activate 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chkash007/.conda/envs/venv/lib

# METR-LA Runs

# Horizon 6/30mins

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207

# Horizon 12/1hr

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207

# Horizon 24/2hrs

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# Horizon 36/3hrs

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12_36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207


# PEMS-BAY Runs

#  Horizon 6/30mins

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325

#  Horizon 12/1hr

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325

# python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325

#  Horizon 24/2hrs


python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325


# Horizon 36/3hrs

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12_36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325


deactivate
