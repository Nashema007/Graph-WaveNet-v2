#!/bin/bash

source venv/bin/activate

# METR-LA Runs

# Horizon 6/30mins

# python generate_training_data.py --output_dir=data/METR-LA/6 --seq_length_x 6 --seq_length_y 6 --traffic_df_filename=data/metr-la.h5

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207


python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/6 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 6 --num_nodes 207



# Horizon 12/1hr

python generate_training_data.py --output_dir=data/METR-LA/12 --seq_length_x 12 --seq_length_y 12 --traffic_df_filename=data/metr-la.h5

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207


# Horizon 24/2hrs

# python generate_training_data.py --output_dir=data/METR-LA/24 --seq_length_x 24 --seq_length_y 24 --traffic_df_filename=data/metr-la.h5

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/24 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 24 --num_nodes 207


# Horizon 36/3hrs

# python generate_training_data.py --output_dir=data/METR-LA/36 --seq_length_x 36 --seq_length_y 36 --traffic_df_filename=data/metr-la.h5

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/METR-LA/36 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 36 --num_nodes 207


# PEMS-BAY Runs

# Horizon 6/30mins

python generate_training_data.py --output_dir=data/PEMS-BAY/6 --seq_length_x 6 --seq_length_y 6 --traffic_df_filename=data/pems-bay.h5

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325


python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325


python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/6 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 6 --num_nodes 325


# Horizon 12/1hr

python generate_training_data.py --output_dir=data/PEMS-BAY/12 --seq_length_x 12 --seq_length_y 12 --traffic_df_filename=data/pems-bay.h5

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325

python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --es_patience 20 --save logs/baseline_v2 --data data/PEMS-BAY/12 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 12 --num_nodes 325



# Horizon 24/2hrs
# python generate_training_data.py --output_dir=data/PEMS-BAY/24 --seq_length_x 24 --seq_length_y 24  --traffic_df_filename=data/pems-bay.h5

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/24 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 24 --num_nodes 325

# Horizon 36/3hrs
# python generate_training_data.py --output_dir=data/PEMS-BAY/36 --seq_length_x 36 --seq_length_y 36 --traffic_df_filename=data/pems-bay.h5

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --data data/PEMS-BAY/36 --dataset_name PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --seq_length 36 --num_nodes 325


deactivate