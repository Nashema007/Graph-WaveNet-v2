import util
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import PurePath


def main(args, save_pred_path='preds.csv', save_metrics_path='last_test_metrics.csv', loader='test', **model_kwargs):
    device = torch.device(args.device)
    adjinit, supports = util.make_graph_inputs(args, device)
    model = GWNet.from_args(args, device, supports, adjinit, **model_kwargs)
    model.load_state_dict(torch.load('./logs/baseline_v2/worldly-forest-7/best_model.pth'))
    model.to(device)
    model.eval()
    print('model loaded successfully')
    data = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs, fill_zeroes=args.fill_zeroes)
    scaler = data['scaler']
    realy = torch.Tensor(data[f'y_{loader}']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    met_df, yhat, _, _, _ = util.calc_tstep_metrics_v(model, device, data[f'{loader}_loader'], scaler, realy, args.seq_length)
    df2 = util.make_pred_df(realy, yhat, scaler, args.seq_length)
    met_df.to_csv(save_metrics_path)
    df2.to_csv(save_pred_path, index=False)
    # if args.plotheatmap: 
    plot_learned_adj_matrix(model)
    return met_df, df2

def plot_learned_adj_matrix(model):
    adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
    adp = adp.cpu().detach().numpy()
    adp = adp / np.max(adp)
    df = pd.DataFrame(adp)
    adj_matrix = df.to_numpy()
    util.save_files('adjacency_matrix.npz', matrix=adj_matrix)
    sns.heatmap(df, cmap="RdYlBu")
    plt.savefig("heatmap.png")


if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    # parser.add_argument('--checkpoint', type=str, help='')
    parser.add_argument('--plotheatmap', action='store_true')
    args = parser.parse_args()
    main(args)
# python test.py --cat_feat_gc --fill_zeroes --do_graph_conv --adaptive_adjacency_matrix  --random_init_adjacency_matrix --data data/METR-LA/12 --dataset_name METR-LA --adjdata data/sensor_graph/adj_mx.pkl --seq_length 12 --num_nodes 207