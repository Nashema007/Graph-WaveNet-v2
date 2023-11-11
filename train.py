import torch
import numpy as np
import pandas as pd
import time
import util
from engine import Trainer
import os
from durbango import pickle_save
from fastprogress import progress_bar

from model import GWNet
from util import calc_tstep_metrics, masked_mae_station, log_metrics_predictions, save_files
from exp_results import summary
import wandb
from pathlib import PurePath, Path

def main(args, **model_kwargs):
    wandb.init(project="graph-wavenet-v2",
               entity="nashema007",
            #    mode='disabled',
               config={
                   "learning_rate": args.learning_rate, 
                   "lr_decay_rate": args.lr_decay_rate, 
                   "epochs": args.epochs, 
                   "batch_size": args.batch_size, 
                   "in_dim": args.in_dim, 
                   "seq_length": args.seq_length, 
                   "num_nodes": args.num_nodes, 
                   "num_hid": args.num_hid, 
                   "dropout": args.dropout, 
                   "weight_decay": args.weight_decay, 
                   "clip": args.clip, 
                   "optimiser": "Adam",
                   "random_init_adjacency_matrix": args.random_init_adjacency_matrix, 
                   "adaptive_adjacency_matrix": args.adaptive_adjacency_matrix, 
                   "adaptive_adjacency_matrix_only": args.adaptive_adjacency_matrix_only, 
                #    "seed": args.seed, 
                   "architecture": "STGNN",
                   "dataset": args.dataset_name, 
                   "adjacency_type": args.adjacency_type, 
                   "n_iters": args.n_iters, 
                   "es_patience": args.es_patience, 
                   "do_graph_conv": args.do_graph_conv, 
                   "cat_feat_gc": args.cat_feat_gc, 
                   "fill_zeroes": args.fill_zeroes, 
                   "adjdata": args.adjdata,
                   "apt_size": args.apt_size,
               })
    config = wandb.config
    Path('./' + args.save + '/' + str(wandb.run.name).lower()).mkdir(parents=True, exist_ok=True)
    pickle_save(args, f'{args.save}/{str(wandb.run.name)}/args.pkl')
    
     # set seed
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    wandb.config.update({"seed": torch.seed()})
    data = util.load_dataset(args.data, config.batch_size, config.batch_size, config.batch_size, n_obs=args.n_obs, fill_zeroes=config.fill_zeroes)
    scaler = data['scaler']
    aptinit, supports = util.make_graph_inputs(config, device)

    model = GWNet.from_args(config, device, supports, aptinit, **model_kwargs)
    if args.checkpoint:
        model.load_checkpoint(torch.load(args.checkpoint))
    model.to(device)
    engine = Trainer.from_args(model, scaler, config)
    wandb.watch(models=model, criterion=engine.loss, log='all', log_freq=10)
    metrics = []
    best_model_save_path = os.path.join(args.save, str(wandb.run.name).lower(), 'best_model.pth')
    
    lowest_mae_yet = 100  # high value, will get overwritten
    mb = progress_bar(list(range(1, config.epochs + 1)))
    epochs_since_best_mae = 0
    train_station_loss = []
    valid_station_loss = []
    test_station_loss = []
    for _ in mb:
        train_loss, train_mape, train_rmse, t_station_loss = [], [], [], []
        data['train_loader'].shuffle()
        for iter, (x, y) in enumerate(data['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            yspeed = trainy[:, 0, :, :]
            if yspeed.max() == 0: continue
            mae, mape, rmse = engine.train(trainx, yspeed)
            train_loss.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            t_station_loss.append(masked_mae_station(trainy, trainx, 0.0).cpu().detach().numpy())
            if config.n_iters is not None and iter >= config.n_iters:
                break
        engine.scheduler.step()
        _, valid_loss, valid_mape, valid_rmse, v_station_loss = eval_(data['val_loader'], device, engine)
        m = dict(train_loss=np.mean(train_loss), train_mape=np.mean(train_mape),
                 train_rmse=np.mean(train_rmse), valid_loss=np.mean(valid_loss),
                 valid_mape=np.mean(valid_mape), valid_rmse=np.mean(valid_rmse))
        
        wandb_metrics = {"Loss/train": m["train_loss"],
                   "MAPE/train": m["train_mape"],
                   "RMSE/train": m["train_rmse"],
                   "Loss/validation": m["valid_loss"],
                   "MAPE/validation": m["valid_mape"],
                   "RMSE/validation": m["valid_rmse"],
                   }
        wandb.log(wandb_metrics)
        t_station_loss = np.array(t_station_loss)
        train_station_loss.append(util.process_train_valid_station_loss(t_station_loss)) 
        v_station_loss = np.array(v_station_loss)
        valid_station_loss.append(util.process_train_valid_station_loss(v_station_loss))
        
        m = pd.Series(m)
        metrics.append(m)
        
        if m.valid_loss < lowest_mae_yet:
            torch.save(engine.model.state_dict(), best_model_save_path)
            lowest_mae_yet = m.valid_loss
            epochs_since_best_mae = 0
        else:
            epochs_since_best_mae += 1
        met_df = pd.DataFrame(metrics)
        mb.comment = f'best val_loss: {met_df.valid_loss.min(): .3f}, current val_loss: {m.valid_loss:.3f}, current train loss: {m.train_loss: .3f}'
        met_df.round(6).to_csv(f'{args.save}/{str(wandb.run.name)}/metrics.csv')
        if epochs_since_best_mae >= config.es_patience: break
    # Metrics on test data
    engine.model.load_state_dict(torch.load(best_model_save_path))
    realy = torch.Tensor(data['y_test']).transpose(1, 3)[:, 0, :, :].to(device)
    test_met_df, yhat, predList, realList, test_station_loss = calc_tstep_metrics(engine.model, device, data['test_loader'], scaler, realy, config.seq_length)
    log_metrics_predictions(test_station_loss, train_station_loss, valid_station_loss)
    save_files(
        PurePath('./results/predictions/predictions-{}-{}.npz'.format(
            str(config.dataset).lower(), 
            str(wandb.run.name).lower()
        )),
        labels=realList,
        predictions=predList,
    )
    save_files(
        PurePath('./results/station_losses/stations_losses-{}-{}.npz'.format(
            str(config.dataset).lower(), 
            str(wandb.run.name).lower()
        )),
        train_station_loss=train_station_loss,
        valid_station_loss=valid_station_loss,
        test_station_loss=test_station_loss,
    )
    test_met_df.round(6).to_csv(os.path.join(args.save, str(wandb.run.name),'test_metrics.csv'))
    print(summary(args.save))

def eval_(ds, device, engine):
    """Run validation."""
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    v_station_loss = []
    s1 = time.time()
    for (x, y) in ds.get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        testy = torch.Tensor(y).to(device).transpose(1, 3)
        metrics = engine.eval(testx, testy[:, 0, :, :])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
        v_station_loss.append(masked_mae_station(testy, testx, 0.0).cpu().detach().numpy())
    total_time = time.time() - s1
    return total_time, valid_loss, valid_mape, valid_rmse, v_station_loss


if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    parser.add_argument('--save', type=str, default='experiment', help='save path')
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--es_patience', type=int, default=20, help='quit if no improvement after this many iterations')

    args = parser.parse_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print(f"Total time spent: {mins:.2f} seconds")
