# Script to run all baseline experiments
import os
import random
import json
import numpy as np
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from utils import get_experiment_dir, get_enhanced_labels
from data.openmic_utils import get_openmic_loaders
from data.sonyc_utils import get_sonyc_loaders
from evaluate.eval_baseline import eval_baseline, forward
from trainer.trainer_baseline import trainer_baseline
from trainer.train_utils import create_model
import evaluate.metrics

def run(config):
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    exp_dir = get_experiment_dir(config)
    
    run_dir = os.path.join(exp_dir, 'seed_{}'.format(config['seed']))
    # tensorboard logger
    if os.path.exists(run_dir):
        print('Experiment already completed')
        return
    writer = SummaryWriter(run_dir)
    
    # get data loaders and metrics function
    if config['dataset'] == 'openmic':
        (train_loader, val_loader, test_loader), (full_dataset, train_inds) = get_openmic_loaders(config)
        n_classes = 20
        metric_fn = evaluate.metrics.metric_fn_openmic
    elif config['dataset'] == 'sonyc':
        (train_loader, val_loader, test_loader), train_dataset = get_sonyc_loaders(config)
        if config['coarse']:
            n_classes = 8
        else:
            n_classes = 23
        metric_fn = evaluate.metrics.metric_fn_sonycust

        # Randomly remove labels
        if 'label_drop_rate' in config:
            label_drop_rate = config['label_drop_rate']
            drop_mask = np.random.rand(*train_dataset.Y_mask.shape)
            drop_mask = train_dataset.Y_mask + drop_mask
            train_dataset.Y_mask = drop_mask > (1 + label_drop_rate)

    # hyper params
    hparams = config['hparams']
    lr = hparams['lr']
    wd = hparams['wd']
    model_params = {'drop_rate':hparams['dropout'], 'n_classes':n_classes, 'n_layers':hparams['n_layers']}
    num_epochs = hparams['num_epochs']
    prune_thres = hparams['prune_thres']
    batch_size = hparams['batch_size']

    # initialize models
    model = create_model(model_params)
   
    # initialize criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # initialize best metric variables
    best_models = [None, None]
    best_val_loss = 100000.0
    best_f1_macro = -1.0

    # teacher training loop
    for epoch in tqdm(range(num_epochs)):
        # drop learning rate every 30 epochs
        if (epoch > 0) and (epoch % 30 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.5
                lr = lr * 0.5

        # first train treating all missing labels as negatives
        train_loss = trainer_baseline(model, train_loader, optimizer, criterion, baseline_type=0)
        print('#### Training ####')
        print('Loss: {}'.format(train_loss))

        val_loss, metrics = eval_baseline(model, val_loader, criterion, n_classes, metric_fn, baseline_type=1)
        val_metric = 'F1_macro' if config['dataset'] == 'openmic' else 'auprc_macro'
        avg_val_metric = np.mean(metrics[val_metric])
        print('#### Validation ####')
        print('Loss: {}\t Macro F1 score: {}'.format(val_loss, avg_val_metric))

        # log to tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss_loss", val_loss, epoch)
        writer.add_scalar(f"val/{val_metric}", avg_val_metric, epoch)

        #Save best models
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_models[0] = deepcopy(model)

        if avg_val_metric > best_f1_macro:
            best_f1_macro = avg_val_metric
            best_models[1] = deepcopy(model)

    # Perform label pruning
    if config['dataset'] == 'openmic':
        X = full_dataset.X[train_inds]
        Y_mask = full_dataset.Y_mask[train_inds]
        X_dataset = TensorDataset(torch.tensor(X, requires_grad=False, dtype=torch.float32))
        loader = DataLoader(X_dataset, batch_size)
        all_predictions = forward(best_models[0], loader, n_classes)
        new_mask = get_enhanced_labels(Y_mask, all_predictions, prune_thres)
        full_dataset.Y_mask[train_inds] = new_mask

    if config['dataset'] == 'sonyc':
        X = train_dataset.X
        Y_mask = train_dataset.Y_mask
        X_dataset = TensorDataset(torch.tensor(X, requires_grad=False, dtype=torch.float32))
        loader = DataLoader(X_dataset, batch_size)
        all_predictions = forward(best_models[0], loader, n_classes)
        new_mask = get_enhanced_labels(Y_mask, all_predictions, prune_thres)
        train_dataset.Y_mask = new_mask
    # Retrain with pruned labels

    # initialize models
    model = create_model(model_params)
   
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # initialize best metric variables
    best_models = [None, None]
    best_val_loss = 100000.0
    best_f1_macro = -1.0

    for epoch in tqdm(range(num_epochs)):
        # drop learning rate every 30 epochs
        if (epoch > 0) and (epoch % 30 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.5
                lr = lr * 0.5

        # train with new mask
        train_loss = trainer_baseline(model, train_loader, optimizer, criterion, baseline_type=1)
        print('#### Training ####')
        print('Loss: {}'.format(train_loss))

        val_loss, metrics = eval_baseline(model, val_loader, criterion, n_classes, metric_fn, baseline_type=1)
        val_metric = 'F1_macro' if config['dataset'] == 'openmic' else 'auprc_macro'
        avg_val_metric = np.mean(metrics[val_metric])
        print('#### Validation ####')
        print('Loss: {}\t Macro F1 score: {}'.format(val_loss, avg_val_metric))

        # log to tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss_loss", val_loss, epoch)
        writer.add_scalar(f"val/{val_metric}", avg_val_metric, epoch)

        #Save best models
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_models[0] = deepcopy(model)

        if avg_val_metric > best_f1_macro:
            best_f1_macro = avg_val_metric
            best_models[1] = deepcopy(model)

    # Test best models
    for i, model in enumerate(best_models):
        test_loss, metrics = eval_baseline(model, test_loader, criterion, n_classes, metric_fn, baseline_type=1)

        print('#### Testing ####')
        print('Test Loss: ', test_loss)
        for key, val in metrics.items():
            print(f'Test {key}: {np.mean(val)}')
        
        # save metrics and model
        torch.save(model.state_dict(), os.path.join(run_dir, f'model_{i}.pth'))
        np.save(os.path.join(run_dir, f'metrics_{i}'), metrics)
        
        # jsonify metrics and write to json as well for manual inspection
        js = {}
        for key, val in metrics.items():
            if not np.ndim(val) == 0:
                js[key] = val.tolist()
            else:
                js[key] = val
        json.dump(js, open(os.path.join(run_dir, f'metrics_{i}.json'), 'w'))
    json.dump(config, open(os.path.join(run_dir, f'config.json'), 'w'))
    
if __name__ == "__main__":
    
    """
    For now just initialize config here
    TODO: Load config from json file
    """
    seeds = [0, 42, 345, 123, 45]
    prune_thres_list = [0.9]
    # seeds = [0]
    config = {
        'logdir': '../logs/OpenMIC_LE/',
        'exp_name': 'labelprune',
        'mode': 0,
        'coarse': 0,
        'data_path': '../data',
        'hparams': {
            'lr': 0.001,
            'wd': 1e-5,
            'n_layers': 3,
            'dropout': 0.3,
            'num_epochs': 100,
            'batch_size': 64,
            'prune_thres': 0.05
        }
    }

    """
    For OpenMIC
    """
    config['dataset'] = 'openmic'
    for prune_thres in prune_thres_list:
        for seed in seeds:
            config['seed'] = seed
            config['hparams']['prune_thres'] = prune_thres
            run(config)

    """
    For SONYC-UST:
    There are few missing labels in SONYC-UST.
    """
    # config['dataset'] = 'sonyc'
    # for prune_thres in prune_thres_list:
    #     for seed in seeds:
    #         config['seed'] = seed
    #         config['hparams']['prune_thres'] = prune_thres
    #         run(config)
