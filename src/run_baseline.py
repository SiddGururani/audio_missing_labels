# Script to run all baseline experiments

import os
import random
import json
import numpy as mp
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter

from utils import get_experiment_dir
from data.openmic_utils import get_openmic_loaders
from data.sonyc_utils import get_sonyc_loaders
from evaluate.eval_baseline import eval_baseline
from trainer.trainer_baseline import trainer_baseline
import evaluate.metrics

def run(config):
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    exp_dir = get_experiment_dir(config)
    base_type = config['type']
    
    run_dir = os.path.join(exp_dir, config['seed'])
    # tensorboard logger
    writer = SummaryWriter(run_dir)
    
    # get data loaders and metrics function
    if config['dataset'] == 'openmic':
        (train_loader, val_loader, test_loader), _ = get_openmic_loaders(config)
        n_classes = 20
        metric_fn = evaluate.metrics.metric_fn_openmic
    elif config['dataset'] == 'sonyc':
        (train_loader, val_loader, test_loader), _ = get_sonyc_loaders(config)
        if config['coarse']:
            n_classes = 8
        else:
            n_classes = 23
        metric_fn = evaluate.metrics.metric_fn_sonycust

    # hyper params
    hparams = config['hparams']
    lr = hparams['lr']
    wd = hparams['wd']
    model_params = {'drop_rate':hparams['dropout'], 'n_classes':n_classes, 'n_layers':hparams['n_layers']}
    num_epochs = hparams['num_epochs']

    # initialize models
    model = create_model(model_params)
   
    # initialize criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # initialize best metric variables
    best_models = [None, None]
    best_val_loss = 100000.0
    best_f1_macro = -1.0

    # training loop
    for epoch in tqdm(range(num_epochs)):
        # drop learning rate every 30 epochs
        if (epoch > 0) and (epoch % 30 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.5
                lr = lr * 0.5

        train_loss = trainer_baseline(model, train_loader, optimizer, criterion, base_type)
        print('#### Training ####')
        print('Loss: {}'.format(train_loss))

        val_loss, metrics = eval_baseline(model, val_loader, criterion, n_classes, metric_fn, baseline_type=base_type)
        val_metric = 'F1_macro' if config['dataset'] == 'openmic' else 'auprc_macro'
        avg_val_metric = np.mean(metrics[val_metric])
        print('#### Validation ####')
        print('Loss: {}\t Macro F1 score: {}'.format(train_loss, avg_val_metric))

        # log to tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/consi_loss", val_loss, epoch)
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
        test_loss, metrics = eval_baseline(model, test_loader, criterion, n_classes, metric_fn, baseline_type=base_type)

        print('#### Testing ####')
        print('Test Loss: ', test_loss)
        for key, val in metrics:
            print(f'Test {key}: {np.mean(val)}')
        
        # save metrics and model
        torch.save(model.state_dict(), os.path.join(run_dir, f'model_{i}.pth'))
        np.save(os.path.join(run_dir, f'metrics_{i}'), metrics)
        
        # jsonify metrics and write to json as well for manual inspection
        js = {}
        for key, val in metrics:
            if not np.ndim(val) == 0:
                js[key] = val.tolist()
            else:
                js[key] = val
        json.dump(js, f'metrics_{i}.json')
    
if __name__ == "__main__":
    
    """
    For now just initialize config here
    TODO: Load config from json file
    """
    # seeds = [0, 42, 345, 123, 45]
    seeds = [0]
    config = {
        'logdir': '../logs',
        'exp_name': 'baseline',
        'type': 0,
        'dataset': 'openmic',
        'mode': 0,
        'coarse': 0,
        'data_path': '../data',
        'hparams': {
            'lr': 0.001,
            'wd': 1e-5,
            'n_layers': 3,
            'dropout': 0.6,
            'num_epochs': 300,
            'batch_size': 100
        }
    }
    for seed in seeds:
        config['seed'] = seed
        run(config)