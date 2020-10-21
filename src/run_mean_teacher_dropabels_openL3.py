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

from utils import get_experiment_dir
from data.openmic_utils import get_openmic_loaders
from data.sonyc_utils import get_sonyc_loaders
from evaluate.eval_mt import eval_mt
from trainer.trainer_mt import trainer_mt
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
    writer = SummaryWriter(run_dir)
    
    # get data loaders and metrics function
    if config['dataset'] == 'openmic':
        (train_loader, val_loader, test_loader), _ = get_openmic_loaders(config)
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
    model_params = {'n_features': hparams['n_features'], 
                    'drop_rate':hparams['dropout'], 
                    'n_classes':n_classes, 
                    'n_layers':hparams['n_layers']}
    num_epochs = hparams['num_epochs']
    c_w = hparams['cw']
    alpha = hparams['alpha']

    # initialize models
    model = create_model(model_params)
    teacher = create_model(model_params, no_grad=True)
   
    # initialize criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # initialize best metric variables
    best_models = [None, None]
    best_teachers = [None, None]
    best_val_loss = 100000.0
    best_f1_macro = -1.0
    best_val_loss_t = 100000.0
    best_f1_macro_t = -1.0

    # training loop
    for epoch in tqdm(range(num_epochs)):
        # drop learning rate every 30 epochs
        if (epoch > 0) and (epoch % 30 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.5
                lr = lr * 0.5

        train_losses = trainer_mt(model, teacher, train_loader, optimizer, criterion, c_w, alpha, epoch)
        print('#### Training ####')
        print('Loss: {}'.format(train_losses))

        val_losses, metrics, metrics_t = eval_mt(model, teacher, val_loader, criterion, n_classes, metric_fn)
        val_metric = 'F1_macro' if config['dataset'] == 'openmic' else 'auprc_macro'
        avg_val_metric = np.mean(metrics[val_metric])
        avg_val_metric_t = np.mean(metrics_t[val_metric])
        print('#### Validation ####')
        print('Losses: {}\nMacro F1 score: {}\t Teacher macro F1 score: {}'.format(val_losses, avg_val_metric, avg_val_metric_t))

        # log to tensorboard
        writer.add_scalar("train/class_loss", train_losses[0], epoch)
        writer.add_scalar("train/consi_loss", train_losses[1], epoch)
        writer.add_scalar("val/class_loss", val_losses[0], epoch)
        writer.add_scalar("val/class_loss_teacher", val_losses[1], epoch)
        writer.add_scalar("val/consi_loss", val_losses[2], epoch)
        writer.add_scalar(f"val/{val_metric}", avg_val_metric, epoch)
        writer.add_scalar(f"val/{val_metric}_teacher", avg_val_metric_t, epoch)

        # Save best models
        if val_losses[0] < best_val_loss:
            best_val_loss = val_losses[0]
            best_models[0] = deepcopy(model)

        if avg_val_metric > best_f1_macro:
            best_f1_macro = avg_val_metric
            best_models[1] = deepcopy(model)

        # Save best teachers
        if val_losses[2] < best_val_loss_t:
            best_val_loss_t = val_losses[2]
            best_teachers[0] = deepcopy(model)

        if avg_val_metric_t > best_f1_macro_t:
            best_f1_macro_t = avg_val_metric_t
            best_teachers[1] = deepcopy(model)

    # Test best models
    for i, (model, teacher) in enumerate(zip(best_models, best_teachers)):
        test_losses, metrics, metrics_t = eval_mt(model, teacher, test_loader, criterion, n_classes, metric_fn)

        print('#### Testing ####')
        print('Test Loss: ', test_losses)
        for key, val in metrics.items():
            print(f'Test {key}: {np.mean(val)}')
        
        for key, val in metrics_t.items():
            print(f'Teacher Test {key}: {np.mean(val)}')
        
        # save metrics and model
        torch.save(model.state_dict(), os.path.join(run_dir, f'model_{i}.pth'))
        np.save(os.path.join(run_dir, f'metrics_{i}'), metrics)

        torch.save(teacher.state_dict(), os.path.join(run_dir, f'teacher_{i}.pth'))
        np.save(os.path.join(run_dir, f'teacher_metrics_{i}'), metrics)
        
        # jsonify metrics and write to json as well for manual inspection
        js = {}
        for key, val in metrics.items():
            if not np.ndim(val) == 0:
                js[key] = val.tolist()
            else:
                js[key] = val
        json.dump(js, open(os.path.join(run_dir, f'metrics_{i}.json'), 'w'))

        js = {}
        for key, val in metrics_t.items():
            if not np.ndim(val) == 0:
                js[key] = val.tolist()
            else:
                js[key] = val
        json.dump(js, open(os.path.join(run_dir, f'teacher_metrics_{i}.json'), 'w'))
    json.dump(config, open(os.path.join(run_dir, f'config.json'), 'w'))
    
if __name__ == "__main__":
    
    """
    For now just initialize config here
    TODO: Load config from json file
    """
    seeds = [0, 42, 345]
    baseline_type = [0, 1]
    # seeds = [42, 345]
    config = {
        'logdir': '../logs/OpenL3',
        'exp_name': 'mean_teacher_droplabels',
        'mode': 0,
        'coarse': 0,
        'data_path': '../data',
        'hparams': {
            'lr': 0.001,
            'wd': 1e-5,
            'n_layers': 1,
            'dropout': 0.6,
            'n_features': 512,
            'num_epochs': 100,
            'batch_size': 64,
            'cw': 3,
            'alpha': 0.999
        }
    }

    """
    For OpenMIC
    """
    # config['dataset'] = 'openmic'
    # for b_t in baseline_type:
    #     for seed in seeds:
    #         config['seed'] = seed
    #         config['type'] = b_t
    #         run(config)
    #     config.pop('seed')
    #     json.dump(config, open('../configs/baseline_{}_{}.json'.format(config['dataset'], b_t), 'w'))

    """
    For SONYC-UST:
    There are few missing labels in SONYC-UST.
    """
    label_drop_rates = [0.1, 0.2, 0.4, 0.8]
    config['dataset'] = 'sonyc'
    for drop_rate in label_drop_rates:
        for seed in seeds:
            config['label_drop_rate'] = drop_rate
            config['seed'] = seed
            run(config)
        