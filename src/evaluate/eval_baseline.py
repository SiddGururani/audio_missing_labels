import sys
import math
import random
from copy import deepcopy
from tqdm import tqdm

import numpy
import torch
import torch.nn as nn
from pprint import pprint
from trainer.train_utils import *
from utils import to_numpy

def eval_baseline(model, data_loader, criterion, n_classes, metric_fn, return_preds=False, baseline_type=0):
    model.eval()
    loss_tracker = AverageMeter()

    all_y_true = torch.Tensor(0, n_classes)
    all_y_mask = torch.BoolTensor(0, n_classes)
    all_predictions = torch.Tensor(0, n_classes)
    for (X, Y_true, Y_mask) in tqdm(data_loader):

        X = X.cuda()
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()

        outputs = model(X)
        if baseline_type == 0:
            # All missing labels are explicit negatives
            Y_true[Y_mask == 0] = 0
            loss = criterion(outputs, Y_true)
        else:
            # All missing labels are ignored/masked in loss
            loss = criterion(outputs[Y_mask], Y_true[Y_mask])
        
        # Store the outputs and target for classification metric computation
        all_y_true = torch.cat((all_y_true, Y_true.detach().cpu()))
        all_y_mask = torch.cat((all_y_mask, Y_mask.detach().cpu()))
        all_predictions = torch.cat((all_predictions, outputs.detach().cpu()))

        # Update average meter
        loss_tracker.update(loss.item())
    
    all_y_mask = to_numpy(all_y_mask)
    all_y_true = to_numpy(all_y_true)
    all_predictions = to_numpy(all_predictions)

    metrics = metric_fn(all_y_true, all_y_mask, all_predictions)
    if return_preds:
        return loss_tracker.avg, metrics, (all_y_true, all_y_mask, all_predictions)
    return loss_tracker.avg, metrics

def forward(model, data_loader, n_classes):
    model.eval()
    all_predictions = torch.Tensor(0, n_classes)
    for X in data_loader:
        X = X[0].cuda()
        outputs = model(X)
        all_predictions = torch.cat((all_predictions, outputs.detach().cpu()))

    return to_numpy(all_predictions)