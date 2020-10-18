import sys
import math
import random
from pprint import pprint
from copy import deepcopy

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..trainer.train_utils import *

def eval_mt(model, teacher, data_loader, criterion, n_classes, metric_fn):
    model.eval()
    teacher.eval()

    class_loss_tracker = AverageMeter()
    consi_loss_tracker = AverageMeter()
    t_class_loss_tracker = AverageMeter()
    
    all_y_true = torch.Tensor(0, n_classes)
    all_y_mask = torch.BoolTensor(0, n_classes)
    all_predictions = torch.Tensor(0, n_classes)
    all_t_predictions = torch.Tensor(0, n_classes)

    for i, (X, Y_true, Y_mask) in tqdm(enumerate(data_loader)):
        X = X.cuda()
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()
            
        outputs = model(X)
        outputs_ = teacher(X)
        
        # Regardless of what criterion or whether this is instrument-wise
        # Let the criterion function deal with it
        class_loss,_ = criterion(outputs, Y_true, Y_mask)
        t_class_loss,_ = criterion(outputs_, Y_true, Y_mask)
        
        consistency_loss = F.mse_loss(outputs, outputs_)

        # Update average meter
        class_loss_tracker.update(class_loss.item())
        consi_loss_tracker.update(consistency_loss.item())
        t_class_loss_tracker.update(t_class_loss.item())
        
        # Store the outputs and target for classification metric computation
        all_y_true = torch.cat((all_y_true, Y_true.detach().cpu()))
        all_y_mask = torch.cat((all_y_mask, Y_mask.detach().cpu()))
        all_predictions = torch.cat((all_predictions, outputs.detach().cpu()))
        all_t_predictions = torch.cat((all_t_predictions, outputs_.detach().cpu()))
    metrics = metric_fn(all_y_true, all_y_mask, all_predictions)
    t_metrics = metric_fn(all_y_true, all_y_mask, all_t_predictions)

    return (class_loss_tracker.avg, consi_loss_tracker.avg, t_class_loss_tracker.avg), \
            metrics, t_metrics
