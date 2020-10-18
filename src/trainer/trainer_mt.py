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

def trainer(model, teacher, data_loader, optimizer, criterion, c_w, alpha, epoch_num):
    global_step = epoch_num * len(data_loader)
    model.train()
    teacher.train()
    class_loss_tracker = AverageMeter()
    consi_loss_tracker = AverageMeter()
    t_class_loss_tracker = AverageMeter()
    
    for i, (X, Y_true, Y_mask) in tqdm(enumerate(data_loader)):

        X = X.cuda()
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()

        outputs = model(X)
        # print('outputted')
        
        # Regardless of what criterion or whether this is instrument-wise
        # Let the criterion function deal with it
        class_loss,_ = criterion(outputs, Y_true, Y_mask)
        
        # Compute consistency loss here
        outputs_ = teacher(X)
        consistency_loss = F.mse_loss(outputs, outputs_)
        
        loss = class_loss + c_w*consistency_loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        # Update teacher
        global_step += 1
        update_ema_variables(model, teacher, global_step, alpha)
        t_class_loss,_ = criterion(outputs_, Y_true, Y_mask)
                                 
        # Update average meters
        class_loss_tracker.update(class_loss.item())
        consi_loss_tracker.update(consistency_loss.item())
        t_class_loss_tracker.update(t_class_loss.item())
    return class_loss_tracker.avg, consi_loss_tracker.avg, t_class_loss_tracker.avg