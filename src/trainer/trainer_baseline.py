from tqdm import tqdm
from trainer.train_utils import *

def trainer_baseline(model, data_loader, optimizer, criterion, baseline_type=0):
    model.train()
    loss_tracker = AverageMeter()
    for (X, Y_true, Y_mask) in tqdm(data_loader):

        # Check if batch has any labels
        if Y_mask.sum() == 0:
            continue

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
        
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # Update average meter
        loss_tracker.update(loss.item())
    return loss_tracker.avg