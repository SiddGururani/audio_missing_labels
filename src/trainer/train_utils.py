import math

from ..model.Attention import DecisionLevelSingleAttention

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch, num_epochs, rampdown_length):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0

def update_ema_variables(model, ema_model, global_step, alpha=0.999):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    if not hasattr(model, 'parameters'):
        ema_model.data.mul_(alpha).add_(1 - alpha, model.data)
        return
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def create_model(params, no_grad=False):
    model = DecisionLevelSingleAttention(128,
                                        params['n_classes'],
                                        params['n_layers'],
                                        128,
                                        params['drop_rate'])
    
    model = model.cuda()
    if no_grad:
        for param in model.parameters():
            param.detach_().requires_grad_(False)
    return model