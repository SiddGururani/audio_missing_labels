import numpy as np

def get_experiment_dir(config):
    logdir = config['logdir']
    exp_name = config['exp_name']
    if 'baseline' in exp_name:
        exp_name += '_{}'.format(config['type'])
    dataset = config['dataset']
    if dataset != 'openmic':
        dataset += '_MODE_{}_{}'.format(config['mode'], 'coarse' if config['coarse'] else 'fine')
        if 'label_drop_rate' in config:
            dataset += '_label_drop_rate_{}'.format(config['label_drop_rate'])

    hparams = ''
    for key, value in config['hparams'].items():
        hparams += '_{}_{}'.format(key, value)
    return f'{logdir}/{exp_name}_{dataset}_{hparams}'

def get_enhanced_labels(Y_mask, all_predictions, prune_thres):
    # get the top prune_thres percentile of implicit negatives per class and mask them out
    new_mask = np.ones_like(Y_mask)
    for i in range(Y_mask.shape[-1]):
        class_mask, class_preds = Y_mask[:, i], all_predictions[:, i]
        impl_negs = np.where(class_mask == 0)
        preds_for_impl_negs = class_preds[class_mask == 0]
        sorted_inds = np.argsort(preds_for_impl_negs)
        num_prune = int(prune_thres * len(sorted_inds))
        prune_inds = sorted_inds[-num_prune:]
        new_mask[impl_negs[0][prune_inds], i] = 0
    return new_mask

def to_numpy(x):
    return x.detach().cpu().numpy()
    