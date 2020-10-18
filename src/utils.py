def get_experiment_dir(config):
    logdir = config['logdir']
    exp_name = config['exp']
    if exp_name == 'baseline':
        exp_name += '_{}'.format(config['type'])
    dataset = config['dataset']
    if dataset != 'openmic':
        dataset += '_MODE_{}_{}'.format(config['mode'], 'coarse' if config['coarse'] else 'fine')
    hparams = ''
    for key, val in config['hparams']:
        hparams += '_{}_{}'.format(key, val)
    return f'{logdir}_{exp_name}_{dataset}_{hparams}'