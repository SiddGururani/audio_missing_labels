def get_experiment_dir(config):
    logdir = config['logdir']
    exp_name = config['exp_name']
    if exp_name == 'baseline':
        exp_name += '_{}'.format(config['type'])
    dataset = config['dataset']
    if dataset != 'openmic':
        dataset += '_MODE_{}_{}'.format(config['mode'], 'coarse' if config['coarse'] else 'fine')
    hparams = ''
    for key, value in config['hparams'].items():
        hparams += '_{}_{}'.format(key, value)
    return f'{logdir}/{exp_name}_{dataset}_{hparams}'