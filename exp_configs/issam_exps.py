from haven import haven_utils as hu
import itertools, copy

EXP_GROUPS = {}


EXP_GROUPS['starter_issam'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'loss_func': {'name': 'cross_entropy'},
                        'max_epoch': [50]
                        })

EXP_GROUPS['clip'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'model': {'name': 'clip'},
                        'max_epoch': [30],
                        })