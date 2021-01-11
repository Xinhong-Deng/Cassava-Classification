from haven import haven_utils as hu
import itertools, copy

EXP_GROUPS = {}


EXP_GROUPS['resnet'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'lr': 0.0001,},
                        'max_epoch': [20]
                        })

EXP_GROUPS['resnext'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'max_epoch': [20]
                        })
