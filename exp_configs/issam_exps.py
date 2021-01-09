from haven import haven_utils as hu
import itertools, copy

EXP_GROUPS = {}


EXP_GROUPS['starter_issam'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'lr': 0.0001,
                        'max_epoch': [20]
                        })
