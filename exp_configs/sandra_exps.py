from haven import haven_utils as hu
import itertools, copy

EXP_GROUPS = {}

symmetric_cross_entropy = [{'name': 'symmetric_cross_entropy', 'alpha': alpha, 'beta': beta} for alpha in [0.1, 0.01, 1] for beta in [1, 0.5, 5]]
cross_entropy = [{'name': 'cross_entropy'}]
loss_functions = symmetric_cross_entropy + cross_entropy

EXP_GROUPS['resnet'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adam', 'lr': 0.0001,},
                        'loss_func': cross_entropy,
                        'model': {'name': 'resnet'},
                        'max_epoch': [50]
                        })

EXP_GROUPS['resnext'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'loss_func': loss_functions,
                        'max_epoch': [50]
                        })
