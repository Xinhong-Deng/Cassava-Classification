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

EXP_GROUPS['efficientnet'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'tf_efficientnet_b4_ns'},
                        'loss_func': cross_entropy,
                        'max_epoch': [50]
                        })

EXP_GROUPS['effnet2'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': [{'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6}, {'name': 'adam', 'lr': 0.0001, 'wd': 1e-6}],
                        'model': [{'name': 'tf_efficientnet_b4_ns'}, {'name': 'tf_efficientnet_b7_ns'}],
                        'loss_func': cross_entropy + [{'name': 'symmetric_cross_entropy', 'alpha': 1.0, 'beta': 5.0}],
                        'max_epoch': [100]
                        })

EXP_GROUPS['bitempered'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'loss_func': {'name': 'bitempered', 't1': 1, 't2': 1, 'reduction': 'mean'},
                        'max_epoch': [30]
                        })

efficientnet_smaller_batch = [{
                        'batch_size': 20,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'tf_efficientnet_b4_ns'},
                        'loss_func': cross_entropy[0],
                        'max_epoch': 30,
                        'train_transform': {'name': 'tf4', 'im_size': 512},
                        'val_transform': {'name': 'tf2', 'im_size': 512},
}]

EXP_GROUPS['transform'] = hu.cartesian_exp_group({
                        'batch_size': 20,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'loss_func': cross_entropy,
                        'max_epoch': [30],
                        'train_transform': [{'name': tf_name, 'im_size': 512} for tf_name in ['tf1', 'tf3']],
                        'val_transform': {'name': 'tf2', 'im_size': 512},
})
# }) + efficientnet_smaller_batch

EXP_GROUPS['spiralcnn'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'spiralcnn'},
                        'loss_func': cross_entropy,
                        'max_epoch': [50]
                        })

EXP_GROUPS['sam'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'sam'},
                        'loss_func': cross_entropy,
                        'model': {'name': 'resnet'},
                        'max_epoch': [50]
                        })














