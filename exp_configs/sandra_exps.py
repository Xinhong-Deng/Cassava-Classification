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



efficientnet_smaller_batch = [{
                        'batch_size': 20,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'tf_efficientnet_b4_ns'},
                        'loss_func': cross_entropy[0],
                        'max_epoch': 30,
                        'train_transform': {'name': 'tf4', 'im_size': 512},
                        'val_transform': {'name': 'tf2', 'im_size': 512},
}]

torchvision_transform = [{
                        'batch_size': 20,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'loss_func': {'name': 'symmetric_cross_entropy', 'alpha': 0.1, 'beta': 1.0},
                        'max_epoch': 30,
                        'train_transform': {'name': 'centercrop', 'im_size': 256},
                        'val_transform': {'name': 'default', 'im_size': 256},
                        }]

EXP_GROUPS['bitemperednew'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'loss_func': {'name': 'bitempered', 't1': 0.2, 't2': 1.1, 'reduction': 'mean'},
                        'max_epoch': [30],
                        'train_transform': [{'name': tf_name, 'im_size': 512} for tf_name in ['tf1', 'tf3']],
                        'val_transform': {'name': 'tf2', 'im_size': 512},
}) + torchvision_transform

EXP_GROUPS['transform'] = hu.cartesian_exp_group({
                        'batch_size': 20,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'resnext50_32x4d_ssl'},
                        'loss_func': cross_entropy,
                        'max_epoch': 30,
                        'train_transform': [{'name': tf_name, 'im_size': 512} for tf_name in ['tf1', 'tf3']],
                        'val_transform': {'name': 'tf2', 'im_size': 512},
}) + torchvision_transform



EXP_GROUPS['vit'] = hu.cartesian_exp_group({
                        'batch_size': 15,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'vit_base_patch16_224'},
                        'loss_func': cross_entropy,
                        'max_epoch': [40],
                        'train_transform': [{'name': tf_name, 'im_size': 224} for tf_name in ['tf1', 'tf3', 'default']],
                        'val_transform': {'name': 'default', 'im_size': 224},
})

EXP_GROUPS['spiralcnn'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'adamW', 'lr': 0.0001, 'wd': 1e-6},
                        'model': {'name': 'spiralcnn'},
                        'loss_func': cross_entropy[0],
                        'max_epoch': [50]
                        })

exp_sam_effinet = [{'batch_size': 32,
                        'opt': {'name': 'sam', 'lr':  0.0001, 'wd': 1e-6, 'rho': 0.05, 'base': 'adamw'},
                        'loss_func': cross_entropy[0],
                        'model': {'name': 'tf_efficientnet_l2_ns_475', 'use_sam': True},
                        'max_epoch': 50,
                        'train_transform': {'name': 'default', 'im_size': 475},
                        'val_transform': {'name': 'default', 'im_size': 475},
}]

exp_sam_resnext = [{'batch_size': 32,
                        'opt': {'name': 'sam', 'lr':  0.0001, 'wd': 1e-6, 'rho': 0.05, 'base': 'adamw'},
                        'loss_func': cross_entropy[0],
                        'model': {'name': 'resnext50_32x4d_ssl', 'use_sam': True},
                        'max_epoch': 50,
                        'train_transform': {'name': 'default', 'im_size': 475},
                        'val_transform': {'name': 'default', 'im_size': 475},
}]


EXP_GROUPS['sam'] = hu.cartesian_exp_group({
                        'batch_size': 32,
                        'opt': {'name': 'sam', 'lr':  0.0001, 'wd': 1e-6, 'rho': 0.05, 'base': 'adamw'},
                        'loss_func': cross_entropy,
                        'model': {'name': 'resnext50_32x4d_ssl', 'use_sam': True},
                        'max_epoch': [50],
                        'train_transform': {'name': 'default', 'im_size': 256},
                        'val_transform': {'name': 'default', 'im_size': 256},
                        }) + exp_sam_resnext +exp_sam_effinet


