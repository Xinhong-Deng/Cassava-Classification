from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # ==================
    # train set
    train_loader = datasets.get_loader(split="train",
                                     exp_dict=exp_dict,
                                     datadir=args.datadir)

    val_loader = datasets.get_loader(split="val",
                                     exp_dict=exp_dict,
                                     datadir=args.datadir)

    test_loader = datasets.get_loader(split="test",
                                     exp_dict=exp_dict,
                                     datadir=args.datadir)

    # Model
    # ==================
    model = models.Model(exp_dict)

    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % (s_epoch))
    
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    for e in range(s_epoch, exp_dict['max_epoch']):
        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate the model
        val_dict = model.val_on_loader(val_loader)

        # get score dict
        score_dict = {}
        score_dict.update(train_dict)
        score_dict.update(val_dict)
        score_dict["epoch"] = e

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
 
        print("\n", score_df.tail(), "\n")
        hu.torch_save(model_path, model.network.state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

    print('Experiment completed et epoch %d' % score_dict['epoch'])


if __name__ == "__main__":
    from haven import haven_wizard as hw
    import exp_configs
    if os.path.exists('job_configs.py'):
        import job_configs
        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None 

    hw.run_wizard(func=trainval, 
                  exp_groups=exp_configs.EXP_GROUPS, 
                  job_config=job_config)

