import numpy as np
import torch
import os
import glob
import deepdish as ddish
import random
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from nilearn.connectome import ConnectivityMeasure
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import csv
import math
from natsort import natsorted

def convert_Dloader(batch_size, data, label, num_workers = 0, shuffle = True):
    data, label = torch.tensor(data,  dtype=torch.float64), torch.from_numpy(label).long()
    dataset = TensorDataset(data, label)
    Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              num_workers=num_workers,
                                              batch_size=int(batch_size),
                                              shuffle=shuffle,
                                              drop_last=True
                                              )
    return Data_loader



def dataloader(args, fold, config=None, phase=None):

    # Please set the directory path Directory paths
    main_path = '/DataCommon2/eskang/ABIDE/'
    cond_path = main_path + 'orig/total_correlation_matrix_ori3.h5'
    # For 10 repetitions
    if args.seed_rp in [1210, 1218, 1403, 1993, 2012, 2017, 5164, 5874, 7777, 8888]:
        idxd_path = main_path + '32_10/idx%d/total_fold_idx3_%d.h5' % (args.seed_rp, fold - 1)
    else:
        idxd_path = main_path + '32_10/idx/total_fold_idx3_%d.h5' % (fold-1)

    conn_dict = ddish.io.load(cond_path)
    indx_dict = ddish.io.load(idxd_path)

    conn_d = conn_dict['data']
    conn_l = conn_dict['label']
    trnidx = indx_dict['tr_idx']
    validx = indx_dict['val_idx']
    tstidx = indx_dict['te_idx']

    if phase == 'tst':
        train_loader = convert_Dloader(trnidx.shape[0], conn_d[trnidx], conn_l[trnidx], num_workers=0, shuffle=False)
    else:
        if config:
            train_loader = convert_Dloader(config['bs'], conn_d[trnidx], conn_l[trnidx], num_workers=0, shuffle=True)
        elif config == None:
            train_loader = convert_Dloader(args.bs, conn_d[trnidx], conn_l[trnidx], num_workers=0, shuffle=True)

    val_loader = convert_Dloader(validx.shape[0], conn_d[validx], conn_l[validx], num_workers=0, shuffle=False)
    test_loader = convert_Dloader(tstidx.shape[0], conn_d[tstidx], conn_l[tstidx], num_workers=0, shuffle=False)

    return train_loader, val_loader, test_loader

