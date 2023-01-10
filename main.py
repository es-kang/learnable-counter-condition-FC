import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from settings import setting as setting
import dataset.dataload as ld

from models.model_adaptive_mask import AdaptiveMask
from models.model_prototype import ProtoClassifier
from models.model_relation import FCTransformer
from models.model_decoder import CLS_Decoder
import utils.load as lo
import utils.make as mk
from utils.make import save_args
from experiments.exp import train, evaluate


def main(args, paths_all, f):

    gpu_id = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Fix the seed
    seed1 = args.seed
    np.random.seed(seed1)
    os.environ["PYTHONHASHSEED"] = str(seed1)
    torch.cuda.manual_seed(seed1)
    torch.cuda.manual_seed_all(seed1)  # if you are using multi-GPU
    torch.manual_seed(seed1)
    random.seed(seed1)
    torch.set_default_dtype(torch.float64)

    model_df = AdaptiveMask(args).to(device)
    model_tf = FCTransformer(args).to(device)
    model_cl = ProtoClassifier(args).to(device)
    model_dc = CLS_Decoder(args).to(device)

    train_data, valid_data, test_data = ld.dataloader(args, f)
    main_path, tpath, mpath, vpath = paths_all

    model_list = [model_df, model_tf, model_cl, model_dc]
    model_list, optimizer, optimizer2 = lo.load_optimizer2(args, model_list)

    criterion_cls = nn.CrossEntropyLoss(reduction="mean").to(device)
    criterion_rec = nn.L1Loss(reduction="mean").to(device)
    loss_list = [criterion_cls, criterion_rec]
    opts_list = [optimizer, optimizer2]

    writer = SummaryWriter(tpath + 'f%d' % f)
    best_bal2 = float("inf")
    val_result = [[0]]
    for epoch in range(1, args.epoch + 1):

        if val_result[0][0] > 0.6:
            # Train the prototype-based FC when the accuracy is over 0.6 (step 2)
            train(args, epoch, device, model_list, opts_list, train_data, loss_list, True)
            val_loss, val_result = evaluate(args, device, epoch, model_list, valid_data, loss_list, 'val')
            tst_loss, tst_result = evaluate(args, device, epoch, model_list, test_data, loss_list, 'tst')
            print("*****************************************************************************************")

        else:
            # Step1
            train(args, epoch, device, model_list, opts_list, train_data, loss_list, False)
            val_loss, val_result = evaluate(args, device, epoch, model_list, valid_data, loss_list, 'val')
            tst_loss, tst_result = evaluate(args, device, epoch, model_list, test_data, loss_list, 'tst')

        if val_loss[0] < best_bal2 and abs(val_result[0][2] - val_result[0][3]) <= 0.1 and epoch % 2 == 0:
            best_bal2 = val_loss[0]
            best_ep_bal2 = epoch
            best_perf_bal2 = tst_result[0]
            torch.save({'model_df': model_df.state_dict(),
                        'model_tf': model_tf.state_dict(),
                        'model_cl': model_cl.state_dict(),
                        'model_dc': model_dc.state_dict()}, mpath + "best_bal_0.1.pt")

        writer.add_scalars('ALL', {"valid": val_loss[0], "test": tst_loss[0]}, epoch)
        writer.add_scalars('REC', {"valid": val_loss[1], "test": tst_loss[1]}, epoch)
        writer.add_scalars('CLS', {"valid": val_loss[2], "test": tst_loss[2]}, epoch)
        writer.add_scalars('ACC', {"valid": val_result[0][0], "test": tst_result[0][0]}, epoch)

    best_epoch = np.unique(np.asarray([best_ep_bal2]))
    # remove_file_train(mpath, best_epoch, args.epoch + 1)

    final_bal2 = [best_ep_bal2, best_perf_bal2[1], best_perf_bal2[0], best_perf_bal2[2], best_perf_bal2[3]]

    return final_bal2


if __name__ == "__main__":

    args = setting.get_args()

    # set the save path
    pre_path = '/main/lr_df_{}_tf_{}_pr_{}_dc_{}/h_cls{}_rec{}_sca{}_roi{}/df_tau_{}_lr_dc2_{}_h_cls2_{}_h_rec2_{}_ep{}_bs{}/rp{}_seed{}/'.format(
                args.lr_df, args.lr_tf, args.lr_pr, args.lr_dc,
                args.h_cls, args.h_rec, args.h_sca, args.h_roi,
                args.df_tau, args.lr_dc2, args.h_cls2, args.h_rec2, args.epoch, args.bs,
                args.seed_rp, args.seed)

    log_path = mk.mk_paths(args, 0, pre_path)[0]
    save_args(log_path, args)
    summary = open(log_path + 'summary.txt', 'w', encoding='utf-8')
    summary.write("Criterion\tAUC\tACC\tSEN\tSPC\n")

    arr_perf = []
    for f in [1,2,3,4,5]:
        print("Fold", f)
        paths = mk.mk_paths(args, f, pre_path)

        final = main(args, paths, f)

        arr_perf.append(final)

        print("Fold Results", final)

    perf = np.array(arr_perf).mean(0)

    np.savez(log_path + 'results_fold', VAL_LOS=np.array(arr_perf))
    mk.writelog(summary, 'Results\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(perf[1], perf[2], perf[3], perf[4]))
    print(pre_path)