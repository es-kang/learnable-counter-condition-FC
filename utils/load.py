import torch
import numpy as np


def load_optimizer2(args, model_list):
    model_df, model_tf, model_cls, model_dec = model_list[0], model_list[1], model_list[2], model_list[3]
    optimizer = torch.optim.AdamW([{"params": model_df.parameters(), "lr": args.lr_df},
                                   {"params": model_tf.intra_net.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.encoding_block_sa.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.cls_token, 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_cls.parameters(), 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_dec.parameters(), "lr": args.lr_dc},
                                   ], lr=args.lr_tf, weight_decay=args.l2)
    optimizer2 = torch.optim.AdamW([{"params": model_dec.parameters(), "lr": args.lr_dc2}], lr=args.lr_dc2, weight_decay=args.l2)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=args.h_sch)
    scheduler = None
    return [model_df, model_tf, model_cls, model_dec], optimizer, optimizer2



def load_optimizer2_wointra(args, model_list):
    model_df, model_tf, model_cls, model_dec = model_list[0], model_list[1], model_list[2], model_list[3]
    optimizer = torch.optim.AdamW([{"params": model_df.parameters(), "lr": args.lr_df},
                                   {"params": model_tf.encoding_block_sa.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.cls_token, 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_cls.parameters(), 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_dec.parameters(), "lr": args.lr_dc},
                                   ], lr=args.lr_tf, weight_decay=args.l2)
    optimizer2 = torch.optim.AdamW([{"params": model_dec.parameters(), "lr": args.lr_dc2}], lr=args.lr_dc2, weight_decay=args.l2)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=args.h_sch)
    scheduler = None
    return [model_df, model_tf, model_cls, model_dec], optimizer, optimizer2


def load_optimizer2_wodf(args, model_list):
    model_tf, model_cls, model_dec = model_list[0], model_list[1], model_list[2]
    optimizer = torch.optim.AdamW([
                                   {"params": model_tf.intra_net.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.encoding_block_sa.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.cls_token, 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_cls.parameters(), 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_dec.parameters(), "lr": args.lr_dc},
                                   ], lr=args.lr_tf, weight_decay=args.l2)
    optimizer2 = torch.optim.AdamW([{"params": model_dec.parameters(), "lr": args.lr_dc2}], lr=args.lr_dc2, weight_decay=args.l2)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=args.h_sch)
    scheduler = None
    return [model_tf, model_cls, model_dec], optimizer, optimizer2


def load_optimizer_config(config, args, model_list):
    model_df, model_tf, model_cls, model_dec = model_list[0], model_list[1], model_list[2], model_list[3]
    optimizer = torch.optim.AdamW([ {"params": model_df.parameters(), "lr": config['lr_df']},
                                   {"params": model_tf.intra_net.parameters(), "lr": config['lr_tf']},
                                   {"params": model_tf.encoding_block_sa.parameters(), "lr": config['lr_tf']},
                                   {"params": model_tf.cls_token, 'lr': config['lr_tf'], "weight_decay": 0},
                                   {"params": model_cls.parameters(), 'lr': config['lr_pr'], "weight_decay": 0},
                                   {"params": model_dec.parameters(), "lr": config['lr_dc']},
                                   ], lr=args.lr_tf, weight_decay=config['l2'])
    optimizer2 = torch.optim.AdamW([{"params": model_dec.parameters(), "lr": config['lr_dc2']}], lr=args.lr_dc2, weight_decay=config['l2'])

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=args.h_sch)
    scheduler = None
    return [model_df, model_tf, model_cls, model_dec], optimizer, optimizer2

def load_optimizer2_wointra(args, model_list):
    model_df, model_tf, model_cls, model_dec = model_list[0], model_list[1], model_list[2], model_list[3]
    optimizer = torch.optim.AdamW([{"params": model_df.parameters(), "lr": args.lr_df},
                                   {"params": model_tf.encoding_block_sa.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.cls_token, 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_cls.parameters(), 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_dec.parameters(), "lr": args.lr_dc},
                                   ], lr=args.lr_tf, weight_decay=args.l2)
    optimizer2 = torch.optim.AdamW([{"params": model_dec.parameters(), "lr": args.lr_dc2}], lr=args.lr_dc2, weight_decay=args.l2)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=args.h_sch)
    scheduler = None
    return [model_df, model_tf, model_cls, model_dec], optimizer, optimizer2

def load_optimizer_dc(args, model_list, saved_model_dict):
    model_df, model_tf, model_cls, model_dec = model_list[0], model_list[1], model_list[2], model_list[3]

    model_dict = model_df.state_dict()
    saved_dict = saved_model_dict['model_df']
    model_dict.update(saved_dict)
    model_df.load_state_dict(model_dict)

    model_dict = model_tf.state_dict()
    saved_dict = saved_model_dict['model_tf']
    model_dict.update(saved_dict)
    model_tf.load_state_dict(model_dict)

    model_dict = model_cls.state_dict()
    saved_dict = saved_model_dict['model_cl']
    model_dict.update(saved_dict)
    model_cls.load_state_dict(model_dict)

    model_dict = model_dec.state_dict()
    saved_dict = saved_model_dict['model_dc']
    model_dict.update(saved_dict)
    model_dec.load_state_dict(model_dict)

    optimizer = torch.optim.AdamW([{"params": model_dec.parameters(), "lr": args.lr_dc2},
                                   ], lr=args.lr_tf, weight_decay=args.l2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=args.h_sch)
    scheduler = None
    return [model_df, model_tf, model_cls, model_dec], optimizer, scheduler