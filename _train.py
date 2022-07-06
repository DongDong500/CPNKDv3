import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
from metrics import StreamSegMetrics
from _get_dataset import _get_dataset
from _validate import _validate
from _load_model import _load_model

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def _train(opts, devices, run_id) -> dict:

    logdir = os.path.join(opts.Tlog_dir, 'run_' + str(run_id).zfill(2))
    writer = SummaryWriter(log_dir=logdir) 

    ### (1) Get datasets

    train_dst, val_dst, test_dst = _get_dataset(opts)
    
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dst, batch_size=opts.test_batch_size, num_workers=opts.num_workers, 
                                shuffle=True, drop_last=True)

    ### (2) Set up criterion

    if opts.loss_type == 'kd_loss':
        criterion = None
    else:
        raise NotImplementedError

    ### (3 -1) Load teacher & student models

    t_model = _load_model(opts=opts, model_name=opts.t_model, verbose=True,
                            pretrain=opts.t_model_params,
                            msg=" Teacher model selection: {}".format(opts.t_model),
                            output_stride=opts.t_output_stride, sep_conv=opts.t_separable_conv).to(devices)
    
    s_model = _load_model(opts=opts, model_name=opts.s_model, verbose=True,
                            msg=" Student model selection: {}".format(opts.s_model),
                            output_stride=opts.output_stride, sep_conv=opts.separable_conv)

    ### (4) Set up optimizer

    if opts.s_model.startswith("deeplab"):
        if opts.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "Adam":
            optimizer = torch.optim.Adam(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise NotImplementedError
    else:
        optimizer = optim.RMSprop(s_model.parameters(), 
                                    lr=opts.lr, 
                                    weight_decay=opts.weight_decay,
                                    momentum=opts.momentum)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    ### (5) Resume student model & scheduler

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        if torch.cuda.device_count() > 1:
            s_model = nn.DataParallel(s_model)
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        s_model.load_state_dict(checkpoint["model_state"])
        s_model.to(devices)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            resume_epoch = checkpoint["cur_itrs"]
            print("Training state restored from %s" % opts.ckpt)
        else:
            resume_epoch = 0
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
        torch.cuda.empty_cache()
    else:
        print("[!] Train from scratch...")
        resume_epoch = 0
        if torch.cuda.device_count() > 1:
            s_model = nn.DataParallel(s_model)
        s_model.to(devices)

    #### (6) Set up metrics

    metrics = StreamSegMetrics(opts.num_classes)
    early_stopping = utils.EarlyStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.save_ckpt, save_model=opts.save_model)
    dice_stopping = utils.DiceStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.save_ckpt, save_model=opts.save_model)
    best_score = 0.0

    ### (7) Train

    B_epoch = 0
    B_test_score = None

    for epoch in range(resume_epoch, opts.total_itrs):
        s_model.train()
        metrics.reset()
        running_loss = 0.0

        for (images, lbl) in train_loader:
            images = images.to(devices)
            lbl = lbl.to(devices)

            optimizer.zero_grad()
            
            s_outputs = s_model(images)
            probs = nn.Softmax(dim=1)(s_outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            t_outputs = t_model(images)
            
            weights = lbl.detach().cpu().numpy().sum() / (lbl.shape[0] * lbl.shape[1] * lbl.shape[2])
            weights = torch.tensor([weights, 1-weights], dtype=torch.float32).to(devices)
            criterion = utils.KDLoss(weight=weights, alpha=opts.alpha, temperature=opts.T)
            loss = criterion(s_outputs, t_outputs, lbl)
            loss.backward()
            
            optimizer.step()
            
            metrics.update(lbl.detach().cpu().numpy(), preds)
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        score = metrics.get_results()
        epoch_loss = running_loss / len(train_loader.dataset)

        print("[{}] Epoch: {}/{} Loss: {:.5f}".format('Train', epoch+1, opts.total_itrs, epoch_loss))
        print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
        print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
        print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        
