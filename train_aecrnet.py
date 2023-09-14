from re import L
import torch
import os
import time
import math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch
import warnings
from torch import nn
from option import opt, log_dir
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from metrics import psnr, ssim
from models.AECRNet import *
from models.CR import *
from data_utils.RS import *

import json


warnings.filterwarnings('ignore')

models_ = {
    'cdnet': Dehaze(3, 3),
}

loaders_ = {
    'RS_train': RS_train_loader,
    'RS_test': RS_test_loader,
    'dota_train': RSDota_train_loader,
    'dota_test': RSDota_test_loader
}

start_time = time.time()
model_name = opt.model_name
T = opt.epochs


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, optim, criterion, epoch):
    net.train()
    epoch_loss = 0

    for iteration, batch in enumerate(loader_train, 1):
        input = batch[0]
        gt = batch[1]
        input = input.to(opt.device)
        gt = gt.to(opt.device)

        out = net(input)

        loss_vgg7, all_ap, all_an, loss_rec = 0, 0, 0, 0
        if opt.w_loss_l1 > 0:
            loss_rec = criterion[0](out, gt)
        if opt.w_loss_vgg7 > 0:
            loss_vgg7, all_ap, all_an = criterion[1](out, gt, input)

        loss = opt.w_loss_l1*loss_rec + opt.w_loss_vgg7*loss_vgg7
        epoch_loss += loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        if iteration % 10 == 0:
            print(f'\rEpoch[{epoch}]({iteration}/{len(loader_train)}) loss:{loss.item():.5f} l1:{opt.w_loss_l1*loss_rec:.5f} contrast: {opt.w_loss_vgg7*loss_vgg7:.5f} all_ap:{all_ap:.5f} all_an:{all_an:.5f}|time_used :{(time.time() - start_time) / 60 :.1f}', end='', flush=True)

    return epoch_loss / len(loader_train)


def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            pred = net(inputs)

        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)

    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed_torch(666)
    start_epoch = 0
    losses = []
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    net = models_[opt.net]
    net = net.to(opt.device)

    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(ablation=opt.is_ab))
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters(
    )), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()

    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]

    pytorch_total_params = sum(p.numel()
                               for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    if not opt.resume and os.path.exists(f'./logs_train/{opt.model_name}.txt'):
        print(f'./logs_train/{opt.model_name}.txt 已存在，请删除该文件……')
        exit()

    with open(f'./logs_train/args_{opt.model_name}.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    if opt.resume and (os.path.exists(opt.model_dir) or opt.pre_model != 'null'):
        if opt.pre_model != 'null':
            ckp = torch.load('./trained_models/'+opt.pre_model)
            print(f'resume from pre_model')
        else:
            ckp = torch.load(opt.model_dir)
            print(f'resume from {opt.model_dir}')

        net.load_state_dict(ckp['model'])

        if opt.pre_model == 'null':
            optimizer.load_state_dict(ckp['optimizer'])
            start_epoch = ckp['epoch'] + 1
            max_ssim = ckp['max_ssim']
            max_psnr = ckp['max_psnr']
            psnrs = ckp['psnrs']
            ssims = ckp['ssims']
            losses = ckp['losses']

        print(f'max_psnr: {max_psnr}, max_ssim: {max_ssim}')
        print(f'start_epoch:{start_epoch}, start training ---')
    else:
        print('train from scratch *** ')


    for epoch in range(start_epoch, opt.epochs):
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(epoch, T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        loss = train(net, loader_train, optimizer, criterion, epoch)
        losses.append(loss)

        if epoch % opt.eval_step == 0:
            save_model_dir = opt.model_dir
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)

            log = f'\nepoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'

            print(log)
            with open(f'./logs_train/{opt.model_name}.txt', 'a') as f:
                f.write(log + '\n')

            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                save_model_dir = opt.model_dir.split('.')[0] + '_best.pk'

            torch.save({
                'epoch': epoch,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_model_dir)

            print(f'\n model saved at epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{T}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{T}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{T}_psnrs.npy', psnrs)


if __name__ == "__main__":
    main()
