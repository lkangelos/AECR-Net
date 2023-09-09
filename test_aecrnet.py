import os
import torch
import numpy as np
from data_utils.RS import RSDota_test_loader
from metrics import psnr, ssim
from models.AECRNet import Dehaze
from torch.backends import cudnn
from option import opt


models_={
	'cdnet': Dehaze(3, 3),
}

loaders_={
	'dota_test': RSDota_test_loader
}


def test(net,loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test):
            inputs = inputs.to(opt.device);targets = targets.to(opt.device)
            pred = net(inputs)

            ssim1 = ssim(pred, targets).item()
            psnr1 = psnr(pred, targets)
            ssims.append(ssim1)
            psnrs.append(psnr1)

    return np.mean(ssims), np.mean(psnrs)


def main():
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    if opt.resume and os.path.exists(opt.model_dir):
        if opt.pre_model != 'null':
            ckp = torch.load('./trained_models/'+opt.pre_model)
        else:
            opt.model_dir = opt.model_dir.split('.pk')[0] + '_best.pk'
            ckp = torch.load(opt.model_dir)

        print(f'resume from {opt.model_dir}')
        net.load_state_dict(ckp['model'])
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        print(f'max_psnr: {max_psnr} max_ssim: {max_ssim}')
    else:
        print('expect --resume!!!')
        os._exit(0)

    ssim_eval, psnr_eval = test(net, loader_test)
    print("PSNR:{} SSIM:{}".format(psnr_eval, ssim_eval))


if __name__ == "__main__":
    main()