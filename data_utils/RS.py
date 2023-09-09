import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt

BS = opt.bs
print(BS)
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size

class RSHAZE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        print(f'path={path}')
        super(RSHAZE_Dataset,self).__init__()
        self.size = size
        print('crop size',size)
        self.train=train

        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'input'))
        self.haze_imgs=[os.path.join(path,'input',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):

        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,9000)
                haze=Image.open(self.haze_imgs[index])        
        img=self.haze_imgs[index]
        clear_name=img.split('/')[-1]
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear

    def augData(self, data, target):

        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90*rand_rot)
                target = FF.rotate(target, 90*rand_rot)

        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)

root = '/home/louanqi/pycharmp/data'

RS_train_loader=DataLoader(dataset=RSHAZE_Dataset(root+'/rshaze/train', train=True,size=crop_size),batch_size=BS,shuffle=True)
RS_test_loader=DataLoader(dataset=RSHAZE_Dataset(root+'/rshaze/test',train=False,size='whole img'),batch_size=1,shuffle=False)

RSDota_train_loader=DataLoader(dataset=RSHAZE_Dataset(root+'/dota2/rsdota3000',train=True,size=crop_size),batch_size=BS,shuffle=True)
RSDota_test_loader=DataLoader(dataset=RSHAZE_Dataset(root+'/dota2/rsdota3000',train=False,size='whole img'),batch_size=1,shuffle=False)

# for debug
#ITS_test = '/home/why/workspace/CDNet/net/debug/test_h5/'
#ITS_test_loader_debug=DataLoader(dataset=RESIDE_Dataset(ITS_test,train=False,size='whole img'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass
