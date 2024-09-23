import cv2
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import os.path as osp

from torchvision import transforms as pth_transforms

class MOVi(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/root/onethingai-tmp/data/movi_c',
                 split='train',
                 resolution=224):
        self.img_root = osp.join(data_root,split,'video')
        self.img_infos = sorted(os.listdir(self.img_root))
        self.resolution = resolution
        # self.transform = pth_transforms.Compose([
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        img = cv2.resize(cv2.imread(filename), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img


class MOVi_test(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/root/onethingai-tmp/data/movi_c',
                 split='train',
                 resolution=224):
        self.img_root = osp.join(data_root,split,'video')
        self.ann_root = osp.join(data_root,split,'seg')
        self.img_infos = sorted(os.listdir(self.img_root))

        # self.transform = pth_transforms.Compose([
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resolution = resolution
        self.max_mse = 1
        self.target_root = ''

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = '/root/onethingai-tmp/data/movi_c/val/video/00017/0.jpg' # osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        img = cv2.resize(cv2.imread(filename), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        annoname = '/root/onethingai-tmp/data/movi_c/val/seg/00017/0.png' # osp.join(self.ann_root, self.img_infos[vid_idx], str(img_idx)+'.png')
        ann = cv2.imread(annoname)[:,:,0]
        ann = cv2.resize(ann, (self.resolution,self.resolution), interpolation=cv2.INTER_NEAREST)
        
        return img, torch.from_numpy(ann)


class MOVi_composition(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/root/onethingai-tmp/data/movi_c',
                 split='train',
                 resolution=224):
        self.img_root = osp.join(data_root,split,'video')
        self.ann_root = osp.join(data_root,split,'seg')
        self.img_infos = sorted(os.listdir(self.img_root))

        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resolution = resolution

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        # idx=12362 
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        img = cv2.resize(cv2.imread(filename), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        
        idx2 = random.randint(0,self.__len__()-1)
        # print(idx, idx2)
        vid_idx2 = idx2 // 24
        img_idx2 = idx2 % 24
        filename2 = osp.join(self.img_root, self.img_infos[vid_idx2], str(img_idx2)+'.jpg')
        img2 = cv2.resize(cv2.imread(filename2), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = self.transform(img2)
        return img, img2


class MOVi_dino(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/root/onethingai-tmp/data/movi_c',
                 split='train',
                 resolution=224):
        self.img_root = osp.join(data_root,split,'video')
        self.ann_root = osp.join(data_root,split,'seg')
        self.img_infos = sorted(os.listdir(self.img_root))
        self.resolution = resolution
        
        # self.transform = pth_transforms.Compose([
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        feat = torch.load('/root/onethingai-tmp/data/dino_key_MOVi/{}.pt'.format(idx), map_location='cpu', weights_only=True).float()
        
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        img = cv2.resize(cv2.imread(filename), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        
        annoname = osp.join(self.ann_root, self.img_infos[vid_idx], str(img_idx)+'.png')
        ann = cv2.imread(annoname)[:,:,0]
        ann = cv2.resize(ann, (28,28), interpolation=cv2.INTER_NEAREST)
        return feat, img, ann

class MOVi_video(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/root/onethingai-tmp/data/movi_c',
                 split='train'):
        self.img_root = osp.join(data_root,split,'video')
        self.img_infos = sorted(os.listdir(self.img_root))
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resolution = 224

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def __getitem__(self, idx):
        vid_idx = idx
        imgs = []
        for img_idx in range(24):
            filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
            img = cv2.resize(cv2.imread(filename), (self.resolution,self.resolution), interpolation=cv2.INTER_AREA)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            imgs.append(img.unsqueeze(0))
            
        imgs = torch.cat(imgs, dim=0)
        return imgs

class MOVi_video_test(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/root/onethingai-tmp/data/movi_c',
                 split='train'):
        self.img_root = osp.join(data_root,split,'video')
        self.ann_root = osp.join(data_root,split,'seg')
        self.img_infos = sorted(os.listdir(self.img_root))
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def __getitem__(self, idx):
        vid_idx = idx
        imgs = []
        anns = []
        for img_idx in range(24):
            filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
            img = cv2.resize(cv2.imread(filename), (224,224), interpolation=cv2.INTER_AREA)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            imgs.append(img.unsqueeze(0))
            
            annoname = osp.join(self.ann_root, self.img_infos[vid_idx], str(img_idx)+'.png')
            ann = cv2.imread(annoname)[:,:,0]
            ann = cv2.resize(ann, (224,224), interpolation=cv2.INTER_NEAREST)
            ann = torch.from_numpy(ann)
            anns.append(ann.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0) # [L,3,H,W]
        anns = torch.cat(anns, dim=0) # [L,H,W]
        return imgs, anns

class MOVi_property_predict(torch.utils.data.Dataset):

    def __init__(self,
                 data_root='/root/onethingai-tmp/data/movi_c',
                 split='train'):
        self.img_root = osp.join(data_root,split,'video')
        self.ann_root = osp.join(data_root,split,'seg')
        self.inst_root = osp.join(data_root,split,'instance')
        self.img_infos = sorted(os.listdir(self.img_root))

        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos) * 24

    def __getitem__(self, idx):
        vid_idx = idx // 24
        img_idx = idx % 24
        filename = osp.join(self.img_root, self.img_infos[vid_idx], str(img_idx)+'.jpg')
        annoname = osp.join(self.ann_root, self.img_infos[vid_idx], str(img_idx)+'.png')
        category = np.load(osp.join(self.inst_root, self.img_infos[vid_idx], 'category.npy'))
        num = category.shape[0]
        category = np.concatenate((category, 255+np.zeros((11-num,))), axis=0)
        bboxes_3d = np.load(osp.join(self.inst_root, self.img_infos[vid_idx], 'bboxes_3d.npy'))[:,img_idx]
        positions = np.load(osp.join(self.inst_root, self.img_infos[vid_idx], 'positions.npy'))[:,img_idx]
        pos = np.concatenate((bboxes_3d.reshape(-1,24), positions), axis=1)
        pos = np.concatenate((pos, np.zeros((11-num,27))), axis=0)
        img = cv2.resize(cv2.imread(filename), (224,224), interpolation=cv2.INTER_AREA)
        ann = cv2.resize(cv2.imread(annoname), (224,224), interpolation=cv2.INTER_NEAREST)[:,:,0]
        
        img1 = self.transform(img)
        return img1, torch.from_numpy(ann), category, pos
    
if __name__ == "__main__":
    dataset = MOVi()
    for i in range(10):
        dataset.__getitem__(i*24)

