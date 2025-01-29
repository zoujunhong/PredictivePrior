# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')
from dataset.dataset import MOVi as Dataset
from model.utils.dino import vit_small as Model
import numpy as np
import datetime as datetime
from random import randint
palette = [np.array([randint(1,255),randint(1,255),randint(1,255)]) for i in range(196*20)]
num_slots=11
# train one epoch

@torch.compile
def test(dino: Model, data_loader):
    dino.eval()
    norm_layer = nn.LayerNorm(384, elementwise_affine=False)
    for i,data in enumerate(data_loader):
        if i % 100 == 0:
            print(i)
        with torch.no_grad():
                imgs = data.cuda()

                dino_feat = dino.forward_key(imgs)
                dino_feat = dino_feat.squeeze(0)
                
                # feat = dino_feat.half().cpu()
                # feat_load = torch.load('/zoujunhong/data/dino_key_MOVi/{}.pt'.format(i))\
                # dino_feat = dino_feat.half().float()
                
                # x_lowrank,_,_=torch.pca_lowrank(dino_feat, 3, True, 20)
                # x_lowrank = x_lowrank.reshape(1,28,28,3).permute(0,3,1,2).contiguous()
                # x_lowrank = F.interpolate(x_lowrank, scale_factor=224//28, mode='bilinear', align_corners=False)
                # x_lowrank = x_lowrank - torch.min(x_lowrank)
                # x_lowrank = x_lowrank / torch.max(x_lowrank)
                # x_lowrank = (x_lowrank.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                # cv2.imwrite('demo/feat.png', x_lowrank)
                
                # # attns = segmentation_module.get_last_selfattention(imgs)[:,:,1:,0]
                # # for i in range(6):
                # #     attn = attns[0,i,...].reshape(1,1,28,28)
                # #     attn = F.interpolate(attn, scale_factor=224//28, mode='bilinear', align_corners=False)
                # #     attn = attn / (torch.max(attn)+1e-8)
                # #     print(torch.min(attn))
                # #     attn = attn.squeeze().numpy() * 255
                # #     cv2.imwrite('demo/attn{}.png'.format(i),attn.astype(np.uint8))
                    
                # clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.8).fit(dino_feat.cpu().numpy())
                # labels = []
                # for x in range(len(clustering.labels_)):
                #     labels.append(clustering.labels_[x])
                # labels = np.array(labels).reshape(28,28)
                # labels = cv2.resize(labels, (224,224), interpolation=cv2.INTER_NEAREST)
                # cv2.imwrite('demo/labels.png', labels * 20)
                
                # object_mask = labels * 20
                # object_mask = np.expand_dims(object_mask, axis=-1).repeat(3, axis=-1)
                # object_mask_palette = object_mask.copy()
                # for i in range(object_mask.shape[0]):
                #     for j in range(object_mask.shape[1]):
                #         object_mask_palette[i,j,:] = palette[object_mask[i,j,0]]
                
                # masks = masks.squeeze().numpy().astype(np.uint8) * 20
                # masks = np.expand_dims(masks, axis=-1).repeat(3, axis=-1)
                # masks_palette = masks.copy()
                # for i in range(masks.shape[0]):
                #     for j in range(masks.shape[1]):
                #          masks_palette[i,j,:] = palette[masks[i,j,0]]
                
                # rec = ((rec_obj.clamp(-1,1).squeeze().permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
                # to_save = np.concatenate((origin_img, object_mask_palette*0.5 + origin_img * 0.5, masks_palette*0.5 + origin_img * 0.5), axis=1)
                # cv2.imwrite('demo/mask.png', to_save)
                
                feat = dino_feat.half().cpu()
                torch.save(feat, '/root/autodl-tmp/data/Vision/movi_c/dino_key/{}.pt'.format(i))

                # a = input("input:")
                # if a == 0:
                #     break


def main():
    # Network Builders
    
    print('pretrained model loaded !')
    dataset_train = Dataset()
    print(dataset_train.__len__())
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, pin_memory=False,
                                    num_workers=0)
    
    # load nets into gpu
    model = Model(patch_size=8)
    to_load = torch.load('/root/autodl-tmp/model/pretrain/dino/dino_deitsmall8_pretrain.pth',map_location=torch.device("cpu"),weights_only=True)
    model.load_state_dict(to_load,strict=True)
    model = model.cuda()

    test(model, loader_train)



if __name__ == '__main__':
    main()
