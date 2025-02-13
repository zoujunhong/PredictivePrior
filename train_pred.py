# System libs
import os
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
import time
# import math
import random
import argparse
# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
# Our libs
from dataset.dataset import MOVi as Dataset
from model.utils.dino import vit_small
from model.utils.PredictivePrior import DINOPredictor as Model
from utils_train import AverageMeter, get_params_groups, cosine_scheduler, MultiEpochsDataLoader
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import datetime as datetime
torch.set_float32_matmul_precision('high')
seed_value = 42   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

# train one epoch

def train(model, dino, data_loader, optimizers, epoch, gpu, lr_schedule):
    batch_time = AverageMeter()
    ave_loss_1 = AverageMeter()
    ave_loss_2 = AverageMeter()

    model.train()
    epoch_iters = len(data_loader)
    data_loader.sampler.set_epoch(epoch)

    # main loop
    tic = time.time()
    for idx,data in enumerate(data_loader):
        
        it = len(data_loader) * epoch + idx
        for i, param_group in enumerate(optimizers.param_groups):
            param_group["lr"] = lr_schedule[it] * param_group["base_lr"]

        img = data
        img = img.cuda(gpu)
        optimizers.zero_grad()
        # forward pass
        with torch.no_grad():   
            dino_feat = dino.forward_key(img)   
             
        loss_1 = model(dino_feat)
        loss_total = loss_1
        
        loss_total.backward()
        optimizers.step()

        # # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # # update average loss and acc
        ave_loss_1.update(loss_1.item())

        if dist.get_rank()==0:
            print('[{}][{}/{}], lr: {:.2f}, '
                  'time: {:.2f}, '
                  'Loss: {:.6f}'
                  .format(epoch, idx, epoch_iters, lr_schedule[it], batch_time.average(),
                  ave_loss_1.average()))

def checkpoint(nets, args, epoch):
    print('Saving checkpoints...')
    net_encoder = nets.module
    
    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot, exist_ok=True)
        
    torch.save(
        net_encoder.state_dict(),
        '{}/model_epoch_{}.pth'.format(args.saveroot, epoch))

def main(gpu,args):
    # Network Builders
    load_gpu = gpu+args.start_gpu
    rank = gpu
    torch.cuda.set_device(load_gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=args.gpu_num,
        rank=rank,
        timeout=datetime.timedelta(seconds=300))
    
    dino = vit_small(patch_size=8)
    to_load = torch.load('/root/to/dino_deitsmall8_pretrain.pth',map_location=torch.device("cpu"),weights_only=True)
    dino.load_state_dict(to_load,strict=True)
    dino = dino.cuda(load_gpu)
    
    dataset_train = Dataset()
    sampler_train =torch.utils.data.distributed.DistributedSampler(dataset_train)
    loader_train = MultiEpochsDataLoader(dataset_train, batch_size=args.batchsize, shuffle=False, sampler=sampler_train, 
                                    pin_memory=True, num_workers=args.workers, drop_last=True)
    
    # load nets into gpu
    model = Model()
    model = model.cuda(load_gpu)

    if args.resume_epoch!=0:
        to_load = torch.load(os.path.join(args.saveroot,'model_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cuda:"+str(load_gpu)))
        model.load_state_dict(to_load,strict=True)

    model= torch.nn.parallel.DistributedDataParallel(
                    model,
                device_ids=[load_gpu],
                find_unused_parameters=False)

    # Set up optimizers
    param_groups = get_params_groups(model, lr=args.lr)
    optimizer = torch.optim.AdamW(param_groups)

    lr_schedule = cosine_scheduler(
        1,  # linear scaling rule
        0.1,
        args.total_epoch, 
        len(loader_train),
        warmup_iters=0,
    )
    
    # Main loop
    for epoch in range(args.resume_epoch, args.total_epoch):
        print('Epoch {}'.format(epoch))
        train(model, dino, loader_train, optimizer, epoch, load_gpu, lr_schedule)

        # checkpointing
        if dist.get_rank() == 0 and (epoch+1)%args.save_step==0:
            checkpoint(model, args, epoch+1)

    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument("--batchsize",type=int,default=32)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--start_gpu",type=int,default=0)
    parser.add_argument("--gpu_num",type=int,default=2)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--saveroot",type=str,default='/path/to/save/checkpoint')
    parser.add_argument("--total_epoch",type=int,default=20)
    parser.add_argument("--resume_epoch",type=int,default=0)
    parser.add_argument("--save_step",type=int,default=10)
    parser.add_argument("--port",type=int,default=45321)
    args = parser.parse_args()

    print(args)

    mp.spawn(main, nprocs=args.gpu_num, args=(args,))
