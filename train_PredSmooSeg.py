# System libs
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["TORCH_LOGS"] = "+dynamo"
import copy
import time
import json
# import math
import random
import argparse
# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
# Our libs
from dataset.dataset import MOVi as Dataset
from model.PredSeg_tf import SlotAttentionAutoEncoder as Model
from utils_train import AverageMeter, get_params_groups, cosine_scheduler, MultiEpochsDataLoader
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import datetime as datetime
torch.set_float32_matmul_precision('medium')
seed_value = 42   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

from loss.loss import RecLPIPSLoss
# train one epoch
# @torch.compile
def train(model, data_loader, optimizers, epoch, gpu, lr_schedule, sigma_schedule, rec_loss: RecLPIPSLoss):
    batch_time = AverageMeter()
    ave_loss_1 = AverageMeter()
    ave_loss_2 = AverageMeter()
    ave_loss_3 = AverageMeter()
    ave_loss_4 = AverageMeter()
    
    model.train()
    epoch_iters = len(data_loader)
    data_loader.sampler.set_epoch(epoch)

    # main loop
    tic = time.time()
    for idx,data in enumerate(data_loader):
        
        it = len(data_loader) * epoch + idx
        for i, param_group in enumerate(optimizers.param_groups):
            param_group["lr"] = lr_schedule[it] * param_group["base_lr"]

        _sigma = sigma_schedule[it] if it < len(sigma_schedule) else sigma_schedule[-1]
        imgs = data
        
        # forward pass
        imgs = imgs.cuda(gpu)
        rec, loss_smooth, loss_consistent = model(imgs, _sigma)
            
        loss_rec = rec_loss(rec, imgs)
        loss_total = loss_rec['total_loss'] + loss_smooth + loss_consistent
        
        # # Backward
        loss_total.backward()
        optimizers.step()

        optimizers.zero_grad()
        # # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # # update average loss and acc
        ave_loss_1.update(loss_rec['recon_loss'].item())
        ave_loss_2.update(loss_rec['percept_loss'].item())
        ave_loss_3.update(loss_smooth.item())
        ave_loss_4.update(loss_consistent.item())

        if dist.get_rank()==0:
            print('[{}][{}/{}], lr: {:.3f}, '
                  'time: {:.2f}, '
                  'Loss: {:.3f}, {:.3f}, {:.3f}, {:.3f}'
                  .format(epoch, idx, epoch_iters, lr_schedule[it], batch_time.average(),  
                  ave_loss_1.average(), ave_loss_2.average(), ave_loss_3.average(), ave_loss_4.average()))

def checkpoint(nets, args, epoch):
    print('Saving checkpoints...')
    net_encoder = nets.module
    
    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot, exist_ok=True)
        
    torch.save(
        net_encoder.state_dict(),
        '{}/model.pth'.format(args.saveroot, epoch))


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

    model = Model(num_slots=11, resolution=224)
    
    dataset_train = Dataset(resolution=224)
    sampler_train =torch.utils.data.distributed.DistributedSampler(dataset_train)
    loader_train = MultiEpochsDataLoader(dataset_train, batch_size=args.batchsize, shuffle=False, sampler=sampler_train, 
                                    pin_memory=True, num_workers=args.workers, drop_last=True)

    if args.resume_epoch!=0:
        to_load = torch.load(os.path.join(args.saveroot,'model_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cpu"),weights_only=True)
        model.load_state_dict(to_load,strict=False)

    
    model = model.cuda(load_gpu)
    model= torch.nn.parallel.DistributedDataParallel(
                    model,
                device_ids=[load_gpu],
                find_unused_parameters=False)

    # Set up optimizers
    param_groups = get_params_groups(model, lr = args.lr)
    optimizer = torch.optim.AdamW(param_groups)
    # optimizer = Opt(param_groups)     
    
    lr_schedule = cosine_scheduler(
        1.00,  # linear scaling rule
        0.01,
        args.total_epoch, 
        len(loader_train),
        warmup_iters=0)
    
    sigma_schedule = cosine_scheduler(
        1.0,  # linear scaling rule
        0.,
        10, 
        len(loader_train))
    
    # Main loop
    lpips_loss = RecLPIPSLoss().cuda(load_gpu)
    lpips_loss.eval()
    for epoch in range(args.resume_epoch, args.total_epoch):
        print('Epoch {}'.format(epoch))
        train(model, loader_train, optimizer, epoch, load_gpu, lr_schedule, sigma_schedule, lpips_loss)

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
    parser.add_argument("--total_epoch",type=int,default=120)
    parser.add_argument("--resume_epoch",type=int,default=0)
    parser.add_argument("--save_step",type=int,default=15)
    parser.add_argument("--port",type=int,default=45321)
    args = parser.parse_args()

    print(args)

    mp.spawn(main, nprocs=args.gpu_num, args=(args,))
