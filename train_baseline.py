# System libs
import os
import time
import argparse
import datetime as datetime
import random

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist

# Our libs
from dataset.dataset import MOVi_test as Dataset
from model.RHGNet import SlotAttentionAutoEncoder as Model
from utils_train import AverageMeter, get_params_groups, cosine_scheduler, MultiEpochsDataLoader
from loss.loss import RecLPIPSLoss, seg_loss
torch.set_float32_matmul_precision('medium')
seed_value = 42   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

# train one epoch
def train(model, data_loader, optimizers, epoch, gpu, lr_schedule, rec_loss: RecLPIPSLoss, sigma_schedule, lambda_schedule):
    batch_time = AverageMeter()
    ave_loss_1 = AverageMeter()
    ave_loss_2 = AverageMeter()
    ave_loss_3 = AverageMeter()
    ave_loss_4 = AverageMeter()
    ave_loss_5 = AverageMeter()
    
    model.train()
    epoch_iters = len(data_loader)
    data_loader.sampler.set_epoch(epoch)

    # main loop
    tic = time.time()
    for idx,data in enumerate(data_loader):
        
        it = len(data_loader) * epoch + idx
        for i, param_group in enumerate(optimizers.param_groups):
            param_group["lr"] = lr_schedule[it] * param_group["base_lr"]

        sigma = sigma_schedule[it] if it < len(sigma_schedule) else sigma_schedule[-1]
        imgs, masks = data
        
        imgs = imgs.cuda(gpu)
        masks = masks.cuda(gpu)
        
        optimizers.zero_grad()
        # forward pass
        rec, loss_td, pred_mask = model(imgs, sigma=sigma)
        loss_rec = rec_loss(rec, imgs)
        loss_total = loss_rec['total_loss'] + loss_td
        
        # # Backward
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizers.step()

        # # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # # update average loss and acc
        # ave_loss_1.update(loss_rec.item())
        ave_loss_1.update(loss_rec['recon_loss'].item())
        ave_loss_2.update(loss_rec['percept_loss'].item())
        ave_loss_3.update(loss_td.item())
        # ave_loss_4.update(loss_seg.item())

        if dist.get_rank()==0:
            print('[{}][{}/{}], lr: {:.2f}, '
                  'time: {:.2f}, '
                  'Loss: {:.3f}, {:.3f}, {:.3f}'
                  .format(epoch, idx, epoch_iters, lr_schedule[it], batch_time.average(),
                  ave_loss_1.average(), ave_loss_2.average(), ave_loss_3.average()))

def checkpoint(nets, optimizer, args, epoch):
    print('Saving checkpoints...')
    net_encoder = nets.module
    
    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot, exist_ok=True)
        
    torch.save(
        net_encoder.state_dict(),
        '{}/model_epoch_{}.pth'.format(args.saveroot, epoch))
    # torch.save(
    #     optimizer.state_dict(),
    #     '{}/opt_epoch_{}.pth'.format(args.saveroot, epoch))


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

    model = Model(resolution=224)
    
    dataset_train = Dataset(resolution=224)
    sampler_train =torch.utils.data.distributed.DistributedSampler(dataset_train)
    loader_train = MultiEpochsDataLoader(dataset_train, batch_size=args.batchsize, shuffle=False, sampler=sampler_train, 
                                    pin_memory=True, num_workers=args.workers)
    
    # load nets into gpu
    model = model.cuda(load_gpu)
    # model = torch.compile(model.cuda(load_gpu))
    # to_load = torch.load('/root/onethingai-tmp/savemodel/RHGNet/model_epoch_120.pth',map_location=torch.device("cuda:"+str(load_gpu)), weights_only=True)
    # model.load_state_dict(to_load,strict=True)

    if args.resume_epoch!=0:
        to_load = torch.load(os.path.join(args.saveroot,'model_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cuda:"+str(load_gpu)), weights_only=True)
        model.load_state_dict(to_load,strict=True)

    model= torch.nn.parallel.DistributedDataParallel(
                    model,
                device_ids=[load_gpu],
                find_unused_parameters=False)

    # Set up optimizers
    param_groups = get_params_groups(model, lr = args.lr#
                                     , spetial_list=
                                     {'encoder':{'params': [], 'base_lr': 0.1*args.lr, 'weight_decay': 0.01},
                                      'encoder_bias':{'params': [], 'base_lr': 0.1*args.lr, 'weight_decay': 0.0}})
    optimizer = torch.optim.AdamW(param_groups)
    
    lr_schedule = cosine_scheduler(
        1.00,  # linear scaling rule
        0.01,
        args.total_epoch, 
        len(loader_train),
        warmup_epochs=1)
    
    sigma_schedule = cosine_scheduler(
        1.0,  # linear scaling rule
        0.0,
        10, 
        len(loader_train),
        warmup_epochs=0)
    
    lambda_schedule = cosine_scheduler(
        0.0,  # linear scaling rule
        1.0,
        args.total_epoch, 
        len(loader_train),
        warmup_epochs=0)
    
    # Main loop
    lpips_loss = RecLPIPSLoss().cuda(load_gpu)
    lpips_loss.eval()
    
    model = torch.compile(model)
    lpips_loss = torch.compile(lpips_loss)
    for epoch in range(args.resume_epoch, args.total_epoch):
        print('Epoch {}'.format(epoch))
        train(model, loader_train, optimizer, epoch, load_gpu, lr_schedule, lpips_loss, sigma_schedule, lambda_schedule)

        # checkpointing
        if dist.get_rank() == 0 and (epoch+1)%args.save_step==0:
            checkpoint(model, optimizer, args, epoch+1)

    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument("--batchsize",type=int,default=16)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--start_gpu",type=int,default=0)
    parser.add_argument("--gpu_num",type=int,default=3)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--saveroot",type=str,default='/root/onethingai-tmp/savemodel/MOVi/RHGNet')
    parser.add_argument("--total_epoch",type=int,default=60)
    parser.add_argument("--resume_epoch",type=int,default=0)
    parser.add_argument("--save_step",type=int,default=5)
    parser.add_argument("--port",type=int,default=45325)
    args = parser.parse_args()

    print(args)

    mp.spawn(main, nprocs=args.gpu_num, args=(args,))
