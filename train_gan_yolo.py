# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val_gan_yolo  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets_gan_yolo import create_dataloader, create_dataloader_rgb_ir # create_dual_dataloader,
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.denormalizer import denormalizer
from models import networks
from models import reg
from models import transformer
from models.AutomaticWeightedLoss import AutomaticWeightedLoss
# import models.reg as reg
# import models.transformer as transformer
from gradnorm_pytorch.gradnorm_pytorch import GradNormLossWeighter

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def smoothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d 
    return d

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def weight_l1_loss(y_pred, y_true, x_true):
    return torch.mean(torch.abs(y_pred - y_true) * (1 + torch.clamp(y_true - x_true / 2, min=0.0, max=1.0)))

def adapt_l1_l2_loss(y_pred, y_true, targets):
    bc, c, ht, wt = y_pred.shape
    wi = torch.zeros_like(y_true)
    for t in list(targets):
        i = int(t[0])
        x, y, w, h = t[2:]
        x = int(x * wt)
        y = int(y * ht)
        w = int(w * wt)
        h = int(h * ht)
        wi[i, :, y:y+h, x:x+w] = 1
    a = torch.sum(torch.abs(y_pred - y_true) * (1 - wi)) / torch.sum(1 - wi)
    b = torch.sum((torch.abs(y_pred - y_true) ** 2) * wi) / torch.sum(wi)
    return a + b

def argu_defect_loss(y_pred, y_true, targets):
    bc, c, ht, wt = y_pred.shape
    wi = torch.zeros_like(y_true)
    for t in list(targets):
        i = int(t[0])
        x, y, w, h = t[2:]
        x = int(x * wt)
        y = int(y * ht)
        w = int(w * wt)
        h = int(h * ht)
        wi[i, :, y:y+h, x:x+w] = 1
    a = torch.sum(torch.abs(y_pred - y_true) * (1 - wi)) / torch.sum(1 - wi)
    b = torch.sum(torch.abs(y_pred - y_true) * wi * 2) / torch.sum(wi)
    return a + b

def argu_defect_loss_l2(y_pred, y_true, targets):
    bc, c, ht, wt = y_pred.shape
    wi = torch.zeros_like(y_true)
    for t in list(targets):
        i = int(t[0])
        x, y, w, h = t[2:]
        x = int(x * wt)
        y = int(y * ht)
        w = int(w * wt)
        h = int(h * ht)
        wi[i, :, y:y+h, x:x+w] = 1
    # a = torch.sum(torch.abs(y_pred - y_true) * (1 - wi)) / torch.sum(1 - wi)
    # b = torch.sum(torch.abs(y_pred - y_true) * wi * 2) / torch.sum(wi)
    a = torch.sum(((y_pred - y_true) ** 2) * (1 - wi)) / torch.sum(1 - wi)
    b = torch.sum(((y_pred - y_true) ** 2) * wi * 2) / torch.sum(wi)
    return a + b

class FocalLossRegression(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLossRegression, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # 计算平方损失
        loss = F.l1_loss(inputs, targets, reduction='none')

        # 计算调整后的权重
        weights = self.alpha * (1 - torch.exp(-loss)) ** self.gamma

        # 应用权重
        weighted_loss = weights * loss

        # 求平均损失
        return torch.sum(weighted_loss)

class RLW(nn.Module):
    r"""Random Loss Weighting (RLW).
    
    This method is proposed in `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_ \
    and implemented by us.

    """
    def __init__(self, task_num, device):
        super(RLW, self).__init__()
        self.task_num = task_num
        self.device = device
        
    def forward(self, losses):
        losses = torch.stack(losses).to(self.device)
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        return loss

class DWA(nn.Module):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    Args:
        T (float, default=2.0): The softmax temperature.

    """
    def __init__(self, task_num, epochs, device):
        super(DWA, self).__init__()
        self.task_num = task_num
        self.T = 2.0
        self.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.device = device
        
    def update_train_loss_buffer(self, losses, epoch):
        losses = torch.stack(losses)
        self.train_loss_buffer[:,epoch] = losses.detach().cpu().numpy()
        
    def forward(self, losses, epoch):
        losses = torch.stack(losses).to(self.device)
        if epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:,epoch-1]/self.train_loss_buffer[:,epoch-2]).to(self.device)
            batch_weight = self.task_num*F.softmax(w_i/self.T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        return loss
    
class DWA2(nn.Module):
    r"""Dynamic Weight Average (DWA).
    
    以Batch为时间单位的更新

    Args:
        T (float, default=2.0): The softmax temperature.

    """
    def __init__(self, task_num, batchs, device):
        super(DWA2, self).__init__()
        self.task_num = task_num
        self.T = 2.0
        self.train_loss_buffer = np.zeros([self.task_num, batchs])
        self.device = device
        
    def update_train_loss_buffer(self, losses, batch):
        losses = torch.stack(losses)
        b = batch % self.train_loss_buffer.shape[1]
        self.train_loss_buffer[:,b] = losses.detach().cpu().numpy()
        
    def forward(self, losses, batch):
        losses = torch.stack(losses).to(self.device)
        if batch > 1:
            b1 = (batch - 1) % self.train_loss_buffer.shape[1]
            b2 = (batch - 2) % self.train_loss_buffer.shape[1]
            w_i = torch.Tensor(self.train_loss_buffer[:,b1]/self.train_loss_buffer[:,b2]).to(self.device)
            batch_weight = self.task_num*F.softmax(w_i/self.T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        return loss

def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best, g_best, d_best, r_best = w / 'last.pt', w / 'best.pt',w / 'netG_best.pt',w / 'netD_best.pt',w / 'netR_A_best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    # train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = False  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    train_path_rgb = data_dict['train_rgb']
    test_path_rgb = data_dict['val_rgb']
    train_path_ir = data_dict['train_ir']
    test_path_ir = data_dict['val_ir']
    # Freeze
    # freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # for k, v in model.named_parameters():
    #     v.requires_grad = True  # train all layers
    #     if any(x in k for x in freeze):
    #         LOGGER.info(f'freezing {k}')
    #         v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.optimizer == 'Adam':
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect, rank=LOCAL_RANK, workers=workers,
                                              image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader_rgb_ir(test_path_rgb, test_path_ir, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache,
                                       rect=True, rank=-1, workers=workers * 2, pad=0.0,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    
    #gan
    # netD注释掉
    device_ids = [int(id_) for id_ in opt.device.split(',')]
    netD = networks.define_D(6, 64, 'basic', 3, 'batch', 'normal', 0.02, device_ids)
    netG =  networks.define_G(3, 3, 64, 'unet_256', 'batch', True, 'normal', 0.02, device_ids)
    epoch_num = '52' # 52 183
    netG_load_path = os.path.join('/root/workspace/project/my-pytorch-CycleGAN-and-pix2pix/get_sh_and_pth/pth/class_5_1/epoch_{}/'.format(epoch_num),'{}_net_G.pth'.format(epoch_num))
    state_dict_netG = torch.load(netG_load_path)
    netD_load_path = os.path.join('/root/workspace/project/my-pytorch-CycleGAN-and-pix2pix/get_sh_and_pth/pth/class_5_1/epoch_{}/'.format(epoch_num),'{}_net_D.pth'.format(epoch_num))
    state_dict_netD = torch.load(netD_load_path)
    netRA_load_path = os.path.join('/root/workspace/project/my-pytorch-CycleGAN-and-pix2pix/get_sh_and_pth/pth/class_5_1/epoch_{}/'.format(epoch_num),'{}_net_R_A.pth'.format(epoch_num))
    state_dict_netRA = torch.load(netRA_load_path)

    
    
    if hasattr(state_dict_netG, '_metadata'):
        del state_dict_netG._metadata
    if hasattr(state_dict_netD, '_metadata'):
        del state_dict_netD._metadata
    if hasattr(state_dict_netRA, '_metadata'):
        del state_dict_netRA._metadata
    # netD注释掉
    if isinstance(netD, torch.nn.DataParallel):
        netD.module.load_state_dict(state_dict_netD)
    else:
        netD.load_state_dict(state_dict_netD)
        netD = nn.DataParallel(netD, device_ids=device_ids) 
    
    if isinstance(netG, torch.nn.DataParallel):
        netG.module.load_state_dict(state_dict_netG)
    else:
        netG.load_state_dict(state_dict_netG)
        netG = nn.DataParallel(netG, device_ids=device_ids) 
        
    # for k, v in netG.named_parameters():
    #     v.requires_grad = False  # train all layers
    
    
    netR_A = reg.Reg()
    netspatial_transform = transformer.Transformer_2D()
    
    # REG注释掉
    netR_A.load_state_dict(state_dict_netRA)
    netR_A = nn.DataParallel(netR_A, device_ids=device_ids)
    
    criterionGAN = networks.GANLoss('vanilla').to(device)
    
    criterionL1 = torch.nn.L1Loss()
    criterionL2 = torch.nn.MSELoss()
    criterionPix = FocalLossRegression()
    # netD注释掉
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizer_D = torch.optim.SGD(netD.parameters(), lr=0.00005, momentum=hyp['momentum'])
    
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizer_G = torch.optim.SGD(netG.parameters(), lr=0.0005, momentum=hyp['momentum'])
    # optimizer.add_param_group({'params': g2})
    # Reg注释掉
    optimizer_R_A = torch.optim.Adam(netR_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizer_R_A = torch.optim.SGD(netR_A.parameters(), lr=0.0005, momentum=hyp['momentum'])
    
    # netD注释掉
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lf)
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lf)
    # Reg注释掉
    scheduler_R_A = lr_scheduler.LambdaLR(optimizer_R_A, lr_lambda=lf)
    
    # AWL 多任务权重
    # awl = AutomaticWeightedLoss(num=5, weight=[0.5, 0.001, 0.05, 0.1, 0.8])
    # # optimizer_awl = SGD(awl.parameters(), lr=0.0002, momentum=hyp['momentum'])
    # optimizer_awl = torch.optim.Adam(awl.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # scheduler_awl = lr_scheduler.LambdaLR(optimizer_awl, lr_lambda=lf)
    
    # random 多任务权重
    # rlw = RLW(task_num=5, device=device)
    
    ## DWA
    # dwa = DWA(task_num=5, epochs=epochs, device=device)
    
    ## DWA2
    # dwa2 = DWA2(task_num=5, batchs=nb, device=device)
    
    # GradNorm 多任务权重
    # backbone_parameter = netG.module.model.model[-2].weight
    # print(backbone_parameter)
    # loss_weighter = GradNormLossWeighter(
    #     num_losses = 5,
    #     learning_rate = 1e-4,
    #     restoring_force_alpha = 0.,                  # 0. is perfectly balanced losses, while anything greater than 1 would account for the relative training rates of each loss. in the paper, they go as high as 3.
    #     grad_norm_parameters = backbone_parameter
    # )
    
    
    #gan
    
    
    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    mid_stop_epoch = epochs - 10 if epochs > 10 else epochs // 2
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(10, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%13s' * 14) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'loss_D_fake', 'loss_D_real', 'loss_f', 'loss_l1', 'loss_GAN', 'loss_SR', 'loss_SM', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()
        # optimizer_awl.zero_grad()
        for i, (imgs_rgb, imgs_ir, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            # imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            imgs_rgb = imgs_rgb.to(device, non_blocking=True)
            imgs_ir = imgs_ir.to(device, non_blocking=True)

            # Warmup
            if False and ni <= nw: # No need to warmup
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs_rgb.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs_rgb.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs_rgb = nn.functional.interpolate(imgs_rgb, size=ns, mode='bilinear', align_corners=False)
            
            gan_img = netG(imgs_rgb)
            # REG注释掉
            Trans = netR_A(gan_img, imgs_ir) 
            SysRegist_A2B = netspatial_transform(gan_img, Trans)
            # if i == 0 or i % 2 == 0:
            # netD注释掉
            set_requires_grad(netD, True)  # enable backprop for D
            optimizer_D.zero_grad()
            fake_AB = torch.cat((imgs_rgb, gan_img), 1)  # we use conditional GANs; we need to feed both input and 
            pred_fake = netD(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # Real
            # netD注释掉
            real_AB = torch.cat((imgs_rgb, imgs_ir), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, True)

            # combine loss and calculate gradients
            # netD注释掉
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            # netD注释掉
            loss_D.backward(retain_graph=True)
            optimizer_D.step()
            # update G
            # netD注释掉
            set_requires_grad(netD, False)  # enable backprop for D
            optimizer_G.zero_grad()
            # REG注释掉
            optimizer_R_A.zero_grad()
            # netD注释掉
            fake_AB = torch.cat((imgs_rgb, gan_img), 1)
            pred_fake = netD(fake_AB)
            loss_gan = criterionGAN(pred_fake, True)
            loss_L1 = criterionL1(gan_img, imgs_ir) # * 50
            # loss_L1 = criterionL2(gan_img, imgs_ir) # * 50
            # loss_L1 = weight_l1_loss(gan_img, imgs_ir, imgs_rgb)
            # loss_L1 = adapt_l1_l2_loss(gan_img, imgs_ir, targets)
            # loss_L1 = argu_defect_loss_l2(gan_img, imgs_ir, targets)
            # loss_L1 = argu_defect_loss(gan_img, imgs_ir, targets)
            # loss_L1 = criterionPix(gan_img, imgs_ir) / 100
            
            # REG注释掉
            loss_SR = criterionL1(SysRegist_A2B, imgs_ir) # * 20 ###SR
            loss_SM = smoothing_loss(Trans) # * 10 ###SM
            # loss_SR = 0
            # loss_SM = 0

            # Forward
        
            gan_img = denormalizer(gan_img)
            # gan_img = gan_img.add(1)
            # gan_img = gan_img.div(2)
            
            #zjh add
            # imgs_ir = denormalizer(imgs_ir)
            # imgs_ir_resize = upconv(imgs_ir)
            # imgs_ir_resize = imgs_ir
            #zjh add

            if imgsz == 1024:
                gan_resize = upconv(gan_img)
            else:
                gan_resize = gan_img
            with amp.autocast(enabled=cuda):
                pred = model(gan_resize)  # forward
                loss_f, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                # loss_items = torch.zeros(3, device=device)
                
                #zjh add
                # pred_petct = model(imgs_ir_resize)  # forward
                # loss_f_petct, loss_items_petct = compute_loss(pred_petct, targets.to(device))  # loss scaled by batch_size
                #zjh add

                # netD注释掉
                # loss_gan = 0
                # loss_f = 0

                #zjh add
                # loss_all = loss_L1 * 0.2 + loss_gan * 0.5 + loss_SR * 0.1 + loss_SM + loss_f + loss_f_petct
                # loss_items2 = torch.tensor((loss_L1, loss_gan, loss_SR, loss_SM)).to(device)
                # loss_items = torch.cat((loss_items, loss_items2))
                # loss_items = torch.cat((loss_items, loss_items_petct))
                #zjh add
                ## 固定权重
                loss_all = loss_L1 * 0.5 + loss_gan * 0.001 + loss_SR * 0.05 + loss_SM * 0.1 + loss_f * 0.8
                ## 早停
                # if epoch < mid_stop_epoch:
                #     loss_all = awl(loss_L1, loss_gan, loss_SR, loss_SM, loss_f[0])
                # else:
                #     loss_all = loss_L1 * 0.5 + loss_gan * 0.001 + loss_SR * 0.05 + loss_SM * 0.1 + loss_f * 0.8
                ## 随机
                # loss_all = rlw([loss_L1, loss_gan, loss_SR, loss_SM, loss_f[0]])
                ## dwa
                # loss_all = dwa([loss_L1 * 0.5, loss_gan * 0.001, loss_SR * 0.05, loss_SM * 0.1, loss_f[0] * 0.8], epoch)
                ## dwa2
                # loss_all = dwa2([loss_L1 * 0.5, loss_gan * 0.001, loss_SR * 0.05, loss_SM * 0.1, loss_f[0] * 0.8], ni)

                loss_items2 = torch.tensor((loss_D_fake, loss_D_real, loss_f[0], loss_L1, loss_gan, loss_SR, loss_SM)).to(device)
                loss_items = torch.cat((loss_items, loss_items2))
                if RANK != -1:
                    loss_all *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss_all *= 4.

            # Backward
            # scaler.scale(loss_all).backward()
            loss_all.backward()
            # loss_weighter.backward([loss_L1, loss_gan, loss_SR, loss_SM, loss_f[0]])
            optimizer_G.step() 
            optimizer_R_A.step()
            # optimizer_awl.step()
            # optimizer_awl.zero_grad()

            # Optimize
            if ni - last_opt_step >= accumulate:
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%13s' * 2 + '%13.4g' * 12) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], gan_resize.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, gan_resize, targets, paths, plots, opt.sync_bn)
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------
            ## dwa2历史loss更新
            # dwa2.update_train_loss_buffer([mloss[6] * 0.5, mloss[7] * 0.001, mloss[8] * 0.05, mloss[9] * 0.1, mloss[5] * 0.8], ni)
        ## dwa历史loss更新
        # dwa.update_train_loss_buffer([mloss[6] * 0.5, mloss[7] * 0.001, mloss[8] * 0.05, mloss[9] * 0.1, mloss[5] * 0.8], epoch)
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        # REG注释掉
        scheduler_R_A.step()
        scheduler_G.step()
        scheduler_D.step()
        # scheduler_awl.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val_gan_yolo.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           gan_model=netG,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    torch.save(netG.module.cpu().state_dict(), g_best)
                    torch.save(netD.cpu().state_dict(), d_best)
                    torch.save(netR_A.cpu().state_dict(), r_best)
                    netG.cuda(0)
                    netD.cuda(0)
                    netR_A.cuda(0)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}_yolo.pt')
                    torch.save(netG.module.cpu().state_dict(), w / f'epoch{epoch}_netG.pt')
                    torch.save(netD.cpu().state_dict(), w / f'epoch{epoch}_netD.pt')
                    torch.save(netR_A.cpu().state_dict(), w / f'epoch{epoch}_netR_A.pt')
                    netG.cuda(0)
                    netD.cuda(0)
                    netR_A.cuda(0)
                    
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    state_dict_netG = torch.load(g_best)
                    if hasattr(state_dict_netG, '_metadata'):
                        del state_dict_netG._metadata
                    if isinstance(netG, torch.nn.DataParallel):
                        netG.module.load_state_dict(state_dict_netG)
                    else:
                        device_ids = [int(id_) for id_ in opt.device.split(',')]
                        netG.load_state_dict(state_dict_netG)
                        netG = nn.DataParallel(netG, device_ids=device_ids) 
                    results, _, _ = val_gan_yolo.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            gan_model=netG,
                                            iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=is_coco,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-gan.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])

    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
