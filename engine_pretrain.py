# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched
import wandb
import numpy as np

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, global_rank=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    wandb_images = []
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images = samples["image"].to(device, non_blocking=True)
        labels = samples["label"].to(device, non_blocking=True)
        mask = samples["mask"].to(device, non_blocking=True)
        valid = samples["valid"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, y, mask = model(images, labels, bool_masked_pos=mask, valid=valid)
        loss_value = loss.item()
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            if global_rank == 0:
                wandb.log({'train_loss': loss_value_reduce, 'lr': lr})
                if len(wandb_images) < 20:
                    imagenet_mean = np.array([0.485, 0.456, 0.406])
                    imagenet_std = np.array([0.229, 0.224, 0.225])
                    y = y[[0]]
                    y = model.module.unpatchify(y)
                    y = torch.einsum('nchw->nhwc', y).detach().cpu()
                    mask = mask[[0]]
                    mask = mask.detach().float().cpu()
                    mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
                    mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
                    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
                    x = images[[0]]
                    x = x.detach().float().cpu()
                    x = torch.einsum('nchw->nhwc', x)
                    tgt = labels[[0]]
                    tgt = tgt.detach().float().cpu()
                    tgt = torch.einsum('nchw->nhwc', tgt)
                    im_masked = tgt * (1 - mask)

                    frame = torch.cat((x, im_masked, y, tgt), dim=2)
                    frame = frame[0]
                    frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).int()
                    wandb_images.append(wandb.Image(frame.numpy(), caption="x; im_masked; y; tgt"))

    if global_rank == 0 and len(wandb_images) > 0:
        wandb.log({"Training examples": wandb_images})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}