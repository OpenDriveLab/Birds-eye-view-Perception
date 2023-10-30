# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=1000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--mem-only',
        action='store_true',
        help='Conduct the memory analysis only')
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='Omit the pre-computation acceleration')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    if not args.no_acceleration:
        cfg.model.img_view_transformer.accelerate=True
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 100
    pure_inf_time = 0
    D = model.module.img_view_transformer.D
    out_channels = model.module.img_view_transformer.out_channels
    depth_net = model.module.img_view_transformer.depth_net
    view_transformer = model.module.img_view_transformer
    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            img_feat = \
                model.module.image_encoder(data['img_inputs'][0][0].cuda())
            B, N, C, H, W = img_feat.shape
            x = depth_net(img_feat.reshape(B * N, C, H, W))
            depth_digit = x[:, :D, ...]
            tran_feat = x[:, D:D + out_channels, ...]
            depth = depth_digit.softmax(dim=1)
        input = [img_feat] + [d.cuda() for d in data['img_inputs'][0][1:]]

        if i == 0:
            precomputed_memory_allocated = 0.0
            if view_transformer.accelerate:
                start_mem_allocated = torch.cuda.memory_allocated()
                view_transformer.pre_compute(input)
                end_mem_allocated = torch.cuda.memory_allocated()
                precomputed_memory_allocated = \
                    end_mem_allocated - start_mem_allocated
                ref_max_mem_allocated = torch.cuda.max_memory_allocated()
                # occupy the memory
                size = (ref_max_mem_allocated - end_mem_allocated) // 4
                occupy_tensor = torch.zeros(
                    size=(size, ), device='cuda', dtype=torch.float32)
            print('Memory analysis: \n'
                  'precomputed_memory_allocated : %d B / %.01f MB \n' %
                  (precomputed_memory_allocated,
                   precomputed_memory_allocated / 1024 / 1024))
            start_mem_allocated = torch.cuda.memory_allocated()
            bev_feat = view_transformer.view_transform_core(
                input, depth, tran_feat)[0]
            end_max_mem_allocated = torch.cuda.max_memory_allocated()
            peak_memory_allocated = \
                end_max_mem_allocated - start_mem_allocated
            total_memory_requirement = \
                precomputed_memory_allocated + peak_memory_allocated
            print('Memory analysis: \n'
                  'Memory requirement : %d B / %.01f MB \n' %
                  (total_memory_requirement,
                   total_memory_requirement / 1024 / 1024))
            if args.mem_only:
                return

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            view_transformer.view_transform(input, depth, tran_feat)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            return fps


if __name__ == '__main__':
    repeat_times = 1
    fps_list = []
    for _ in range(repeat_times):
        fps = main()
        time.sleep(5)
        fps_list.append(fps)
    fps_list = np.array(fps_list, dtype=np.float32)
    print(f'Mean Overall fps: {fps_list.mean():.4f} +'
          f' {np.sqrt(fps_list.var()):.4f} img / s')
