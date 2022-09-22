# Copyright (c) OpenPerceptionX. All rights reserved.
import torch
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.parallel.data_container import DataContainer


@RUNNERS.register_module()
class EpochBasedRunner_video(EpochBasedRunner):

    def __init__(self,
                 model,
                 eval_model=None,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                 max_iters=None,
                 max_epochs=None):
        super().__init__(model, batch_processor, optimizer, work_dir, logger, meta, max_iters, max_epochs)
        keys.append('img_metas')
        self.keys = keys
        self.eval_model = eval_model
        self.eval_model.eval()

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            num_samples = data_batch['img'].data[0].size(1)
            data_list = []
            prev_bev = None
            for i in range(num_samples):
                data = {}
                for key in self.keys:
                    if key not in ['img_metas', 'img', 'points']:
                        data[key] = data_batch[key]
                    else:
                        if key == 'img':
                            data['img'] = DataContainer(data=[data_batch['img'].data[0][0:1, i]],
                                                        cpu_only=data_batch['img'].cpu_only,
                                                        stack=True)
                        elif key == 'img_metas':
                            data['img_metas'] = DataContainer(data=[[data_batch['img_metas'].data[0][0][i]]],
                                                              cpu_only=data_batch['img_metas'].cpu_only)
                        elif key == 'points':
                            data['points'] = DataContainer(data=[[data_batch['points'].data[0][0][i]]],
                                                           cpu_only=data_batch['points'].cpu_only)
                        else:
                            pass
                            # assert False
                            # DataContainer(data=[tmp_data], cpu_only=data_batch[key].cpu_only)
                if i != num_samples - 1:
                    data['return_bev'] = DataContainer(data=[True], cpu_only=True)
                data_list.append(data)

            with torch.no_grad():
                for i in range(num_samples - 1):
                    #if i>0 and pre_data_list[i]['img_metas'].data[0][0]['sample_idx'] == pre_data_list[i-1]['img_metas'].data[0][0]['next_idx']:
                    if data_list[i]['img_metas'].data[0][0]['prev_bev']:
                        data_list[i]['prev_bev'] = DataContainer(data=[prev_bev], cpu_only=False)
                    # print(i)
                    # print(data_list[i])
                    prev_bev = self.eval_model.val_step(data_list[i], self.optimizer, **kwargs)
            if data_list[-1]['img_metas'].data[0][0]['prev_bev']:
                data_list[-1]['prev_bev'] = DataContainer(data=[prev_bev], cpu_only=False)
            outputs = self.model.train_step(data_list[-1], self.optimizer, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs