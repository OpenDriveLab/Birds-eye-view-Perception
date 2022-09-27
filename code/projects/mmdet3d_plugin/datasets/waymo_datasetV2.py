# ==============================================================================
# Copyright (c) 2022 OpenPerceptionX. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import time
import os
import tempfile
import copy
import cv2

import numpy as np
import torch
import mmcv
from mmcv.utils import print_log

from mmdet.datasets import DATASETS
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from projects.mmdet3d_plugin.models.utils.draw_bbox import show_multi_modality_result
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet3d.core import show_result
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               LiDARInstance3DBoxes, points_cam2img)
from mmcv.parallel import DataContainer as DC
import random
import os.path as osp
try:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2, label_pb2
    from waymo_open_dataset.protos import breakdown_pb2, metrics_pb2
    from waymo_open_dataset.metrics.python import config_util_py as config_util

except ImportError:
    print('WARNING no tensorflow')

from .waymo_dataset import WaymoDataset_video


@DATASETS.register_module()
class WaymoDataset_videoV2(WaymoDataset_video):

    def __init__(self, gt_bin_file=None, use_pkl_annos=True, gt_bboxes_2d_file=None, calib_file=None, refine_with_2dbbox=False, verbose=False, img_format='.png', *args,
                 **kwargs):

        super(WaymoDataset_videoV2, self).__init__(*args, **kwargs)
        self.img_format = img_format
        self.verbose = verbose
        self.gt_bin_file = gt_bin_file
        self.calib_file = calib_file
        self.use_pkl_annos = use_pkl_annos
        self.refine_with_2dbbox = refine_with_2dbbox
        self.gt_bboxes_2d_file = gt_bboxes_2d_file

        if self.gt_bboxes_2d_file is not None:
            self.gt_bboxes_2d_info = mmcv.load(self.gt_bboxes_2d_file)
        else:
            self.gt_bboxes_2d_info = None
        if self.use_pkl_annos and self.calib_file is not None:
            self.calib_info = mmcv.load(self.calib_file)
        else:
            self.calib_info = None

    def load_annotations(self, ann_file):
        if isinstance(ann_file, (list, tuple)):
            ann_info = []
            for file in ann_file:
                ann_info += mmcv.load(file)
        elif isinstance(ann_file, str):
            ann_info = mmcv.load(ann_file)
        else:
            raise ValueError
        return ann_info

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Standard input_dict consists of the
                data information.
                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']

        img_filename_0 = os.path.join(self.data_root, info['image']['image_path'])
        image_filenames = []
        for i in range(5):
            file_path = img_filename_0.replace('image_0', f'image_{i}')
            file_path = osp.splitext(file_path)[0] + self.img_format
            image_filenames.append(file_path)

        lidar2imgs = []
        lidar2cam_rts = []
        cam_intrinsics = []

        calib_key = info['image']['image_path'].replace('image_0', 'velodyne').replace('png', 'bin').replace('jpg', 'bin')
        if self.use_pkl_annos and self.calib_info is not None and calib_key in self.calib_info:
            calib_info = self.calib_info[calib_key]
            for i in range(5):
                rect = calib_info['R0_rect'].astype(np.float32)
                Trv2c = calib_info[f'Tr_velo_to_cam_{i}'].astype(np.float32)
                Pi = calib_info[f'P{i}'].astype(np.float32)
                lidar2img = Pi @ rect @ Trv2c
                lidar2imgs.append(lidar2img)
                lidar2cam_rts.append(rect @ Trv2c)
                cam_intrinsics.append(Pi)
        else:
            calib_file = img_filename_0.replace('image_0', 'calib').replace('png', 'txt').replace('jpg', 'txt')
            calibs = mmcv.load(calib_file)
            calibs = [each.strip() for each in calibs]
            for i in range(5):
                rect = info['calib']['R0_rect'].astype(np.float32)
                Trv2c = np.zeros([4, 4])
                Trv2c[-1, -1] = 1.
                Trv2c[:3, :4] = np.array(calibs[i + 6].split(' ')[1:]).reshape([3, 4])
                Pi = info['calib'][f'P{i}'].astype(np.float32)
                lidar2img = Pi @ rect @ Trv2c
                lidar2imgs.append(lidar2img)
                lidar2cam_rts.append(rect @ Trv2c)  # The extrinsics that tranforms data from lidar to camera.
                cam_intrinsics.append(Pi)

        # pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=None,
            img_prefix=None,
            img_filename=image_filenames,
            lidar2img=lidar2imgs,
            cam_intrinsic=cam_intrinsics,
            lidar2cam=lidar2cam_rts,
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            if annos is None:
                return None
            input_dict['ann_info'] = annos

        ego_pose_r = np.array(info['pose'])[:3, :3]
        ego_pose_t = np.array(info['pose'])[:3, 3]
        can_bus = np.zeros(18)

        can_bus[:3] = ego_pose_t
        can_bus[3:7] = Quaternion(matrix=ego_pose_r)
        patch_angle = quaternion_yaw(Quaternion(matrix=ego_pose_r)) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict.update(
            dict(
                scene_token=sample_idx // 1000,
                prev_idx=index - self.load_interval,
                next_idx=index + self.load_interval,
                can_bus=can_bus
            )
        )

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        Args:
            index (int): Index of the annotation data to get.
        Returns:
            dict: annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api

        info = self.data_infos[index]
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        # annos = info['annos'] do not use this for mutli-view
        # we need other objects to avoid collision when sample

        custom_label_map = {
            'name': [],  # 1
            'bbox': [],  # 4
            'dimensions': [],  # 3
            'location': [],  # 3
            'rotation_y': [],  # 1
            'diffculty': [],
        }
        if self.use_pkl_annos:
            annos = info['annos']
            keep = (annos['bbox'] != 0).any(-1)
            custom_label_map['name'] = annos['name'][keep]
            custom_label_map['location'] = annos['location'][keep]
            custom_label_map['dimensions'] = annos['dimensions'][keep]
            custom_label_map['rotation_y'] = annos['rotation_y'][keep]
            custom_label_map['bbox'] = annos['bbox'][keep]

            num_points_in_gt = annos.get('num_points_in_gt', np.full_like(annos['rotation_y'], -1, dtype=np.int32))[
                keep]
            for num_point in num_points_in_gt:
                if num_point == -1:
                    custom_label_map['diffculty'].append(0)
                else:
                    if num_point <= 5:
                        custom_label_map['diffculty'].append(2)
                    else:
                        custom_label_map['diffculty'].append(1)

            custom_label_map['diffculty'] = np.array(custom_label_map['diffculty'], dtype=np.int32)

            if len(custom_label_map['name']) == 0:
                return None

        else:
            lable_all_path = info['image']['image_path'].replace('image_0', 'label_all').replace('.png', '.txt').replace('.jpg', '.txt')
            lable_all_path = os.path.join(self.data_root, lable_all_path)
            label_all = []
            for line in mmcv.load(lable_all_path):
                box_info = line.strip().split(' ')
                if box_info[4:8] == ['0', '0', '0', '0']:  # remove back bboxes
                    continue
                label_all.append(box_info)
            if len(label_all) == 0:
                return None
            for info in label_all:
                custom_label_map['name'].append(info[0])
                custom_label_map['location'].append(info[11:11 + 3])
                custom_label_map['dimensions'].append([info[10], info[8], info[9]])
                custom_label_map['rotation_y'].append(info[14])
                custom_label_map['bbox'].append(info[4:8])
                custom_label_map['diffculty'].append(0)

        custom_label_map['name'] = np.array(custom_label_map['name'])
        custom_label_map['location'] = np.array(custom_label_map['location'])
        custom_label_map['dimensions'] = np.array(custom_label_map['dimensions'])
        custom_label_map['rotation_y'] = np.array(custom_label_map['rotation_y'])
        custom_label_map['bbox'] = np.array(custom_label_map['bbox'])
        custom_label_map['diffculty'] = np.array(custom_label_map['diffculty'])
        annos = custom_label_map

        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox'].astype(np.float32)

        # selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        # gt_bboxes = gt_bboxes[selected].astype('float32')
        # gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names,
            diffculty=annos['diffculty'])
        return anns_results

    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 jsonfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str: float]: results of each evaluation metric
        """

        if jsonfile_prefix is not None and pklfile_prefix is None:
            pklfile_prefix = jsonfile_prefix

        # metrics = ['waymo', 'kitti']
        metric = {'waymo'}
        # assert metric in metrics, f'invalid metric {metric}'
        show = False
        if show:
            if out_dir is None:
                out_dir = pklfile_prefix
                # results = self.get_front_results(results)
            self.show(results, out_dir, pipeline=pipeline, max_show=10)

        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='kitti')
            from mmdet3d.core.evaluation import kitti_eval
            gt_annos = [info['annos'] for info in self.data_infos]

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
        if 'waymo' in metric:
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = os.path.join(eval_tmp_dir.name, 'results_dir')
            else:
                eval_tmp_dir = None

            torch.cuda.empty_cache()
            print_log('Starting format_waymo_results...', logger=logger)
            pd_bbox, pd_type, pd_frame_id, pd_score = self.format_waymo_results(
                results,
                pklfile_prefix,
                submission_prefix,
                logger
            )

            self.waymo_results_final_path = f'{pklfile_prefix}_T.pkl'
            print_log(f'pkl save to {self.waymo_results_final_path}', logger=logger)
            outputs = [pd_bbox.numpy(), pd_type.numpy(), pd_frame_id.numpy(), pd_score.numpy()]
            mmcv.dump(outputs, self.waymo_results_final_path)

            if self.split == 'testing_camera': return {}
            if self.gt_bin_file is None:
                print_log('Starting get_boxes_from_pkl...', logger=logger)
                gt_bbox, gt_type, gt_frame_id, difficulty = self.get_boxes_from_pkl()
            else:
                print_log(f'Starting get_boxes_from_gtbin...{self.gt_bin_file}', logger=logger)
                gt_bbox, gt_type, gt_frame_id, difficulty = self.get_boxes_from_gtbin(logger=logger, valid_pd_frame_id=set(pd_frame_id.numpy()))

            decoded_outputs = [[gt_bbox, gt_type, gt_frame_id, difficulty], [pd_bbox, pd_type, pd_frame_id, pd_score]]
            print_log('Starting compute_ap...', logger=logger)
            metrics = self.compute_ap(decoded_outputs)
            print_log('End.', logger=logger)
            results = {}
            for key in metrics:
                if 'TYPE_SIGN' in key:
                    continue
                metric_status = ('%s: %s') % (key, metrics[key].numpy())
                results[key] = metrics[key].numpy()
                print(metric_status)
            LET_AP = 0
            LET_APH = 0
            LET_APL = 0
            for cls in ['CYCLIST', 'PEDESTRIAN', 'VEHICLE']:
                LET_AP += metrics[f'3d_ap_OBJECT_TYPE_TYPE_{cls}_LEVEL_2'].numpy()
                LET_APH += metrics[f'3d_ap_ha_weighted_OBJECT_TYPE_TYPE_{cls}_LEVEL_2'].numpy()
                LET_APL += metrics[f'3d_ap_la_weighted_OBJECT_TYPE_TYPE_{cls}_LEVEL_2'].numpy()
            print('LET-AP', LET_AP / 3)
            print('LET-APH', LET_APH / 3)
            print('LET-APL', LET_APL / 3)

            return results
        else:
            print_log(metric)
            raise NotImplementedError
        return {}

    def format_waymo_results(self, outputs, pklfile_prefix=None, submission_prefix=None, logger=None):

        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = os.path.join(tmp_dir.name, 'results_dir')
        else:
            tmp_dir = None

        file_idx_list = []
        results_maps = {}

        for idx in range(len(outputs)):

            output = outputs[idx]['pts_bbox']
            box_preds, scores_3d, labels_3d = output['boxes_3d'], output['scores_3d'], output['labels_3d']

            try:
                #gt_bboxes = self.get_ann_info(idx)['gt_bboxes_3d']
                ann_info = self.get_ann_info(idx)
                gt_bboxes = ann_info['gt_bboxes_3d']
                gt_labels = ann_info['gt_labels_3d']

            except:
                gt_bboxes = None

            box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_inds = ((box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:])).all(-1)
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']

            data_info = self.get_data_info(idx)
            lidar2imgs = data_info['lidar2img']

            img_shapes = [(1280, 1920, 3), (1280, 1920, 3), (1280, 1920, 3), (886, 1920, 3), (886, 1920, 3)]

            cam_valid_list = []
            n_pred = box_preds.tensor.shape[0]
            for i in range(5):
                if n_pred == 0:
                    break
                corners_3d = box_preds.corners
                num_bbox = corners_3d.shape[0]
                pts_4d = torch.cat([corners_3d.view(-1, 3), torch.ones((num_bbox * 8, 1))], dim=-1)
                lidar2img = lidar2imgs[i]
                pts_2d = pts_4d @ lidar2img.T

                # every corner of obj should be in front of the camera
                valid_cam_inds = (pts_2d[:, 2].view(num_bbox, 8) > 0).all(-1)

                pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5, max=1e5)
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]
                imgfov_pts_2d = pts_2d[..., :2].view(num_bbox, 8, 2)

                image_shape = box_preds.tensor.new_tensor(img_shapes[i])

                minxy = imgfov_pts_2d.min(dim=1)[0]
                maxxy = imgfov_pts_2d.max(dim=1)[0]
                pred_2d_bboxes_camera = torch.cat([minxy, maxxy], dim=1)

                valid_cam_inds &= ((pred_2d_bboxes_camera[:, 0] < image_shape[1]) &
                                   (pred_2d_bboxes_camera[:, 1] < image_shape[0]) & (pred_2d_bboxes_camera[:, 2] > 0) &
                                   (pred_2d_bboxes_camera[:, 3] > 0))

                # calculate gt_2d_bboxes_camera
                if self.refine_with_2dbbox and gt_bboxes is not None:
                    bbox_in_cam = pred_2d_bboxes_camera[valid_cam_inds]
                    bbox_in_cam[:, [0, 2]] = torch.clamp(bbox_in_cam[:, [0, 2]], min=0, max=image_shape[1])
                    bbox_in_cam[:, [1, 3]] = torch.clamp(bbox_in_cam[:, [1, 3]], min=0, max=image_shape[0])

                    gt_2d_bboxes_camera_ = torch.tensor(
                        self.gt_bboxes_2d_info[str(sample_idx)]['gt_bboxes_2d'][i]).reshape(-1, 4)
                    cls2id = {
                        'Car': 0,
                        'Pedestrian': 1,
                        'Cyclist': 2,
                    }

                    gt_labels_ = torch.tensor(
                        [cls2id[each] for each in self.gt_bboxes_2d_info[str(sample_idx)]['gt_labels_2d'][i]])
                    len_gt = len(gt_labels_)

                    pairwise_iou = bbox_overlaps(bbox_in_cam,
                                                 torch.tensor(gt_2d_bboxes_camera_, device=bbox_in_cam.device))

                    _pred_labels = torch.as_tensor(labels_3d)[valid_cam_inds].unsqueeze(1).repeat(1, len_gt)

                    valid_cam_inds[valid_cam_inds.clone()] = ((pairwise_iou > 0.2) & (_pred_labels == gt_labels_)).any(
                        -1)


                    #valid_cam_inds[valid_cam_inds.clone()] = (pairwise_iou > 0.2).any(
                    #    -1)  # only bbox iou here, add labels_3d.numpy()[valid_inds] and gt_labels_3d should be better

                cam_valid_list.append(valid_cam_inds)
            if n_pred > 0:
                valid_cam_inds = cam_valid_list[0] | cam_valid_list[1] | cam_valid_list[2] | cam_valid_list[3] | \
                             cam_valid_list[4]
                valid_inds = valid_cam_inds & valid_inds
            if valid_inds.sum() > 0:
                result = dict(
                    bbox=None,
                    box3d_camera=None,
                    box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                    scores=scores_3d[valid_inds].numpy(),
                    label_preds=labels_3d[valid_inds].numpy(),
                    sample_idx=sample_idx)
            else:
                result = dict(
                    bbox=np.zeros([0, 4]),
                    box3d_camera=np.zeros([0, 7]),
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx)
            if self.split == 'training' or self.split == 'testing_camera':
                # results_list.append(result)
                idx = (sample_idx // 1000) % 1000
                frame_idx = sample_idx % 1000
                file_idx_list.append(idx)
                if idx not in results_maps.keys():
                    results_maps[idx] = {frame_idx: result}
                else:
                    results_maps[idx][frame_idx] = result


        file_idx_list = list(set(file_idx_list))

        pd_bbox, pd_type, pd_frame_id, pd_score = [], [], [], []

        id_time_map = {}
        with open(f'./filter_waymo.txt', 'r') as f:
            for each in f.readlines():
                id, time = each.strip().split(' ')
                id_time_map[int(id)] = int(time)

        k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        k2w_cls_map = np.array([k2w_cls_map[each] for each in self.CLASSES])

        for file_idx in file_idx_list:
            result = results_maps[file_idx]
            for frame_num in result.keys():
                frame_result = result[frame_num]
                sample_idx = frame_result['sample_idx']
                if sample_idx not in id_time_map:
                    continue

                n_pred = len(frame_result['label_preds'])
                if n_pred == 0:
                    continue
                width = frame_result['box3d_lidar'][:, 3]
                length = frame_result['box3d_lidar'][:, 4]
                height = frame_result['box3d_lidar'][:, 5]
                x = frame_result['box3d_lidar'][:, 0]
                y = frame_result['box3d_lidar'][:, 1]
                z = frame_result['box3d_lidar'][:, 2]
                z += height / 2

                rotation_y = frame_result['box3d_lidar'][:, 6]
                heading = -(rotation_y + np.pi / 2)
                while (heading < -np.pi).any():
                    heading[heading < -np.pi] += 2 * np.pi
                while (heading > np.pi).any():
                    heading[heading > np.pi] -= 2 * np.pi

                box = np.stack([x, y, z, length, width, height, heading], axis=-1)
                box = np.round(box, 4)

                cls = k2w_cls_map[frame_result['label_preds']]

                frame_id = np.full(n_pred, id_time_map[sample_idx])
                score = np.round(frame_result['scores'], 4)

                pd_bbox.append(box)
                pd_type.append(cls)
                pd_frame_id.append(frame_id)
                pd_score.append(score)

        pd_bbox = tf.concat(pd_bbox, axis=0)
        pd_type = tf.concat(pd_type, axis=0)
        pd_frame_id = tf.concat(pd_frame_id, axis=0)
        pd_score = tf.concat(pd_score, axis=0)
        return (pd_bbox, pd_type, pd_frame_id, pd_score)

    def get_boxes_from_pkl(self):
        id_time_map = {}
        with open(f'./filter_waymo.txt', 'r') as f:
            for each in f.readlines():
                id, time = each.strip().split(' ')
                id_time_map[int(id)] = int(time)

        k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        k2w_cls_map = np.array([k2w_cls_map[each] for each in self.CLASSES])

        gt_bbox, gt_type, gt_frame_id, difficulty = [], [], [], []
        for idx in range(len(self.data_infos)):
            ann_info = self.get_ann_info(idx)
            if ann_info is None:
                continue
            data_info = self.data_infos[idx]
            img_idx = data_info['image']['image_idx']
            frame_timestamp_micros = id_time_map[img_idx]
            n_pred = len(ann_info['gt_labels_3d'])

            cls = k2w_cls_map[ann_info['gt_labels_3d']]
            gt_bbox_3d = ann_info['gt_bboxes_3d'].tensor.numpy()

            width = np.round(gt_bbox_3d[:, 3], 4)
            length = np.round(gt_bbox_3d[:, 4], 4)
            height = np.round(gt_bbox_3d[:, 5], 4)
            x = np.round(gt_bbox_3d[:, 0], 4)
            y = np.round(gt_bbox_3d[:, 1], 4)
            z = np.round(gt_bbox_3d[:, 2], 4)
            z += height / 2

            rotation_y = np.round(gt_bbox_3d[:, 6], 4)
            heading = -(rotation_y + np.pi / 2)

            box = np.stack([x, y, z, length, width, height, heading], axis=-1)

            gt_bbox.append(box)
            gt_type.append(cls)
            gt_frame_id.append(np.full(n_pred, frame_timestamp_micros))
            difficulty.append(ann_info['diffculty'])

        gt_bbox = tf.concat(gt_bbox, axis=0)
        gt_type = tf.concat(gt_type, axis=0)
        gt_frame_id = tf.concat(gt_frame_id, axis=0)
        difficulty = tf.concat(difficulty, axis=0)

        return gt_bbox, gt_type, gt_frame_id, difficulty

    def get_boxes_from_gtbin(self, remove_gt=True, valid_pd_frame_id=None, score_th=None, logger=None):

        if os.path.exists(self.gt_bin_file + '.pkl'):
            print_log(f'Directly load from {self.gt_bin_file}.pkl', logger=logger)
            return mmcv.load(self.gt_bin_file + '.pkl')

        pd_bbox, pd_type, pd_frame_id, difficulty = [], [], [], []

        stuff1 = metrics_pb2.Objects()
        with open(self.gt_bin_file, 'rb') as rf:
            stuff1.ParseFromString(rf.read())

        print_log(f'Loading {len(stuff1.objects)} objects from gt_bin_file...', logger=logger)
        for i in range(len(stuff1.objects)):
            obj = stuff1.objects[i].object
            if obj.type == 3:  # label_pb2.Label.TYPE_SIGN
                continue
            if score_th is not None:
                if stuff1.objects[i].score <= score_th:
                    continue
            if remove_gt and obj.most_visible_camera_name == '':
                continue
            if valid_pd_frame_id is not None and stuff1.objects[i].frame_timestamp_micros not in valid_pd_frame_id:
                continue
            pd_frame_id.append(stuff1.objects[i].frame_timestamp_micros)
            box = tf.constant([obj.camera_synced_box.center_x, obj.camera_synced_box.center_y, obj.camera_synced_box.center_z,
                   obj.camera_synced_box.length, obj.camera_synced_box.width, obj.camera_synced_box.height,
                   obj.camera_synced_box.heading], dtype=tf.float32)
            pd_bbox.append(box)
            pd_type.append(obj.type)

            if obj.num_lidar_points_in_box:
                if obj.num_lidar_points_in_box <= 5 or obj.detection_difficulty_level == 2:
                    difficulty.append(2)
                else:
                    difficulty.append(1)
            else:
                difficulty.append(0)

        pd_bbox = tf.stack(pd_bbox)
        pd_type = tf.constant(pd_type, dtype=tf.uint8)
        pd_frame_id = tf.constant(pd_frame_id, dtype=tf.int64)
        difficulty = tf.constant(difficulty, dtype=tf.uint8)
        return pd_bbox, pd_type, pd_frame_id, difficulty

    def compute_ap(self, decoded_outputs):
        """Compute average precision."""
        [[gt_bbox, gt_type, gt_frame_id, difficulty], [pd_bbox, pd_type, pd_frame_id, pd_score]] = decoded_outputs

        scalar_metrics_3d, _ = self.build_waymo_metric(
            pd_bbox, pd_type, pd_score, pd_frame_id,
            gt_bbox, gt_type, gt_frame_id, difficulty)

        return scalar_metrics_3d

    def build_waymo_metric(self, pred_bbox, pred_class_id, pred_class_score,
                           pred_frame_id, gt_bbox, gt_class_id, gt_frame_id, difficulty,
                           gt_speed=None, box_type='3d', breakdowns=None):
        """Build waymo evaluation metric."""
        # metadata = waymo_metadata.WaymoMetadata()
        metadata = None
        if breakdowns is None:
            # breakdowns = ['RANGE', 'SIZE', 'OBJECT_TYPE']
            breakdowns = ['RANGE', 'OBJECT_TYPE']
        waymo_metric_config = self._build_waymo_metric_config(
            metadata, box_type, breakdowns)

        def detection_metrics(prediction_bbox,
                              prediction_type,
                              prediction_score,
                              prediction_frame_id,
                              prediction_overlap_nlz,
                              ground_truth_bbox,
                              ground_truth_type,
                              ground_truth_frame_id,
                              ground_truth_difficulty,
                              config,
                              ground_truth_speed=None):
            if ground_truth_speed is None:
                num_gt_boxes = tf.shape(ground_truth_bbox)[0]
                ground_truth_speed = tf.zeros((num_gt_boxes, 2), dtype=tf.float32)
            metrics_module = tf.load_op_library(
                tf.compat.v1.resource_loader.get_path_to_datafile('metrics_ops.so'))
            return metrics_module.detection_metrics(
                prediction_bbox=prediction_bbox,
                prediction_type=prediction_type,
                prediction_score=prediction_score,
                prediction_frame_id=prediction_frame_id,
                prediction_overlap_nlz=prediction_overlap_nlz,
                ground_truth_bbox=ground_truth_bbox,
                ground_truth_type=ground_truth_type,
                ground_truth_frame_id=ground_truth_frame_id,
                ground_truth_difficulty=ground_truth_difficulty,
                ground_truth_speed=ground_truth_speed,
                config=config)

        ap, ap_ha, ap_la, pr, pr_ha, pr_la, tmp = detection_metrics(
            prediction_bbox=tf.cast(pred_bbox, tf.float32),
            prediction_type=tf.cast(pred_class_id, tf.uint8),
            prediction_score=tf.cast(pred_class_score, tf.float32),
            prediction_frame_id=tf.cast(pred_frame_id, tf.int64),
            prediction_overlap_nlz=tf.zeros_like(pred_frame_id, dtype=tf.bool),
            ground_truth_bbox=tf.cast(gt_bbox, tf.float32),
            ground_truth_type=tf.cast(gt_class_id, tf.uint8),
            ground_truth_frame_id=tf.cast(gt_frame_id, tf.int64),
            ground_truth_difficulty=tf.cast(difficulty, dtype=tf.uint8),
            ground_truth_speed=None,
            config=waymo_metric_config.SerializeToString())

        # All tensors returned by Waymo's metric op have a leading dimension
        # B=number of breakdowns. At this moment we always use B=1 to make
        # it compatible to the python code.

        scalar_metrics = {'%s_ap' % box_type: ap[0],
                          '%s_ap_ha_weighted' % box_type: ap_ha[0],
                          '%s_ap_la_weighted' % box_type: ap_la[0]}
        curve_metrics = {'%s_pr' % box_type: pr[0],
                         '%s_pr_ha_weighted' % box_type: pr_ha[0], }
        # '%s_pr_la_weighted' % box_type: pr_la[0]}

        breakdown_names = config_util.get_breakdown_names_from_config(
            waymo_metric_config)
        for i, metric in enumerate(breakdown_names):
            # There is a scalar / curve for every breakdown.
            scalar_metrics['%s_ap_%s' % (box_type, metric)] = ap[i]
            scalar_metrics['%s_ap_ha_weighted_%s' % (box_type, metric)] = ap_ha[i]
            scalar_metrics['%s_ap_la_weighted_%s' % (box_type, metric)] = ap_la[i]
            curve_metrics['%s_pr_%s' % (box_type, metric)] = pr[i]
            curve_metrics['%s_pr_ha_weighted_%s' % (box_type, metric)] = pr_ha[i]
            # curve_metrics['%s_pr_la_weighted_%s' % (box_type, metric)] = pr_la[i]
        return scalar_metrics, curve_metrics

    def _build_waymo_metric_config(self, metadata, box_type, waymo_breakdown_metrics):
        """Build the Config proto for Waymo's metric op."""
        config = metrics_pb2.Config()
        # num_pr_points = metadata.NumberOfPrecisionRecallPoints()
        num_pr_points = 101
        config.score_cutoffs.extend(
            [i * 1.0 / (num_pr_points - 1) for i in range(num_pr_points)])
        config.matcher_type = metrics_pb2.MatcherProto.Type.TYPE_HUNGARIAN
        if box_type == '2d':
            config.box_type = label_pb2.Label.Box.Type.TYPE_2D
        else:
            config.box_type = label_pb2.Label.Box.Type.TYPE_3D
        # Default values
        config.iou_thresholds[:] = [0.0, 0.5, 0.3, 0.3, 0.3]
        diff = metrics_pb2.Difficulty()
        # diff.levels.append(1)
        diff.levels.append(2)

        config.let_metric_config.enabled = True
        config.let_metric_config.longitudinal_tolerance_percentage = 0.1
        config.let_metric_config.min_longitudinal_tolerance_meter = 0.5
        config.let_metric_config.sensor_location.x = 1.43
        config.let_metric_config.sensor_location.y = 0
        config.let_metric_config.sensor_location.z = 2.18

        config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.ONE_SHARD)
        config.difficulties.append(diff)
        # Add extra breakdown metrics.
        for breakdown_value in waymo_breakdown_metrics:
            breakdown_id = breakdown_pb2.Breakdown.GeneratorId.Value(breakdown_value)
            config.breakdown_generator_ids.append(breakdown_id)
            config.difficulties.append(diff)
        return config

    def show(self, results, out_dir='.', show=True, pipeline=None, max_show=100):
        """Results visualization.
        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        # pipeline = self._get_pipeline(pipeline)
        count = 0
        for i, result in enumerate(results):
            if i % 100 != 0:
                continue
            count += 1
            if count > max_show: break

            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']



            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))
            scores = result['scores_3d'].cpu().numpy()

            data_info = self.get_data_info(i)
            lidar2img = data_info['lidar2img']

            img_filename_0 = self.data_infos[i]['image']['image_path']
            image_filenames = []
            for j in range(5):
                image_filenames.append(img_filename_0.replace('image_0', f'image_{j}'))
            try:
                ann_info = self.get_ann_info(i)
                if ann_info is None:
                    continue
                show_gt_bboxes = ann_info['gt_bboxes_3d']
            except:
                show_gt_bboxes = None
            for j in range(5):
                show_multi_modality_result(
                    mmcv.imread(os.path.join(self.data_root, image_filenames[j])),
                    show_gt_bboxes,
                    show_pred_bboxes,
                    lidar2img[j],
                    out_dir,
                    image_filenames[j],
                    box_mode='lidar',
                    show=show,
                    scores=scores,
                )

            bev_img = np.zeros([1500, 1100, 3], dtype=np.float32)

            def world2bev_vis(x, y):
                return int((x + 35) * 10), int((-y + 75) * 10)

            for corners in show_pred_bboxes.corners[:, [4, 7, 3, 0], :2]:
                corners = np.array([world2bev_vis(*corner) for corner in corners])
                _img = np.zeros([1500, 1100, 3], dtype=np.float32)
                _img = cv2.fillPoly(_img, [corners], (241, 101, 72))
                bev_img = cv2.addWeighted(bev_img, 1, _img, 0.5, 0)
            try:
                for corners in show_gt_bboxes.corners[:, [4, 7, 3, 0], :2]:
                    corners = np.array([world2bev_vis(*corner) for corner in corners])
                    _img = np.zeros([1500, 1100, 3], dtype=np.float32)
                    _img = cv2.fillPoly(_img, [corners], (61, 102, 255))
                    bev_img = cv2.addWeighted(bev_img, 1, _img, 0.5, 0)
            except:
                pass

            bev_img = cv2.circle(bev_img, world2bev_vis(0, 0), 5, (0, 255, 0), thickness=-1)

            bev_out_path = os.path.join(out_dir, img_filename_0.replace('image_0', f'bev'))
            mmcv.imwrite(bev_img, bev_out_path)


@DATASETS.register_module()
class WaymoDataset_videoV3(WaymoDataset_videoV2):

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index - self.queue_length + 1, index + 1))

        # index_list = sorted(index_list[1:])
        seed = time.time()
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict, seed=seed)
            queue.append(example)
        return self.union2one_test(queue)

    def union2one_test(self, queue):

        imgs_list = [each['img'][0].data for each in queue]
        if 'points' in queue[0].keys():
            points_list = [each['points'].data for each in queue]
            queue[-1]['points'] = DC(points_list, cpu_only=False, stack=False)
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'][0].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = metas_map[i]['can_bus'][:3]
                prev_angle = metas_map[i]['can_bus'][-1]
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = metas_map[i]['can_bus'][:3]
                tmp_angle = metas_map[i]['can_bus'][-1]
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = tmp_pos
                prev_angle = tmp_angle

        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue


@DATASETS.register_module()
class WaymoDataset_videoV4(WaymoDataset_videoV2):
    """
    WaymoDataset with 2d mask.
    """


    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        Args:
            index (int): Index of the annotation data to get.
        Returns:
            dict: annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api

        info = self.data_infos[index]
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        # annos = info['annos'] do not use this for mutli-view
        # we need other objects to avoid collision when sample

        custom_label_map = {
            'name': [],  # 1
            'bbox': [],  # 4
            'dimensions': [],  # 3
            'location': [],  # 3
            'rotation_y': [],  # 1
        }
        if self.use_pkl_annos:
            annos = info['annos']
            keep = (annos['bbox'] != 0).any(-1)
            custom_label_map['name'] = annos['name'][keep]
            custom_label_map['location'] = annos['location'][keep]
            custom_label_map['dimensions'] = annos['dimensions'][keep]
            custom_label_map['rotation_y'] = annos['rotation_y'][keep]
            custom_label_map['bbox'] = annos['bbox'][keep]
            if len(custom_label_map['name']) == 0:
                return None

        else:
            lable_all_path = info['image']['image_path'].replace('image_0', 'label_all').replace('.png', '.txt')
            lable_all_path = os.path.join(self.data_root, lable_all_path)
            label_all = []
            for line in mmcv.load(lable_all_path):
                box_info = line.strip().split(' ')
                if box_info[4:8] == ['0', '0', '0', '0']:  # remove back bboxes
                    continue
                label_all.append(box_info)
            if len(label_all) == 0:
                return None
            for info in label_all:
                custom_label_map['name'].append(info[0])
                custom_label_map['location'].append(info[11:11 + 3])
                custom_label_map['dimensions'].append([info[10], info[8], info[9]])
                custom_label_map['rotation_y'].append(info[14])
                custom_label_map['bbox'].append(info[4:8])
            custom_label_map['name'] = np.array(custom_label_map['name'])
            custom_label_map['location'] = np.array(custom_label_map['location'])
            custom_label_map['dimensions'] = np.array(custom_label_map['dimensions'])
            custom_label_map['rotation_y'] = np.array(custom_label_map['rotation_y'])
            custom_label_map['bbox'] = np.array(custom_label_map['bbox'])

        annos = custom_label_map

        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))

        gt_labels_3d = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d).astype(np.int64)

        if self.gt_bboxes_2d_info:
            sample_idx = str(info['image']['image_idx']).zfill(7)

            gt_bboxes_2d = self.gt_bboxes_2d_info[sample_idx]['gt_bboxes_2d']
            gt_bboxes_2d = [_.astype(np.float32) for _ in gt_bboxes_2d]

            gt_names_2d = self.gt_bboxes_2d_info[sample_idx]['gt_labels_2d']
            gt_labels_2d = [[] for _ in range(5)]
            for idx, gt_names_2d_percam in enumerate(gt_names_2d):
                for cat in gt_names_2d_percam:
                    if cat in self.CLASSES:
                        gt_labels_2d[idx].append(self.CLASSES.index(cat))
                    else:
                        gt_labels_2d[idx].append(-1)
            gt_labels_2d = [np.array(_).astype(np.int64) for _ in gt_labels_2d]
        else:
            gt_bboxes_2d = [[] for _ in range(5)]
            gt_labels_2d = [[] for _ in range(5)]

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes_2d,
            labels=gt_labels_2d,
            gt_names=gt_names)

        return anns_results
