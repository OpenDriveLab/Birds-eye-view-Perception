# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Tests for waymo_open_dataset.metrics.python.detection_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyquaternion
import mmcv
ERROR = 1e-6
from google.protobuf import text_format
"""Utitlities."""
#import functools
import argparse

from lingvo.tasks.car.waymo import waymo_metadata

# from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import breakdown_pb2

import os
import tempfile
from os import path as osp



try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
    # open_dataset.CameraName.
    import numpy as np
    import tensorflow as tf
    from glob import glob
    from os.path import join
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

except ImportError:
    print('WARNING check waymo_dataset')



class BinSaver():

    CLASSES = ('Unknown', 'Car', 'Pedestrian', 'Sign', 'Cyclist')
    def __init__(self,
                data_root,
                split,
                time_id_map
                 ):
        self.data_root = data_root
        self.split = split
        self.time_id_map = time_id_map

    def parse_objects(self, result, context_name,
                      frame_timestamp_micros):
        """Parse one prediction with several instances in kitti format and
        convert them to `Object` proto.
        Args:
            kitti_result (dict): Predictions in kitti format.
                - name (np.ndarray): Class labels of predictions.
                - dimensions (np.ndarray): Height, width, length of boxes.
                - location (np.ndarray): Bottom center of boxes (x, y, z).
                - rotation_y (np.ndarray): Orientation of boxes.
                - score (np.ndarray): Scores of predictions.
            T_k2w (np.ndarray): Transformation matrix from kitti to waymo.
            context_name (str): Context name of the frame.
            frame_timestamp_micros (int): Frame timestamp.
        Returns:
            :obj:`Object`: Predictions in waymo dataset Object proto.
        """

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        def parse_one_object(instance_idx):

            """Parse one instance in kitti format and convert them to `Object`
            proto.
            Args:
                instance_idx (int): Index of the instance to be converted.
            Returns:
                :obj:`Object`: Predicted instance in waymo dataset \
                    Object proto.
            """

            cls = self.CLASSES[result['label_preds'][instance_idx]]

            length = round(result['box3d_lidar'][instance_idx, 3], 4)
            width = round(result['box3d_lidar'][instance_idx, 4], 4)
            height = round(result['box3d_lidar'][instance_idx, 5], 4)
            x = round(result['box3d_lidar'][instance_idx, 0], 4)
            y = round(result['box3d_lidar'][instance_idx, 1], 4)
            z = round(result['box3d_lidar'][instance_idx, 2], 4)
            rotation_y = round(result['box3d_lidar'][instance_idx, 6], 4)
            score = round(result['scores'][instance_idx], 4)
            heading = rotation_y
            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            # o.object.camera_synced_box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[cls]
            o.score = score

            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros

            return o

        objects = metrics_pb2.Objects()

        for instance_idx in range(len(result['label_preds'])):
            o = parse_one_object(instance_idx)
            objects.objects.append(o)

        return objects

    def convert_one(self, idx):

        file_idx = self.file_idx_list[idx]

        result = self.results_maps[file_idx]
        file_pathname = self.waymo_tfrecord_pathnames[idx]


        file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')
        for frame_num, frame_data in enumerate(file_data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))
            filename = f'{self.prefix}{file_idx:03d}{frame_num:03d}'
            context_name = frame.context.name
            frame_timestamp_micros = frame.timestamp_micros
            if frame_num in result.keys():
                # if filename in self.name2idx:
                objects = self.parse_objects(result[frame_num], context_name,
                                             frame_timestamp_micros)
            else:
                objects = metrics_pb2.Objects()
            # else:
                print(filename, 'not found.')
            #    objects = metrics_pb2.Objects()

            with open(
                    join(self.waymo_results_save_dir, f'{filename}.bin'),
                    'wb') as f:
                f.write(objects.SerializeToString())

    def get_file_names(self):
        """Get file names of waymo raw data."""
        self.waymo_tfrecord_pathnames = sorted(
            glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

    def combine(self, pathnames):
        """Combine predictions in waymo format for each sample together.
        Args:
            pathnames (str): Paths to save predictions.
        Returns:
            :obj:`Objects`: Combined predictions in Objects proto.
        """
        combined = metrics_pb2.Objects()

        for pathname in pathnames:
            objects = metrics_pb2.Objects()
            with open(pathname, 'rb') as f:
                objects.ParseFromString(f.read())
            for o in objects.objects:
                combined.objects.append(o)
        return combined


    def format_waymo_results(self, outputs, pklfile_prefix=None):
        pd_bbox, pd_type, pd_frame_id, pd_score = outputs
        pd_bbox, pd_type, pd_score, pd_frame_id = pd_bbox.numpy(), pd_type.numpy(), pd_score.numpy(), pd_frame_id.numpy()

        id_list = []
        for time in pd_frame_id:
            if str(time) in self.time_id_map:
                id_list.append(int(self.time_id_map[str(time)]))
            else:
                id_list.append(-1)
        id_list = np.array(id_list)
        id_set = list(set(id_list))
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        waymo_root = osp.join(
            self.data_root.split('kitti_format')[0], 'waymo_format')
        if self.split == 'training':
            self.waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
            self.prefix = '1'
        elif self.split == 'testing':
            self.waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
            self.prefix = '2'
        elif self.split == 'testing_camera':
            self.waymo_tfrecords_dir = osp.join(waymo_root, 'testing_camera')
            self.prefix = '3'
        else:
            raise ValueError('Not supported split value.')
        save_tmp_dir = tempfile.TemporaryDirectory()
        self.waymo_results_save_dir = save_tmp_dir.name
        #self.waymo_results_save_dir = '.cache'
        self.waymo_results_final_path = f'{pklfile_prefix}.bin'
        self.get_file_names()
        file_idx_list = []
        results_maps = {}

        for id_ in id_set:
            if id_ == -1: continue
            index = (id_list == id_)

            # from IPython import embed
            # embed()
            # exit()
            pd_bbox_ = pd_bbox[index]
            pd_type_ = pd_type[index]
            pd_score_ = pd_score[index]
            sample_idx = id_


            result = dict(
                    bbox=None,
                    box3d_camera=None,
                    box3d_lidar=pd_bbox_,
                    scores=pd_score_,
                    label_preds=pd_type_,
                    sample_idx=sample_idx)

            if self.split == 'training':
                # results_list.append(result)
                idx = (sample_idx // 1000) % 1000
                frame_idx = sample_idx % 1000
                file_idx_list.append(idx)
                if idx not in results_maps.keys():
                    results_maps[idx] = {frame_idx: result}
                else:
                    results_maps[idx][frame_idx] = result
            elif self.split == 'testing_camera':
                idx = (sample_idx // 1000) % 1000
                frame_idx = sample_idx % 1000
                file_idx_list.append(idx)
                if idx not in results_maps.keys():
                    results_maps[idx] = {frame_idx: result}
                else:
                    results_maps[idx][frame_idx] = result
                #assert False
        print('Start converting ...')

        self.file_idx_list = list(set(file_idx_list))
        print(self.file_idx_list)
        self.results_maps = results_maps
        for i, key in enumerate(results_maps.keys()):
            print(i, len(results_maps[key]))
        # print(len(results_maps[0]), len(results_maps[1]))
        import mmcv
        mmcv.track_parallel_progress(self.convert_one, range(len(self.file_idx_list)), 32)
        print('\nFinished ...')
        # self.convert_one(idx, results)

        pathnames = sorted(glob(join(self.waymo_results_save_dir, '*.bin')))
        combined = self.combine(pathnames)
        mmcv.mkdir_or_exist(pklfile_prefix)
        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())
        print('save bin to', self.waymo_results_final_path)
        return self.waymo_results_final_path, tmp_dir


def get_time_id_map(file_path='./filter_waymo.txt'):
    time_id_map = {}
    id_useless_gt = {}
    with open(file_path, 'r') as f:
        for each in f.readlines()[:-1]:
            id, time = each.strip().split(' ')
            id_useless_gt[id] = []
            try:
                with open(f'data/waymo/kitti_format/training/label_all/{id}.txt', 'r') as g:
                    for box_info in g.readlines():
                        box_info = box_info.strip().split(' ')
                        if box_info[4:8] == ['0', '0', '0', '0']:
                            id_useless_gt[id].append(['%.2f' % float(each) for each in box_info[8:11]])
            except:
                pass
            time_id_map[time] = id
    return time_id_map, id_useless_gt

def front_get_time_id_map(file_path='./filter_waymo.txt'):
    time_id_map = {}
    id_usefull_gt = {}
    usefull = 0
    with open(file_path, 'r') as f:
        for each in f.readlines()[:-1]:
            id, time = each.strip().split(' ')
            id_usefull_gt[id] = []
            try:
                with open(f'data/waymo/kitti_format/training/label_0/{id}.txt', 'r') as g:
                    for box_info in g.readlines():
                        box_info = box_info.strip().split(' ')
                        if box_info[4:8] != ['0', '0', '0', '0']:
                            id_usefull_gt[id].append(['%.2f' % float(each) for each in box_info[8:11]])
                            usefull += 1
            except:
                print(f'data/waymo/kitti_format/training/label_0/{id}.txt ', 'not found')
                pass
            time_id_map[time] = id
    print('usefull ', usefull)
    return time_id_map, id_usefull_gt

# a, b = get_time_id_map(file_path='./filter_waymo.txt')




def get_boxes_from_bin(file, remove_gt=False, remove_pred=False, keep_front=False, score_th=None, gt=False):
    # if keep_front:
    #     time_id_map, id_usefull_gt = front_get_time_id_map()
    # elif remove_gt:
    #     time_id_map, id_useless_gt = get_time_id_map()

    pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = [], [], [], [], []
    stuff1 = metrics_pb2.Objects()

    with open(file, 'rb') as rf:
        stuff1.ParseFromString(rf.read())
        print('len gt', len(stuff1.objects))
        for i in range(len(stuff1.objects)):
            obj = stuff1.objects[i].object

            # print(dir(stuff1.objects[1]))
            # print(stuff1.objects[1].context_name)
            # print(stuff1.objects[1].frame_timestamp_micros)

            # print(['%.2f' % obj.box.height, '%.2f' % obj.box.width, '%.2f' % obj.box.length])
            # from IPython import embed
            # embed()
            if keep_front:
                pass
                # try:
                #     id = time_id_map[str(stuff1.objects[i].frame_timestamp_micros)]
                # except:
                #     continue
                # if ['%.2f' % obj.box.height, '%.2f' % obj.box.width, '%.2f' % obj.box.length] not in id_usefull_gt[id]:
                #     # print('not front')
                #     continue
            elif remove_gt and obj.most_visible_camera_name == '':
                continue
                # try:
                #     id = time_id_map[str(stuff1.objects[i].frame_timestamp_micros)]
                # except:
                #     continue
                # if ['%.2f' % obj.box.height, '%.2f' % obj.box.width, '%.2f' % obj.box.length] in id_useless_gt[id]:
                #     continue
            # print(obj.num_lidar_points_in_box)
            if obj.type == 3:
                continue
            if score_th is not None:
                if stuff1.objects[i].score <= score_th:
                    continue
            pd_frame_id.append(stuff1.objects[i].frame_timestamp_micros)
            if gt:
                box = [obj.camera_synced_box.center_x, obj.camera_synced_box.center_y, obj.camera_synced_box.center_z,
                 obj.camera_synced_box.length, obj.camera_synced_box.width, obj.camera_synced_box.height, obj.camera_synced_box.heading]
            else:
                box = [obj.box.center_x, obj.box.center_y, obj.box.center_z,
                 obj.box.length, obj.box.width, obj.box.height, obj.box.heading]
            pd_bbox.append(box)
            pd_score.append(stuff1.objects[i].score)
            pd_type.append(obj.type)
            # difficulty.append(2)
            if obj.num_lidar_points_in_box:
                if obj.num_lidar_points_in_box <= 5 or obj.detection_difficulty_level == 2:
                    difficulty.append(2)
                else:
                    difficulty.append(1)
            else:
                difficulty.append(0)
    pd_bbox = np.array(pd_bbox)
    pd_bbox = tf.convert_to_tensor(pd_bbox)
    # from IPython import embed
    # embed()
    # exit()
    pd_type = tf.concat(pd_type, axis=0)
    pd_frame_id = tf.concat(pd_frame_id, axis=0)
    pd_score = tf.concat(pd_score, axis=0)
    difficulty = tf.concat(difficulty, axis=0)
    return pd_bbox, pd_type, pd_frame_id, pd_score, difficulty



    # pd_bbox, pd_type, pd_frame_id, pd_score, _ = self.get_boxes_from_txt(pd_file)


def eval(pd_file, gt_file='data/waymo/waymo_format/gt_synced.bin'):
    gt_bbox, gt_type, gt_frame_id, _, difficulty = get_boxes_from_bin(gt_file, remove_gt=True, gt=True)
    if pd_file.endswith('bin'):
        pd_bbox, pd_type, pd_frame_id, pd_score, _ = get_boxes_from_bin(pd_file, remove_pred=False)
    elif pd_file.endswith('pkl'):

        pd_bbox, pd_type, pd_frame_id, pd_score = mmcv.load(pd_file)
        pd_bbox = tf.convert_to_tensor(pd_bbox)
        pd_type = tf.convert_to_tensor(pd_type)
        pd_frame_id = tf.convert_to_tensor(pd_frame_id)
        pd_score = tf.convert_to_tensor(pd_score)

    decoded_outputs = [[gt_bbox, gt_type, gt_frame_id, difficulty], [pd_bbox, pd_type, pd_frame_id, pd_score]]
    metrics = compute_ap(decoded_outputs)
    # print(metrics)
    results = {}
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

from absl import flags


def define_flags():
    """Add training flags."""

    flags.DEFINE_string('master', 'local', 'Location of the session.')

    flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU.')

    flags.DEFINE_string('model_dir', '/tmp/results/ped_eval/',
                        'training directory root')

    # Optimizer
    flags.DEFINE_float('lr', 3e-3, 'learning rate')
    flags.DEFINE_integer('epochs', 75, 'number of batches to train on')

    # Dataset
    flags.DEFINE_integer('batch_size', 1,
                         'batch size for training')
    flags.DEFINE_integer('test_batch_size', 2,
                         'batch size for testing')
    flags.DEFINE_integer('cycle_length', 128,
                         'number of parallel file readers')
    flags.DEFINE_integer('num_parallel_calls', 128,
                         'number of parallel dataloader threads')
    flags.DEFINE_integer('shuffle_buffer_size', 1024,
                         'buffer size for shuffling data')
    flags.DEFINE_float('percentile', 1.00, 'percentile of validation data.')
    flags.DEFINE_string('data_path', '/home/yuewang/data/waymo/processed', 'data path')

    # Task specific params
    flags.DEFINE_integer('max_num_points', 245760,
                         'maximum number of lidar points')
    flags.DEFINE_integer('max_num_bboxes', 200,
                         'maximum number of bounding bboxes')
    flags.DEFINE_integer('class_id', 2, 'class id (car=1, pedestrian=2)')
    flags.DEFINE_integer('difficulty', 1, 'difficulty level (1 or 2)')

    flags.DEFINE_integer('pillar_map_size', 512,
                         'birds-eye view pillar size (256 for car, 512 for pedestrian)')
    flags.DEFINE_float('pillar_map_range', 75.2, 'birds-eye view detection range')

    flags.DEFINE_string('norm_type', 'sync_batch_norm',
                        'normalization type to use')
    flags.DEFINE_string('act_type', 'relu',
                        'activation type to use')
    flags.DEFINE_float('nms_iou_threshold', 0.2,
                       'nms ios threshold (0.7 for car, 0.2 for pedestrian)')
    flags.DEFINE_float('nms_score_threshold', 0.0, 'prediction score threshold')
    flags.DEFINE_integer('max_nms_boxes', 200,
                         'maximum number of bounding boxes to keep after NMS')
    flags.DEFINE_bool('use_oriented_per_class_nms', True,
                      'whether to use oriented NMS')

    # For evaluation
    flags.DEFINE_bool('eval_once', True, 'eval once or forever (during training)')
    flags.DEFINE_string('ckpt_path', '/home/yuewang/data/waymo_pretrained_model/eccv/ped/ped',
                        'checkpoint path')
    return flags.FLAGS


FLAGS = define_flags()


def get_shape(tensor, ndims=None):
    """Returns tensor's shape as a list which can be unpacked, unlike tf.shape.
    Tries to return static shape if it's available. Note that this means
    some of the outputs will be ints while the rest will be Tensors.
    Args:
      tensor: The input tensor.
      ndims: If not None, returns the shapes for the first `ndims` dimensions.
    """
    tensor = tf.convert_to_tensor(tensor)
    dynamic_shape = tf.shape(tensor)

    # Early exit for unranked tensor.
    if tensor.shape.ndims is None:
        if ndims is None:
            return dynamic_shape
        else:
            return [dynamic_shape[x] for x in range(ndims)]

    # Ranked tensor.
    if ndims is None:
        ndims = tensor.shape.ndims
    else:
        ndims = min(ndims, tensor.shape.ndims)

    # Return mixture of static and dynamic dims.
    static_shape = tensor.shape.as_list()
    shapes = [
        static_shape[x] if static_shape[x] is not None else dynamic_shape[x]
        for x in range(ndims)
    ]

    return shapes


def compute_ap(decoded_outputs, save_bin=False):
    """Compute average precision."""
    [[gt_bbox, gt_type, gt_frame_id, difficulty], [pd_bbox, pd_type, pd_frame_id, pd_score]] = decoded_outputs
    if save_bin:
        bin_saver = BinSaver(data_root='data/waymo',
                split='training',
                time_id_map=get_time_id_map()[0])
        bin_saver.format_waymo_results([pd_bbox, pd_type, pd_frame_id, pd_score], pklfile_prefix='output.bin')
    scalar_metrics_3d, _ = build_waymo_metric(
        pd_bbox, pd_type, pd_score, pd_frame_id,
        gt_bbox, gt_type, gt_frame_id, difficulty)

    return scalar_metrics_3d


def build_waymo_metric(pred_bbox, pred_class_id, pred_class_score,
                       pred_frame_id, gt_bbox, gt_class_id, gt_frame_id, difficulty,
                       gt_speed=None, box_type='3d', breakdowns=None):
    """Build waymo evaluation metric."""
    metadata = waymo_metadata.WaymoMetadata()
    if breakdowns is None:
        breakdowns = ['RANGE', 'OBJECT_TYPE']
    waymo_metric_config = _build_waymo_metric_config(
        metadata, box_type, breakdowns)
    # embed()
    # waymo_metric_config = build_config()

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

    # ap, ap_ha, pr, pr_ha, tmp = py_metrics_ops.detection_metrics(
    #     prediction_bbox=tf.cast(pred_bbox, tf.float32),
    #     prediction_type=tf.cast(pred_class_id, tf.uint8),
    #     prediction_score=tf.cast(pred_class_score, tf.float32),
    #     prediction_frame_id=tf.cast(pred_frame_id, tf.int64),
    #     prediction_overlap_nlz=tf.zeros_like(pred_frame_id, dtype=tf.bool),
    #     ground_truth_bbox=tf.cast(gt_bbox, tf.float32),
    #     ground_truth_type=tf.cast(gt_class_id, tf.uint8),
    #     ground_truth_frame_id=tf.cast(gt_frame_id, tf.int64),
    #     ground_truth_difficulty=tf.cast(difficulty, dtype=tf.uint8),
    #     ground_truth_speed=None,
    #     config=waymo_metric_config.SerializeToString())
    #
    # # All tensors returned by Waymo's metric op have a leading dimension
    # # B=number of breakdowns. At this moment we always use B=1 to make
    # # it compatible to the python code.
    #
    # scalar_metrics = {'%s_ap' % box_type: ap[0],
    #                   '%s_ap_ha_weighted' % box_type: ap_ha[0]}
    # curve_metrics = {'%s_pr' % box_type: pr[0],
    #                  '%s_pr_ha_weighted' % box_type: pr_ha[0]}
    #
    # breakdown_names = config_util.get_breakdown_names_from_config(
    #     waymo_metric_config)
    # for i, metric in enumerate(breakdown_names):
    #     # There is a scalar / curve for every breakdown.
    #     scalar_metrics['%s_ap_%s' % (box_type, metric)] = ap[i]
    #     scalar_metrics['%s_ap_ha_weighted_%s' % (box_type, metric)] = ap_ha[i]
    #     curve_metrics['%s_pr_%s' % (box_type, metric)] = pr[i]
    #     curve_metrics['%s_pr_ha_weighted_%s' % (box_type, metric)] = pr_ha[i]
    # return scalar_metrics, curve_metrics


def build_config():
        config = metrics_pb2.Config()
        config_text = """
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels:1
        levels:2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.0
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        """

        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)

        text_format.Merge(config_text, config)
        return config


def _build_waymo_metric_config(metadata, box_type, waymo_breakdown_metrics):
    """Build the Config proto for Waymo's metric op."""
    config = metrics_pb2.Config()
    num_pr_points = metadata.NumberOfPrecisionRecallPoints()
    config.score_cutoffs.extend(
        [i * 1.0 / (num_pr_points - 1) for i in range(num_pr_points)])
    config.matcher_type = metrics_pb2.MatcherProto.Type.TYPE_HUNGARIAN
    if box_type == '2d':
        config.box_type = label_pb2.Label.Box.Type.TYPE_2D
    else:
        config.box_type = label_pb2.Label.Box.Type.TYPE_3D
    # Default values
    config.iou_thresholds[:] = [0.0, 0.5, 0.3, 0.3, 0.3]  # None, Car,  Cyclist, Sign, Pedestrian
    # config.iou_thresholds[:] = [0.0, -1.0, -1.0, -1.0, -1.0]
    # for class_name, threshold in metadata.IoUThresholds().items():
    #
    #    cls_idx = metadata.ClassNames().index(class_name)
    #    print(cls_idx, class_name, threshold)
    #    config.iou_thresholds[cls_idx] = threshold
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
    # config_text = """
    # difficulties {
    # levels:1
    # levels:2
    # }
    # """
    # text_format.Merge(config_text, config)
    return config


class_maps = {
    1: 'car',
    2: 'bicycle',
    4: 'pedestrian'
}

def load_gt(gt_bbox, gt_type, gt_frame_id):
    from projects.mmdet3d_plugin.datasets.nuscnes_eval import DetectionBox_modified
    from nuscenes.eval.common.data_classes import EvalBoxes
    import tqdm
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    gt_bbox = gt_bbox.numpy()
    gt_type = gt_type.numpy()
    gt_frame_id = gt_frame_id.numpy()
    sample_tokens = list(set(gt_frame_id))
    all_annotations = EvalBoxes()
    # Load annotations and filter predictions and annotations.

    for sample_token in tqdm.tqdm(sample_tokens):
        index = gt_frame_id == sample_token
        gt_bbox_ = gt_bbox[index]
        gt_type_ = gt_type[index]
        sample_boxes = []
        for bbox, type_ in zip(gt_bbox_, gt_type_):
            if type_ == 3:
                continue
            sample_boxes.append(
                    DetectionBox_modified(
                        token='',
                        sample_token=str(sample_token),
                        translation=bbox[:3],
                        size=bbox[3:6],
                        rotation=pyquaternion.Quaternion(axis=[0, 0, 1], radians=bbox[-1]).elements,
                        velocity=[0, 0],
                        num_pts=100,
                        detection_name=class_maps[type_],
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name='',
                    )
                )

        all_annotations.add_boxes(str(sample_token), sample_boxes)
    return all_annotations


def load_pd(pd_bbox, pd_type, pd_frame_id, pd_score):
    from projects.mmdet3d_plugin.datasets.nuscnes_eval import DetectionBox_modified
    from nuscenes.eval.common.data_classes import EvalBoxes
    import tqdm
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    pd_bbox = pd_bbox.numpy()
    pd_type = pd_type.numpy()
    pd_frame_id = pd_frame_id.numpy()
    pd_score = pd_score.numpy()
    sample_tokens = list(set(pd_frame_id))
    all_annotations = EvalBoxes()
    # Load annotations and filter predictions and annotations.

    for sample_token in tqdm.tqdm(sample_tokens):
        index = pd_frame_id == sample_token
        pd_bbox_ = pd_bbox[index]
        pd_type_ = pd_type[index]
        pd_score_ = pd_score[index]
        sample_boxes = []
        for bbox, type_, score in zip(pd_bbox_, pd_type_, pd_score_):
            sample_boxes.append(
                    DetectionBox_modified(
                        token='',
                        sample_token=str(sample_token),
                        translation=bbox[:3],
                        size=bbox[3:6],
                        rotation=pyquaternion.Quaternion(axis=[0, 0, 1], radians=bbox[-1]).elements,
                        velocity=[0, 0],
                        num_pts=100,
                        detection_name=class_maps[type_],
                        detection_score=float(score),  # GT samples do not have a score.
                        attribute_name='',
                    )
                )

        all_annotations.add_boxes(str(sample_token), sample_boxes)
    return all_annotations


def eval_waymo_with_NDS(pd_file, gt_file):

    from projects.mmdet3d_plugin.datasets.nuscnes_eval import DetectionBox_modified
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
        DetectionMetricDataList
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.common.utils import center_distance




    gt_bbox, gt_type, gt_frame_id, _, difficulty = get_boxes_from_bin(gt_file, remove_gt=True)
    pd_bbox, pd_type, pd_frame_id, pd_score, _ = get_boxes_from_bin(pd_file, remove_pred=False)

    #
    # timestamps = [1522688014970187]
    #
    # for i, timestamp in enumerate(timestamps):
    #
    #     index = gt_frame_id.numpy() == timestamp
    #     gt_bbox_ = gt_bbox[index]
    #     gt_type_ = gt_type[index]
    #     gt_frame_id_ = gt_frame_id[index]
    #     difficulty_ = difficulty[index]
    #
    #     index = gt_type_.numpy() == 1
    #     gt_bbox = gt_bbox_[index]
    #     gt_type = gt_type_[index]
    #     gt_frame_id = gt_frame_id_[index]
    #     difficulty = difficulty_[index]
    #
    #     pd_index = pd_frame_id.numpy() == timestamp
    #     pd_bbox_ = pd_bbox[pd_index]
    #     pd_type_ = pd_type[pd_index]
    #     pd_frame_id_ = pd_frame_id[pd_index]
    #     pd_score_ = pd_score[pd_index]
    #
    #     pd_index = pd_type_.numpy() == 1
    #     pd_bbox = pd_bbox_[pd_index]
    #     pd_type = pd_type_[pd_index]
    #     pd_frame_id = pd_frame_id_[pd_index]
    #     pd_score = pd_score_[pd_index]


    pd = load_pd(pd_bbox, pd_type, pd_frame_id, pd_score)
    gt = load_gt(gt_bbox, gt_type, gt_frame_id)

    metric_data_list = DetectionMetricDataList()

    # print(self.cfg.dist_fcn_callable, self.cfg.dist_ths)
    # self.cfg.dist_ths = [0.3]
    # self.cfg.dist_fcn_callable
    from nuscenes.eval.detection.config import config_factory
    cfg = config_factory('detection_cvpr_2019')
    cfg.class_names = ['car', 'pedestrian', 'bicycle']
    cfg.class_range = {
        'bicycle': 75,
        'pedestrian': 75,
        'car': 75
    }

    for class_name in cfg.class_names:
        for dist_th in [0.5, 1, 2, 4]:
            md = accumulate(gt, pd, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)

    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------

    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        # Compute APs.
        for dist_th in [0.5, 1, 2, 4]:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    # Compute evaluation time.
    metrics.add_runtime(0)
    print(metric_data_list, metrics)
    return metrics, metric_data_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--bin", help='path to bin', default=None)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "11"
    # pd_file = 'Tue_Jan_25_01_28_40_2022.bin'
    # pd_file = 'test/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/Sat_Feb__5_16_50_00_2022.bin'
    # pd_file = 'test/detr3d_res101_gridmask_waymo/Wed_Feb__9_17_49_42_2022.bin'
    # pd_file = 'test/four/Wed_Feb__9_21_08_32_2022.bin'
    # pd_file = 'test/waymo_imp/Mon_Feb_14_14_09_34_2022.bin'
    # pd_file = 'test/waymo_imp/Mon_Feb_14_19_08_21_2022.bin'
    # pd_file = 'test/waymo_imp/Tue_Feb_15_17_22_36_2022.bin'  # front
    if args.bin:
        pd_file = args.bin
    else:
        pd_file = 'test/waymo_imp/Tue_Feb_15_15_34_23_2022.bin'  # all

    #pd_file = 'test/waymo_imp2/Tue_Mar_22_19_29_25_2022.bin'
    #pd_file = 'test/waymo_imp2/Tue_Mar_22_21_59_04_2022.bin'

    #pd_file = 'test/nusc2waymo/Fri_Feb_18_20_49_42_2022.bin'
    #pd_file = 'test/nusc2waymo/Wed_Feb_23_18_27_20_2022.bin' # zero-shot
    #pd_file = 'test/waymo_imp_static/Mon_Mar__7_20_19_19_2022.bin'  # static
    #pd_file = 'test/nusc2waymo/Fri_Mar_11_23_33_04_2022.bin'
    gt_file = 'data/waymo/waymo_format/gt_synced.bin'
    eval(pd_file, gt_file)
    exit()
    #
    # metrics, metric_data_list = eval_waymo_with_NDS(pd_file, gt_file)
    # metrics_summary = metrics.serialize()
    #
    # # Print high-level metrics.
    # print('mAP: %.4f' % (metrics_summary['mean_ap']))
    # err_name_mapping = {
    #     'trans_err': 'mATE',
    #     'scale_err': 'mASE',
    #     'orient_err': 'mAOE',
    #     'vel_err': 'mAVE',
    #     'attr_err': 'mAAE'
    # }
    # for tp_name, tp_val in metrics_summary['tp_errors'].items():
    #     print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
    # print('NDS: %.4f' % (metrics_summary['nd_score']))
    # print('Eval time: %.1fs' % metrics_summary['eval_time'])
    #
    # # Print per-class metrics.
    # print()
    # print('Per-class results:')
    # print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
    # class_aps = metrics_summary['mean_dist_aps']
    # class_tps = metrics_summary['label_tp_errors']
    # for class_name in class_aps.keys():
    #     print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
    #           % (class_name, class_aps[class_name],
    #              class_tps[class_name]['trans_err'],
    #              class_tps[class_name]['scale_err'],
    #              class_tps[class_name]['orient_err'],
    #              class_tps[class_name]['vel_err'],
    #              class_tps[class_name]['attr_err']))
    # print(metrics_summary)
    #
    # # eval(pd_file, gt_file)
    # # pd_file = '/home/yang_ye/SA-SSD_Waymo/tools/result_val'
    # # gt_file = '/data/yang_ye/Waymo/val/Label'
    #
    #
