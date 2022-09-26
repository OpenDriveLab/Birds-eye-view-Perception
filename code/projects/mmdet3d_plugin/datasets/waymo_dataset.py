# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# waymo_dataset.py       OpenPerceptionX        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
#
# See:
# https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/waymo_dataset.py
#
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

import time

import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from .kitti_dataset import CustomKittiDataset
# from pyquaternion import Quaternion
from projects.mmdet3d_plugin.models.utils.draw_bbox import show_multi_modality_result
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, LiDARInstance3DBoxes, points_cam2img)
import copy
import cv2
import numpy as np
import torch
from glob import glob
from os.path import join
from mmcv.parallel import DataContainer as DC
import random
from .pipelines.compose import CustomCompose
try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
    import mmcv
    import numpy as np
    import tensorflow as tf
    from glob import glob
    from os.path import join
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

except ImportError:
    print('WARNING check waymo_dataset')


@DATASETS.register_module()
class CustomWaymoDataset(CustomKittiDataset):
    """Waymo Dataset.
    This class serves as the API for experiments on the Waymo Dataset.
    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes
            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(
            self,
            data_root,
            ann_file,
            split,
            pts_prefix='velodyne',
            classes=None,
            modality=None,
            box_type_3d='LiDAR',
            filter_empty_gt=True,
            test_mode=False,
            pipeline=None,
            load_interval=1,
            finetune_key=None,
            pcd_limit_range=[-85, -85, -5, 85, 85, 5],
            nusc=None,  # For compatibility with previous codes
            nusc_can_bis=None,
            **kwargs):
        classes = ['Car', 'Pedestrian', 'Cyclist']
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,  # use my custom compose
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        if finetune_key is not None:
            self.data_infos = self.finetune_based_on(finetune_key)
            self.flag = self.flag[:len(self.data_infos)]
        if 'waymo_infos_val.pkl' in ann_file or 'waymo_infos_test.pkl' in ann_file or 'waymo_infos_test_camera.pkl' in ann_file:
            # to load a subset, just set the load_interval in the dataset config
            data_infos = self.data_infos
            s1, s2, s3, s4, s5 = data_infos[::5], data_infos[1::5], data_infos[2::5], data_infos[3::5], data_infos[4::5]
            data_infos = s1 + s2 + s3 + s4 + s5
            self.data_infos = data_infos
        else:
            if load_interval == 5:
                if len(self.data_infos) <= 40000:
                    load_interval = 1
                self.data_infos = self.data_infos[::load_interval]
            elif load_interval == 1:  # use all data
                data_infos = self.data_infos
                s1, s2, s3, s4, s5 = data_infos[::5], data_infos[1::5], data_infos[2::5], data_infos[3::5], data_infos[
                    4::5]
                data_infos = s1 + s2 + s3 + s4 + s5
                self.data_infos = data_infos

        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

        # self.pipeline = CustomCompose(pipeline)
        self.load_interval = load_interval
        self.nusc = 'Useless'
        self.nusc_can_bus = 'Useless'

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix, f'{idx:07d}.bin')
        return pts_filename

    def finetune_based_on(self, key):
        import os.path as osp
        assert osp.isfile('data/waymo/kitti_format/waymo_scenes_info.json')
        scenes_infos = mmcv.load('data/waymo/kitti_format/waymo_scenes_info.json')
        legal_ids = []
        if key in ['Day', 'Night', 'Dawn/Dusk']:
            for id_, scene in scenes_infos.items():
                if scene['time_of_day'] == key:
                    legal_ids.append(id_)
        elif key in ['Car', 'Cyclist', 'Pedestrian']:
            k2w_cls_map = {
                'Car': 1,
                'Pedestrian': 2,
                'Cyclist': 4,
            }

            key_id = k2w_cls_map[key]
            num_of_key = np.array([value['counter'][key_id] for value in scenes_infos.values()])
            sorted_index = np.argsort(-num_of_key)[:300]
            legal_ids = np.array(list(scenes_infos.keys()))[sorted_index]
        new_data_infos = []
        for each in self.data_infos:
            if each['image']['image_path'].split('/')[-1][:4] in legal_ids:
                new_data_infos.append(each)
        return new_data_infos

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
            image_filenames.append(img_filename_0.replace('image_0', f'image_{i}'))

        calib_file = img_filename_0.replace('image_0', 'calib').replace('png', 'txt').replace('jpg', 'txt')
        calibs = mmcv.load(calib_file)

        calibs = [each.strip() for each in calibs]

        # TODO: consider use torch.Tensor only
        lidar2imgs = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for i in range(5):
            rect = info['calib']['R0_rect'].astype(np.float32)
            # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
            Trv2c = np.zeros([4, 4])
            Trv2c[-1, -1] = 1.
            Trv2c[:3, :4] = np.array(calibs[i + 6].split(' ')[1:]).reshape([3, 4])
            Pi = info['calib'][f'P{i}'].astype(np.float32)
            lidar2img = Pi @ rect @ Trv2c
            lidar2imgs.append(lidar2img)
            lidar2cam_rts.append(rect @ Trv2c)
            cam_intrinsics.append(Pi)

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
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
            dict(scene_token=sample_idx // 1000,
                 prev_idx=index - self.load_interval,
                 next_idx=index + self.load_interval,
                 can_bus=can_bus))

        return input_dict

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None, data_format='waymo'):
        """Format the results to pkl file.
        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            data_format (str | None): Output data format. Default: 'waymo'.
                Another supported choice is 'kitti'.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """

        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert ('waymo' in data_format or 'kitti' in data_format), \
            f'invalid data_format {data_format}'

        if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[0]:
            raise TypeError('Not supported type for reformat results.')
        elif 'pts_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES, pklfile_prefix_, submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES, pklfile_prefix, submission_prefix)
        if 'waymo' in data_format:
            from projects.mmdet3d_plugin.core.evaluation.kitti2waymo import KITTI2Waymo
            # from mmdet3d.core.evaluation.waymo_utils.prediction_kitti_to_waymo import KITTI2Waymo  # noqa
            waymo_root = osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')
            if self.split == 'training':
                waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
                prefix = '1'
            elif self.split == 'testing':
                waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
                prefix = '2'
            elif self.split == 'testing_camera':
                waymo_tfrecords_dir = osp.join(waymo_root, 'testing_camera')
                prefix = '3'
            else:
                raise ValueError('Not supported split value.')
            save_tmp_dir = tempfile.TemporaryDirectory()
            waymo_results_save_dir = save_tmp_dir.name
            waymo_results_final_path = f'{pklfile_prefix}.bin'
            if 'pts_bbox' in result_files:
                converter = KITTI2Waymo(result_files['pts_bbox'], waymo_tfrecords_dir, waymo_results_save_dir,
                                        waymo_results_final_path, prefix)
            else:
                converter = KITTI2Waymo(result_files, waymo_tfrecords_dir, waymo_results_save_dir,
                                        waymo_results_final_path, prefix)
            converter.convert()
            save_tmp_dir.cleanup()

        return result_files, tmp_dir

    def parse_objects(self, result, context_name, frame_timestamp_micros):
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
            width = round(result['box3d_lidar'][instance_idx, 3], 4)
            length = round(result['box3d_lidar'][instance_idx, 4], 4)
            height = round(result['box3d_lidar'][instance_idx, 5], 4)
            x = round(result['box3d_lidar'][instance_idx, 0], 4)
            y = round(result['box3d_lidar'][instance_idx, 1], 4)
            z = round(result['box3d_lidar'][instance_idx, 2], 4)
            z += height / 2
            rotation_y = round(result['box3d_lidar'][instance_idx, 6], 4)
            score = round(result['scores'][instance_idx], 4)

            # y: downwards; move box origin from bottom center (kitti) to
            # true center (waymo)
            # y -= height / 2
            # frame transformation: kitti -> waymo
            # x, y, z = self.transform(T_k2w, x, y, z)

            # different conventions

            heading = -(rotation_y + np.pi / 2)

            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

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
                objects = self.parse_objects(result[frame_num], context_name, frame_timestamp_micros)
            else:
                objects = metrics_pb2.Objects()
                print(filename, 'not found.')

            with open(join(self.waymo_results_save_dir, f'{filename}.bin'), 'wb') as f:
                f.write(objects.SerializeToString())

    def get_file_names(self):
        """Get file names of waymo raw data."""
        self.waymo_tfrecord_pathnames = sorted(glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
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

    def get_front_results(self, outputs):

        for idx in range(len(outputs)):
            output = outputs[idx]['pts_bbox']
            box_preds, scores_3d, labels_3d = output['boxes_3d'], output['scores_3d'], output['labels_3d']

            box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_inds = ((box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:])).all(-1)
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            img_filename_0 = os.path.join(self.data_root, info['image']['image_path'])

            valid_font_ind = (box_preds.tensor[:, 0] < 0) | (
                (box_preds.tensor[:, 1] / box_preds.tensor[:, 0]).abs() > 0.47)
            valid_cam_inds = valid_font_ind & valid_inds
            outputs[idx]['pts_bbox']['scores_3d'][valid_cam_inds.nonzero().squeeze(-1)] = 0.
        return outputs

    def format_waymo_results(self, outputs, pklfile_prefix=None, submission_prefix=None):

        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        waymo_root = osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')
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
        self.waymo_results_final_path = f'{pklfile_prefix}.bin'
        self.get_file_names()

        file_idx_list = []
        results_maps = {}

        for idx in range(len(outputs)):

            output = outputs[idx]['pts_bbox']
            box_preds, scores_3d, labels_3d = output['boxes_3d'], output['scores_3d'], output['labels_3d']

            box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_inds = ((box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:])).all(-1)
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            img_filename_0 = os.path.join(self.data_root, info['image']['image_path'])
            image_filenames = []
            for i in range(5):
                image_filenames.append(img_filename_0.replace('image_0', f'image_{i}'))
            calib_file = img_filename_0.replace('image_0', 'calib').replace('png', 'txt').replace('jpg', 'txt')

            calibs = mmcv.load(calib_file)
            calibs = [each.strip() for each in calibs]

            img_shapes = [(1280, 1920, 3), (1280, 1920, 3), (1280, 1920, 3), (886, 1920, 3), (886, 1920, 3)]

            # cam_valid_list = []
            # for i in range(5):
            #     rect = info['calib']['R0_rect'].astype(np.float32)
            #     # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
            #     Trv2c = np.zeros([4, 4])
            #     Trv2c[-1, -1] = 1.
            #     Trv2c[:3, :4] = np.array(calibs[i + 6].split(' ')[1:]).reshape([3, 4])
            #     # print(Trv2c)
            #     Pi = info['calib'][f'P{i}'].astype(np.float32)
            #     img_shape = img_shapes[i]
            #     Pi = box_preds.tensor.new_tensor(Pi)
            #     box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)
            #     box_corners = box_preds_camera.corners
            #     box_corners_in_image = points_cam2img(box_corners, Pi)
            #     # box_corners_in_image: [N, 8, 2]
            #     minxy = torch.min(box_corners_in_image, dim=1)[0]
            #     maxxy = torch.max(box_corners_in_image, dim=1)[0]
            #     box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            #     # Post-processing
            #     # check box_preds_camera
            #     image_shape = box_preds.tensor.new_tensor(img_shape)
            #
            #     valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
            #                       (box_2d_preds[:, 1] < image_shape[0]) &
            #                       (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
            #     cam_valid_list.append(valid_cam_inds)
            # valid_cam_inds = cam_valid_list[0] | cam_valid_list[1] | cam_valid_list[2] | cam_valid_list[3] | \
            #                  cam_valid_list[4]
            # valid_inds =  & valid_inds
            if valid_inds.sum() > 0:
                result = dict(bbox=None,
                              box3d_camera=None,
                              box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                              scores=scores_3d[valid_inds].numpy(),
                              label_preds=labels_3d[valid_inds].numpy(),
                              sample_idx=sample_idx)
            else:
                result = dict(bbox=np.zeros([0, 4]),
                              box3d_camera=np.zeros([0, 7]),
                              box3d_lidar=np.zeros([0, 7]),
                              scores=np.zeros([0]),
                              label_preds=np.zeros([0, 4]),
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
        mmcv.track_parallel_progress(self.convert_one, range(len(self.file_idx_list)), 32)
        print('\nFinished ...')

        pathnames = sorted(glob(join(self.waymo_results_save_dir, '*.bin')))
        combined = self.combine(pathnames)
        mmcv.mkdir_or_exist(pklfile_prefix)
        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

        return self.waymo_results_final_path, tmp_dir

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

        metric = {'waymo'}

        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'

        show = True
        if show:
            if out_dir is None:
                out_dir = pklfile_prefix
                # results = self.get_front_results(results)
            self.show(results, out_dir, pipeline=pipeline, max_show=100)

        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(results, pklfile_prefix, submission_prefix, data_format='kitti')
            from mmdet3d.core.evaluation import kitti_eval
            gt_annos = [info['annos'] for info in self.data_infos]

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval(gt_annos, result_files_, self.CLASSES, eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                    print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval(gt_annos, result_files, self.CLASSES, eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
        if 'waymo' in metric:
            waymo_root = osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
                # format_results(
            result_files, tmp_dir = self.format_waymo_results(
                results,
                pklfile_prefix,
                submission_prefix,
            )
            if tmp_dir is not None:
                tmp_dir.cleanup()
            import subprocess
            print(result_files)
            return {}
            from .eval_waymo import eval
            waymo_results = eval(result_files)
            #except:
            #    pass
            return waymo_results
            try:
                ret_bytes = subprocess.check_output('mmdet3d/core/evaluation/waymo_utils/' +
                                                    f'compute_detection_metrics_main {pklfile_prefix}.bin ' +
                                                    f'{waymo_root}/gt.bin',
                                                    shell=True)
                ret_texts = ret_bytes.decode('utf-8')
                print_log(ret_texts)
                # parse the text to get ap_dict
                ap_dict = {
                    'Vehicle/L1 mAP': 0,
                    'Vehicle/L1 mAPH': 0,
                    'Vehicle/L2 mAP': 0,
                    'Vehicle/L2 mAPH': 0,
                    'Pedestrian/L1 mAP': 0,
                    'Pedestrian/L1 mAPH': 0,
                    'Pedestrian/L2 mAP': 0,
                    'Pedestrian/L2 mAPH': 0,
                    'Sign/L1 mAP': 0,
                    'Sign/L1 mAPH': 0,
                    'Sign/L2 mAP': 0,
                    'Sign/L2 mAPH': 0,
                    'Cyclist/L1 mAP': 0,
                    'Cyclist/L1 mAPH': 0,
                    'Cyclist/L2 mAP': 0,
                    'Cyclist/L2 mAPH': 0,
                    'Overall/L1 mAP': 0,
                    'Overall/L1 mAPH': 0,
                    'Overall/L2 mAP': 0,
                    'Overall/L2 mAPH': 0
                }
                mAP_splits = ret_texts.split('mAP ')
                mAPH_splits = ret_texts.split('mAPH ')
                for idx, key in enumerate(ap_dict.keys()):
                    split_idx = int(idx / 2) + 1
                    if idx % 2 == 0:  # mAP
                        ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                    else:  # mAPH
                        ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
                ap_dict['Overall/L1 mAP'] = \
                    (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
                     ap_dict['Cyclist/L1 mAP']) / 3
                ap_dict['Overall/L1 mAPH'] = \
                    (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                     ap_dict['Cyclist/L1 mAPH']) / 3
                ap_dict['Overall/L2 mAP'] = \
                    (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                     ap_dict['Cyclist/L2 mAP']) / 3
                ap_dict['Overall/L2 mAPH'] = \
                    (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                     ap_dict['Cyclist/L2 mAPH']) / 3
                if eval_tmp_dir is not None:
                    eval_tmp_dir.cleanup()

                if tmp_dir is not None:
                    tmp_dir.cleanup()
            except:
                print('ERROR')

        return {}

    def bbox2result_kitti(self, net_outputs, class_names, pklfile_prefix=None, submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.
        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.
        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]

            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, bbox, score, label in zip(box_preds, box_preds_lidar, box_2d_preds, scores,
                                                              label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print('{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                  '{:.4f} {:.4f} {:.4f} '
                                  '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                      anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                      bbox[idx][3], dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1],
                                      loc[idx][2], anno['rotation_y'][idx], anno['score'][idx]),
                                  file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            annos[-1]['sample_idx'] = np.array([sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')
        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the boxes into valid format.
        Args:
            box_dict (dict): Bounding boxes to be converted.
                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.
        Returns:
            dict: Valid boxes after conversion.
                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(bbox=np.zeros([0, 4]),
                        box3d_camera=np.zeros([0, 7]),
                        box3d_lidar=np.zeros([0, 7]),
                        scores=np.zeros([0]),
                        label_preds=np.zeros([0, 4]),
                        sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        return dict(
            bbox=box_2d_preds[:, :].numpy(),
            box3d_camera=box_preds_camera[:].tensor.numpy(),
            box3d_lidar=box_preds[:].tensor.numpy(),
            scores=scores[:].numpy(),
            label_preds=labels[:].numpy(),
            sample_idx=sample_idx,
        )

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

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

        label_root = osp.join(self.data_root, 'training')
        label_all = []
        custom_label_map = {
            'name': [],  # 1
            'truncated': [],  # 1
            'alpha': [],  # 1
            'bbox': [],  # 4
            'dimensions': [],  # 3
            'location': [],  # 3
            'rotation_y': [],  # 1
            'index': [],
            'group_ids': [],
            'camera_id': [],
            'difficulty': [],
            'num_points_in_gt': [],
        }
        img_idx = info['image']['image_idx']
        lable_i_root = osp.join(label_root, f'label_all', str(img_idx).zfill(7) + '.txt')

        for line in mmcv.load(lable_i_root):
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

        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d,
                            gt_labels_3d=gt_labels_3d,
                            bboxes=gt_bboxes,
                            labels=gt_labels,
                            gt_names=gt_names)
        return anns_results

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
        pipeline = self._get_pipeline(pipeline)
        count = 0
        for i, result in enumerate(results):
            if i % 100 != 0:
                continue
            count += 1
            if count > max_show: break
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            img_metas, _ = self._extract_data(i, pipeline, ['img_metas', 'img'])
            try:
                gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
                show_gt_bboxes = LiDARInstance3DBoxes(gt_bboxes, origin=(0.5, 0.5, 0))
            except:
                show_gt_bboxes = None

            pred_bboxes = result['boxes_3d'].tensor.numpy()

            scores = np.array(result['scores_3d'].cpu().numpy())

            show_pred_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

            img_filename_0 = data_info['image']['image_path']
            image_filenames = []
            for j in range(5):
                image_filenames.append(img_filename_0.replace('image_0', f'image_{j}'))

            for j in range(5):
                show_multi_modality_result(
                    mmcv.imread(os.path.join(self.data_root, image_filenames[j])),
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['ori_lidar2img'][j],
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
            if show_gt_bboxes is not None:
                for corners in show_gt_bboxes.corners[:, [4, 7, 3, 0], :2]:
                    corners = np.array([world2bev_vis(*corner) for corner in corners])
                    _img = np.zeros([1500, 1100, 3], dtype=np.float32)
                    _img = cv2.fillPoly(_img, [corners], (61, 102, 255))
                    bev_img = cv2.addWeighted(bev_img, 1, _img, 0.5, 0)

            bev_img = cv2.circle(bev_img, world2bev_vis(0, 0), 5, (0, 255, 0), thickness=-1)

            bev_out_dir = osp.join(out_dir, img_filename_0.replace('image_0', f'bev'))
            mmcv.imwrite(bev_img, bev_out_dir)


@DATASETS.register_module()
class WaymoDataset_video(CustomWaymoDataset):

    def __init__(self, queue_length=None, pipeline=None, *args, **kwargs):
        super(WaymoDataset_video, self).__init__(pipeline=[], *args, **kwargs)
        self.queue_length = queue_length
        self.pipeline = CustomCompose(pipeline)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations.
        """
        data = mmcv.load(ann_file)
        return data

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index - self.queue_length, index + 1))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        seed = time.time()
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict, seed=seed)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        if 'points' in queue[0].keys():
            points_list = [each['points'].data for each in queue]
            queue[-1]['points'] = DC(points_list, cpu_only=False, stack=False)
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
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
