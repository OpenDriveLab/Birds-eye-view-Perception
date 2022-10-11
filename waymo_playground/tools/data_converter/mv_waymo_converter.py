# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    # from waymo_open_dataset import dataset_pb2
    from simple_waymo_open_dataset_reader import WaymoDataFileReader
    from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')
from pyquaternion import Quaternion
import mmcv
import numpy as np
import torch as tf
from glob import glob
from os.path import join
#from waymo_open_dataset.utils import range_image_utils, transform_utils
#from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from pyquaternion import quaternion

class Waymo2Nucenes(object):
    """Waymo to KITTI converter.

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (str): Number of workers for the parallel process.
        test_mode (bool): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False

        # turn on eager execution for older tensorflow versions
        # if int(tf.__version__.split('.')[0]) < 2:
        #    tf.enable_eager_execution()

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
            '_SIDE_LEFT'
        ]
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }
        self.cam_name_list = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        #self.tfrecord_pathnames = sorted(
         #   glob(join(self.load_dir, '*.tfrecord')))
        self.tfrecord_pathnames = sorted(glob(join(self.load_dir, '*.tfrecord')))
        #self.label_save_dir = f'{self.save_dir}/label_'
        #self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        #self.calib_save_dir = f'{self.save_dir}/calib'
        #self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        #self.pose_save_dir = f'{self.save_dir}/pose'

        #self.create_folder()
    def convert(self):
        """Convert action."""
        print('Start converting ...')

        train_infos = []
        for file_idx in range(len(self)):
            pathname = self.tfrecord_pathnames[file_idx]
            dataset = WaymoDataFileReader(pathname)

            for frame_idx, frame in enumerate(dataset):
                if frame_idx % 5 != 0:
                    continue
                laser_calib = np.array(frame.context.laser_calibrations[4].extrinsic.transform).reshape(4, 4)
                assert frame.context.laser_calibrations[4].name == 1
                ego_pose = np.array(frame.pose.transform).reshape(4, 4)
                l2e_t = laser_calib[:3, -1]
                #l2e_r = laser_calib[:3, :3]

                l2e_r = np.array([
                    [1., 0, 0],
                    [0, 1., 0],
                    [0, 0, 1.]
                ])
                e2g_t = ego_pose[:3, -1]
                e2g_r = ego_pose[:3, :3]
                if (self.selected_waymo_locations is not None
                        and frame.context.stats.location
                        not in self.selected_waymo_locations):
                    continue
                info = {
                        'frame_token': pathname,
                        'lidar_path': '',
                        'token': '',
                        'sweeps': [],
                        'cams': dict(),
                        'lidar2ego_translation': l2e_t,
                        'lidar2ego_rotation': quaternion.Quaternion(matrix=l2e_r).q,
                        'ego2global_translation': e2g_t,
                        'ego2global_rotation': quaternion.Quaternion(matrix=e2g_r).q,
                        'timestamp': str(frame.timestamp_micros),
                        'gt_boxes': [],
                        'gt_names': [],
                        'gt_velocity': [],
                        'num_lidar_pts': [],
                        'num_radar_pts': [],
                        'valid_flag': []
                    }

                camera_types = {
                        1: 'FRONT',
                        2: 'FRONT_LEFT',
                        3: 'FRONT_RIGHT',
                        4: 'SIDE_LEFT',
                        5: 'SIDE_RIGHT',
                }
                images = [image for image in frame.images]
                cameras = [camera for camera in frame.context.camera_calibrations]
                images[2], images[3] = images[3], images[2]
                image_names1 = [image.name for image in images]
                image_names2 = [camera.name for camera in cameras]
                assert image_names2 == image_names1
                for i, (img, camera) in enumerate(zip(images, cameras)):

                    img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                                   f'{self.prefix}{str(file_idx).zfill(3)}' + \
                                   f'{str(frame_idx).zfill(3)}.png'

                    img_data = mmcv.imfrombytes(img.image)
                    mmcv.imwrite(img_data, img_path)
                    T_cam_to_ego = np.array(camera.extrinsic.transform).reshape(4, 4)
                    sensor2ego_translation = T_cam_to_ego[:3, -1]
                    sensor2ego_rotation = T_cam_to_ego[:3, :3]
                    R = (sensor2ego_rotation.T @ e2g_r.T) @ (
                            np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T)
                    T = (sensor2ego_translation @ e2g_r.T + e2g_t) @ (
                            np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T)
                    T -= e2g_t @ (np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T
                                  ) + l2e_t @ np.linalg.inv(l2e_r).T
                    camera_calib = np.zeros((3, 3))
                    camera_calib[0, 0] = camera.intrinsic[0]
                    camera_calib[1, 1] = camera.intrinsic[1]
                    camera_calib[0, 2] = camera.intrinsic[2]
                    camera_calib[1, 2] = camera.intrinsic[3]
                    camera_calib[2, 2] = 1

                    cam_info = {
                        'data_path': img_path,
                        'sensor2lidar_rotation': quaternion.Quaternion(matrix=R.T).q,
                        'sensor2lidar_translation': T,
                        'sensor2ego_translation': sensor2ego_translation,
                        'sensor2ego_rotation':  quaternion.Quaternion(matrix=sensor2ego_rotation).q,
                        'ego2global_translation': e2g_t,
                        'ego2global_rotation': quaternion.Quaternion(matrix=e2g_r).q,
                        'cam_intrinsic': camera_calib,
                        'type': camera_types[img.name],
                        'sample_data_token': str(frame.images[0].camera_readout_done_time),
                        'timestamp': str(frame.images[0].camera_readout_done_time),
                    }
                    info['cams'].update({camera_types[img.name]: cam_info})

                for obj in frame.laser_labels:
                    height = obj.box.height
                    width = obj.box.width
                    length = obj.box.length
                    x = obj.box.center_x
                    y = obj.box.center_y
                    z = obj.box.center_z - height / 2
                    rotation_y = -obj.box.heading - np.pi / 2
                    v_x, v_y = obj.metadata.speed_x, obj.metadata.speed_y
                    type_ = obj.type
                    info['gt_boxes'].append([x, y, z,  width, length, height, rotation_y])
                    info['gt_names'].append(type_)
                    info['gt_velocity'].append([v_x, v_y])

                train_infos.append(info)
        data = dict(infos=train_infos, metadata={})
        mmcv.dump(data, 'train.pkl')
        return 'train.pkl'

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)


    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}')


    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret
