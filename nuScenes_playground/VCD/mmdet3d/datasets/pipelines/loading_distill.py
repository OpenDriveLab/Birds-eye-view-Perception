# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
try:
    from mmdet3d.utils import CephClient
except:
    print('no ceph')

from mmdet3d.datasets.pipelines.loading import mmlabNormalize


@PIPELINES.register_module()
class PrepareImageInputs_Distill(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config_S,
        data_config_T,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',
        file_client_args=None,
    ):
        self.is_train = is_train
        self.data_config_S = data_config_S
        self.data_config_T = data_config_T
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        #self.client = CephClient(file_client_args, file_client_args['path_mapping'])
        if file_client_args['backend'] == 'petrel':
            self.client = CephClient(file_client_args, file_client_args['path_mapping'])
        elif file_client_args['backend'] == 'disk':
            self.client = None
        #self.client = None

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config_S['Ncams'] < len(
                self.data_config_S['cams']):
            cam_names = np.random.choice(
                self.data_config_S['cams'],
                self.data_config_S['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config_S['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH_S, fW_S = self.data_config_S['input_size']
        fH_T, fW_T = self.data_config_T['input_size']
        if self.is_train:
            # student
            resize_S = float(fW_S) / float(W)
            resize_S += np.random.uniform(*self.data_config_S['resize'])
            resize_dims_S = (int(W * resize_S), int(H * resize_S))
            newW_S, newH_S = resize_dims_S
            crop_h_S = int((1 - np.random.uniform(*self.data_config_S['crop_h'])) *
                         newH_S) - fH_S
            crop_w_S = int(np.random.uniform(0, max(0, newW_S - fW_S)))
            crop_S = (crop_w_S, crop_h_S, crop_w_S + fW_S, crop_h_S + fH_S)

            flip = self.data_config_S['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config_S['rot'])

            # teacher
            resize_T = float(fW_T) / float(W)
            resize_T += np.random.uniform(*self.data_config_T['resize'])
            resize_dims_T = (int(W * resize_T), int(H * resize_T))
            newW_T, newH_T = resize_dims_T
            crop_h_T = int((1 - np.random.uniform(*self.data_config_T['crop_h'])) *
                         newH_T) - fH_T
            crop_w_T = int(np.random.uniform(0, max(0, newW_T - fW_T)))
            crop_T = (crop_w_T, crop_h_T, crop_w_T + fW_T, crop_h_T + fH_T)

        else:
            # student
            resize_S = float(fW_S) / float(W)
            resize_S += self.data_config_S.get('resize_test', 0.0)
            if scale is not None:
                resize_S = scale
            resize_dims_S = (int(W * resize_S), int(H * resize_S))
            newW_S, newH_S = resize_dims_S
            crop_h_S = int((1 - np.mean(self.data_config_S['crop_h'])) * newH_S) - fH_S
            crop_w_S = int(max(0, newW_S - fW_S) / 2)
            crop_S = (crop_w_S, crop_h_S, crop_w_S + fW_S, crop_h_S + fH_S)
            flip = False if flip is None else flip
            rotate = 0

            # teacher
            resize_T = float(fW_T) / float(W)
            resize_T += self.data_config_T.get('resize_test', 0.0)
            if scale is not None:
                resize_T = scale
            resize_dims_T = (int(W * resize_T), int(H * resize_T))
            newW_T, newH_T = resize_dims_T
            crop_h_T = int((1 - np.mean(self.data_config_T['crop_h'])) * newH_T) - fH_T
            crop_w_T = int(max(0, newW_T - fW_T) / 2)
            crop_T = (crop_w_T, crop_h_T, crop_w_T + fW_T, crop_h_T + fH_T)
        
        return resize_S, resize_T, resize_dims_S, resize_dims_T, crop_S, crop_T, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][cam_name]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor

    def get_inputs(self, results, flip=None, scale=None):
        rots = []
        trans = []
        intrins = []

        imgs_S = []
        post_rots_S = []
        post_trans_S = []
        canvas_S = []
        imgs_T = []
        post_rots_T = []
        post_trans_T = []
        canvas_T = []

        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        sensor2sensors = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            #img = Image.fromarray(self.client.get(filename))
            if self.client is not None:
                img = Image.fromarray(self.client.get(filename))
            else:
                img = Image.open(filename)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize_S, resize_T, resize_dims_S, resize_dims_T, crop_S, crop_T, flip, rotate = img_augs
            
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            img_S, post_rot2_S, post_tran2_S = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize_S,
                                   resize_dims=resize_dims_S,
                                   crop=crop_S,
                                   flip=flip,
                                   rotate=rotate)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            img_T, post_rot2_T, post_tran2_T = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize_T,
                                   resize_dims=resize_dims_T,
                                   crop=crop_T,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran_S = torch.zeros(3)
            post_rot_S = torch.eye(3)
            post_tran_S[:2] = post_tran2_S
            post_rot_S[:2, :2] = post_rot2_S
            # for convenience, make augmentation matrices 3x3
            post_tran_T = torch.zeros(3)
            post_rot_T = torch.eye(3)
            post_tran_T[:2] = post_tran2_T
            post_rot_T[:2, :2] = post_rot2_T

            canvas_S.append(np.array(img_S))
            imgs_S.append(self.normalize_img(img_S))
            canvas_T.append(np.array(img_T))
            imgs_T.append(self.normalize_img(img_T))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    #img_adjacent = Image.fromarray(self.client.get(filename_adj))
                    if self.client is not None:
                        img_adjacent = Image.fromarray(self.client.get(filename_adj))
                    else:
                        img_adjacent = Image.open(filename_adj)
                    img_adjacent_S = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims_S,
                        crop=crop_S,
                        flip=flip,
                        rotate=rotate)
                    img_adjacent_T = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims_T,
                        crop=crop_T,
                        flip=flip,
                        rotate=rotate)
                    imgs_S.append(self.normalize_img(img_adjacent_S))
                    imgs_T.append(self.normalize_img(img_adjacent_T))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots_S.append(post_rot_S)
            post_trans_S.append(post_tran_S)
            post_rots_T.append(post_rot_T)
            post_trans_T.append(post_tran_T)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans_S.extend(post_trans_S[:len(cam_names)])
                post_rots_S.extend(post_rots_S[:len(cam_names)])
                post_trans_T.extend(post_trans_T[:len(cam_names)])
                post_rots_T.extend(post_rots_T[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)
        imgs_S = torch.stack(imgs_S)
        imgs_T = torch.stack(imgs_T)

        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots_S = torch.stack(post_rots_S)
        post_trans_S = torch.stack(post_trans_S)
        post_rots_T = torch.stack(post_rots_T)
        post_trans_T = torch.stack(post_trans_T)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas_S'] = canvas_S
        results['canvas_T'] = canvas_T
        results['sensor2sensors'] = sensor2sensors

        inputs_S = (imgs_S, rots, trans, intrins, post_rots_S, post_trans_S, sensor2sensors)
        inputs_T = (imgs_T, rots, trans, intrins, post_rots_T, post_trans_T, sensor2sensors)
        return inputs_S, inputs_T
    
    def __call__(self, results):
        inputs_S, inputs_T = self.get_inputs(results)
        results['img_inputs_S'] = inputs_S
        results['img_inputs_T'] = inputs_T
        return results



@PIPELINES.register_module()
class PointToMultiViewDepth_Distill(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points'] # torch.Size([34752, 5])
        for k in ['img_inputs_S', 'img_inputs_T']:
            imgs, rots, trans, intrins = results[k][:4]
            post_rots, post_trans, bda, sensor2sensors = results[k][4:8]
            depth_map_list = []
            for cid in range(len(results['cam_names'])):
                cam_name = results['cam_names'][cid]
                lidar2lidarego = np.eye(4, dtype=np.float32)
                lidar2lidarego[:3, :3] = Quaternion(
                    results['curr']['lidar2ego_rotation']).rotation_matrix
                lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
                lidar2lidarego = torch.from_numpy(lidar2lidarego)

                lidarego2global = np.eye(4, dtype=np.float32)
                lidarego2global[:3, :3] = Quaternion(
                    results['curr']['ego2global_rotation']).rotation_matrix
                lidarego2global[:3, 3] = results['curr']['ego2global_translation']
                lidarego2global = torch.from_numpy(lidarego2global)

                cam2camego = np.eye(4, dtype=np.float32)
                cam2camego[:3, :3] = Quaternion(
                    results['curr']['cams'][cam_name]
                    ['sensor2ego_rotation']).rotation_matrix
                cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                    'sensor2ego_translation']
                cam2camego = torch.from_numpy(cam2camego)

                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = Quaternion(
                    results['curr']['cams'][cam_name]
                    ['ego2global_rotation']).rotation_matrix
                camego2global[:3, 3] = results['curr']['cams'][cam_name][
                    'ego2global_translation']
                camego2global = torch.from_numpy(camego2global)

                cam2img = np.eye(4, dtype=np.float32)
                cam2img = torch.from_numpy(cam2img)
                cam2img[:3, :3] = intrins[cid]

                lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                    lidarego2global.matmul(lidar2lidarego))
                lidar2img = cam2img.matmul(lidar2cam)
                points_img = points_lidar.tensor[:, :3].matmul(
                    lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                points_img = torch.cat(
                    [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                    1)
                points_img = points_img.matmul(
                    post_rots[cid].T) + post_trans[cid:cid + 1, :]
                depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                                imgs.shape[3])
                depth_map_list.append(depth_map)
            for adj_idx in range(len(results['adjacent'])):
                ressults_adj_i = results['adjacent'][adj_idx]
                points_lidar_adj = results['adj_points'][adj_idx]
                for _id in range(len(results['cam_names'])):
                    cam_name = results['cam_names'][_id]
                    cid = 6*(1+adj_idx) + _id
                    lidar2lidarego = np.eye(4, dtype=np.float32)
                    lidar2lidarego[:3, :3] = Quaternion(
                        ressults_adj_i['lidar2ego_rotation']).rotation_matrix
                    lidar2lidarego[:3, 3] = ressults_adj_i['lidar2ego_translation']
                    lidar2lidarego = torch.from_numpy(lidar2lidarego)

                    lidarego2global = np.eye(4, dtype=np.float32)
                    lidarego2global[:3, :3] = Quaternion(
                        ressults_adj_i['ego2global_rotation']).rotation_matrix
                    lidarego2global[:3, 3] = ressults_adj_i['ego2global_translation']
                    lidarego2global = torch.from_numpy(lidarego2global)

                    cam2camego = np.eye(4, dtype=np.float32)
                    cam2camego[:3, :3] = Quaternion(
                        ressults_adj_i['cams'][cam_name]
                        ['sensor2ego_rotation']).rotation_matrix
                    cam2camego[:3, 3] = ressults_adj_i['cams'][cam_name][
                        'sensor2ego_translation']
                    cam2camego = torch.from_numpy(cam2camego)

                    camego2global = np.eye(4, dtype=np.float32)
                    camego2global[:3, :3] = Quaternion(
                        ressults_adj_i['cams'][cam_name]
                        ['ego2global_rotation']).rotation_matrix
                    camego2global[:3, 3] = ressults_adj_i['cams'][cam_name][
                        'ego2global_translation']
                    camego2global = torch.from_numpy(camego2global)

                    cam2img = np.eye(4, dtype=np.float32)
                    cam2img = torch.from_numpy(cam2img)
                    cam2img[:3, :3] = intrins[cid]

                    lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                        lidarego2global.matmul(lidar2lidarego))
                    lidar2img = cam2img.matmul(lidar2cam)
                    points_img = points_lidar_adj.tensor[:, :3].matmul(
                        lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                    points_img = torch.cat(
                        [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                        1)
                    points_img = points_img.matmul(
                        post_rots[cid].T) + post_trans[cid:cid + 1, :]
                    depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                                    imgs.shape[3])
                    depth_map_list.append(depth_map)
            depth_map = torch.stack(depth_map_list)
            results['gt_depth'+k[-2:]] = depth_map
        return results




@PIPELINES.register_module()
class LoadAnnotationsBEVDepth_Distill(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        bda_aug_param = {'flip_dx': flip_dx, 'flip_dy': flip_dy, 'scale_ratio':scale_ratio, 'rotate_angle': rotate_angle}
        return gt_boxes, rot_mat, bda_aug_param

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot, bda_aug_param = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        # student
        imgs, rots, trans, intrins = results['img_inputs_S'][:4]
        post_rots, post_trans, sensor2sensors = results['img_inputs_S'][4:]
        results['img_inputs_S'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot, sensor2sensors)
        # teacher
        imgs, rots, trans, intrins = results['img_inputs_T'][:4]
        post_rots, post_trans, sensor2sensors = results['img_inputs_T'][4:]
        results['img_inputs_T'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot, sensor2sensors)
        results['bda_aug_param'] = bda_aug_param
        
        adj_gt_boxes_list = []
        adj_gt_labels_list = []
        adj_index_list = [0]
        for ann_info_adj in results['adjacent_ann']:
            gt_boxes, gt_labels = ann_info_adj
            gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
            gt_boxes, bda_rot, bda_aug_param = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                                flip_dx, flip_dy)
            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            adj_gt_boxes_list.append(gt_boxes)
            adj_gt_labels_list.append(gt_labels)
            adj_index_list.append(gt_boxes.shape[0])
        
        gt_bboxes_3d_adj = torch.cat(adj_gt_boxes_list, dim=0)
        gt_bboxes_3d_adj = LiDARInstance3DBoxes(gt_bboxes_3d_adj, box_dim=gt_bboxes_3d_adj.shape[-1],
                                            origin=(0.5, 0.5, 0.5))
        gt_labels_3d_adj = torch.cat(adj_gt_labels_list, dim=0)
        adj_index = torch.tensor(adj_index_list)

        results['gt_bboxes_3d_adj'] = gt_bboxes_3d_adj
        results['gt_labels_3d_adj'] = gt_labels_3d_adj
        results['adj_index'] = adj_index

        return results


@PIPELINES.register_module()
class LoadMotionTrajectory(object):

    def __init__(self, is_train=True):
        self.is_train = is_train

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        # rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        bda_aug_param = {'flip_dx': flip_dx, 'flip_dy': flip_dy, 'scale_ratio':scale_ratio, 'rotate_angle': rotate_angle}
        return gt_boxes, rot_mat, bda_aug_param

    def __call__(self, results):
        adj_gt_boxes_list = []
        adj_gt_labels_list = []
        adj_index_list = [0]
        flip_dx, flip_dy, scale_ratio, rotate_angle = results['bda_aug_param'].values()
        for ann_info_adj in results['motion_ann']:
            gt_boxes, gt_labels = ann_info_adj
            gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
            gt_boxes, bda_rot, bda_aug_param = self.bev_transform(gt_boxes, rotate_angle, scale_ratio,
                                                flip_dx, flip_dy)
            assert results['bda_aug_param'] == bda_aug_param, print(results['bda_aug_param'], bda_aug_param)

            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            adj_gt_boxes_list.append(gt_boxes)
            adj_gt_labels_list.append(gt_labels)
            adj_index_list.append(gt_boxes.shape[0])
        
        gt_bboxes_3d_adj = torch.cat(adj_gt_boxes_list, dim=0)
        gt_bboxes_3d_adj = LiDARInstance3DBoxes(gt_bboxes_3d_adj, box_dim=gt_bboxes_3d_adj.shape[-1],
                                            origin=(0.5, 0.5, 0.5))
        gt_labels_3d_adj = torch.cat(adj_gt_labels_list, dim=0)
        adj_index = torch.tensor(adj_index_list)

        results['gt_bboxes_3d_adj'] = gt_bboxes_3d_adj
        results['gt_labels_3d_adj'] = gt_labels_3d_adj
        results['adj_index'] = adj_index

        return results
