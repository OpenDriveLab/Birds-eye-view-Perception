# copy from https://github.com/Tsinghua-MARS-Lab/HDMapNet/blob/main/data/dataset.py

import os

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes

from torch.utils.data import Dataset

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon
import json
import numpy as np

import torch
from nuscenes.utils.splits import create_splits_scenes
import cv2
from shapely import affinity
from shapely.geometry import LineString, box



class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 drivable_area_classes=['drivable_area'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1,
                 class2label={
                     'ped_crossing': 0,
                     'road_divider': 1,
                     'lane_divider': 1,
                     'contours': 2,
                     'drivable_area': 3,
                     'others': -1,
                 }):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.drivable_area_classes = drivable_area_classes
        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation, canvas_size=(200,200)):
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
        ped_vector_list = self.ped_geoms_to_vectors(ped_geom)

        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        drivable_mask = self.map_explorer[location].get_map_mask(patch_box, patch_angle, self.drivable_area_classes, canvas_size=canvas_size)
        #area_geom = self.get_map_geom(patch_box, patch_angle, self.drivable_area_classes, location)
        #self.nusc_maps[location]. _polygon_geom_to_mask(area_geom, patch_box)
        #area_geom = self.poly_geoms_to_vectors(area_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():

            for line, length in vects:
                vectors.append((line.astype(float), length, self.class2label.get(line_type, -1)))

        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, self.class2label.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, self.class2label.get('contours', -1)))

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        return filtered_vectors, drivable_mask

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.drivable_area_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line:
                        line_vectors.append(self.sample_pts_from_line(l))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors


    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid


class HDMapNetDataset(Dataset):
    def __init__(self, version='v1.0-trainval', dataroot='data/nuscenes', xbound=[-30., 30., 0.15], ybound=[-15., 15., 0.15]):
        super(HDMapNetDataset, self).__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, idx):
        rec = self.nusc.sample[idx]
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])

        imgs = []
        trans = []
        rots = []
        intrins = []
        for cam in CAMS:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
            intrins.append(torch.Tensor(sens['camera_intrinsic']))

        return imgs, torch.stack(trans), torch.stack(rots), torch.stack(intrins), vectors


def rasterize_map(vectors, patch_size, canvas_size, max_channel, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel + 1):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line, confidence in layer_geom:
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx)
    return map_mask, idx



def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def mask_for_lines(lines, mask, thickness, idx):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
    idx += 1
    return mask, idx


class HDMapNetEvalDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, eval_set, result_path, thickness, max_line_count=100, max_channel=3, xbound=[-30., 30., 0.15], ybound=[-15., 15., 0.15]):
        super(HDMapNetEvalDataset, self).__init__(version, dataroot, xbound, ybound)
        scenes = create_splits_scenes()[eval_set]
        with open(result_path, 'r') as f:
            self.prediction = json.load(f)
        self.samples = [samp for samp in self.nusc.sample if self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        self.max_line_count = max_line_count
        self.max_channel = max_channel
        self.thickness = thickness

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        gt_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])

        gt_map, _ = rasterize_map(gt_vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness)
        if self.prediction['meta']['vector']:
            pred_vectors = self.prediction['results'][rec['token']]
            pred_map, confidence_level = rasterize_map(pred_vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness)
        else:
            pred_map = np.array(self.prediction['results'][rec['token']]['map'])
            confidence_level = self.prediction['results'][rec['token']]['confidence_level']

        confidence_level = torch.tensor(confidence_level + [-1] * (self.max_line_count - len(confidence_level)))

        return pred_map, confidence_level, gt_map


class HDMap(object):

    def __init__(self, nusc, data_root, thickness=2, point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 max_line_count=100, max_channel=3, bev_size=(200, 200)):
        ybound = [point_cloud_range[1], point_cloud_range[4], (point_cloud_range[4] - point_cloud_range[1])/bev_size[1]]
        xbound = [point_cloud_range[0], point_cloud_range[3], (point_cloud_range[3] - point_cloud_range[0])/bev_size[0]]
        # xbound = [-51.2, 51.2, 0.512], ybound = [-51.2, 51.2, 0.512],
        self.grid_length = '%.2f' % ((point_cloud_range[4] - point_cloud_range[1])/bev_size[1])

        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.max_line_count = max_line_count
        self.max_channel = max_channel
        self.thickness = thickness
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        # print((patch_h, patch_w),  (canvas_h, canvas_w))
        self.nusc = nusc
        self.vector_map = VectorizedLocalMap(data_root, patch_size=self.patch_size, canvas_size=self.canvas_size)

    def get(self, rec):

        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors, drivable_mask = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'], canvas_size=self.canvas_size)
        gt_map, _ = rasterize_map(vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness)

        gt_map = np.concatenate([gt_map, drivable_mask], 0)
        gt_map = gt_map.transpose(0, 2, 1)
        return gt_map>0

