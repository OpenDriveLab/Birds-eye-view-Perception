import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points,  BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap

# nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes-mini', verbose=True)
#with open('../center_overlap.txt', 'r') as f:
#    anns2 = [each.strip() for each in f.readlines()]


cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams
from pyquaternion import quaternion



from pyquaternion import Quaternion


def box_in_image(box, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    #print('corners_3d', corners_3d)
    #print('intrinsic',intrinsic)
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
    #print('corners_img', corners_img)
    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def get_predicted_data(camera_intrinsic,
                        lidar2ego_translation,
                        lidar2ego_rotation,
                        sensor2ego_translation,
                        sensor2ego_rotation,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None,
                       cam_type=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    #sd_record = nusc.get('sample_data', sample_data_token)
    #cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    #sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    #pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    #data_path = nusc.get_sample_data_path(sample_data_token)


    cam_intrinsic = np.array(camera_intrinsic)
    imsize = (1920, 1280)


    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            # yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            # box.translate(-np.array(pose_record['translation']))
            # box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            pass
        else:
            # Move box to ego vehicle coord system.
            #print('lidar2ego_translation', lidar2ego_translation)
            #print(1, box)
            print('lidar2ego_translation',lidar2ego_translation)
            lidar2ego_translation = [-lidar2ego_translation[1], lidar2ego_translation[2], lidar2ego_translation[0]]
            box.translate(-np.array(lidar2ego_translation))
            #print(lidar2ego_rotation)
            # box.rotate(Quaternion(lidar2ego_rotation).inverse)

            #print('lidar2ego_rotation', lidar2ego_rotation)
            #print(2, box)
            #print('sensor2ego_translation', sensor2ego_translation)
            #  Move box to sensor coord system.
            sensor2ego_translation = [-sensor2ego_translation[1], sensor2ego_translation[2], sensor2ego_translation[0]]
            print('sensor2ego_translation', sensor2ego_translation)
            box.translate(np.array(sensor2ego_translation))
            #print(2.5, box)
            print('sensor2ego_rotation', sensor2ego_rotation, cam_type)
            print(Quaternion(sensor2ego_rotation).yaw_pitch_roll)
            box.rotate(Quaternion(axis=[1, 0, 0],angle=Quaternion(sensor2ego_rotation).yaw_pitch_roll[0]/np.pi*180))
            print(3, box)
            #exit()
        if not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        else:
            print('PASS')

        box_list.append(box)

    return box_list


from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
import os.path as osp
import os
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances, \
    get_panoptic_instances_stats
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample




def render_sample_data(
        gt,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    # lidiar_render(sample_toekn, pred_data, out_path=out_path)
    # sample = nusc.get('sample', sample_toekn)
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        'FRONT',
        'FRONT_LEFT',
        'FRONT_RIGHT',
        'SIDE_LEFT',
        'SIDE_RIGHT'
    ]
    if ax is None:
        _, ax = plt.subplots(2, 3, figsize=(18, 8))
    j = 0
    lidar2ego_translation = gt['lidar2ego_translation']
    lidar2ego_rotation = gt['lidar2ego_rotation']

    for ind, cam in enumerate(cams):
        cam = gt['cams'][cam]
        
        #if cam['type'] !='FRONT': continue

        cam_path = cam['data_path']
        camera_intrinsic = cam['cam_intrinsic']
        sensor2ego_translation = cam['sensor2ego_translation']
        sensor2ego_rotation = cam['sensor2ego_rotation']
        # sample_data_token = samples['cams'][cam]['sample_data_token']

        #sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = 'camera'  # sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':

            boxes = [Box([-record[1], record[2], record[0]], record[3:6],  quaternion.Quaternion(axis=[1, 0, 0], angle=record[-1]),
                         name=0, token='predicted') for record in gt['gt_boxes']]
            #boxes = [
            #    Box([-5, 0., 10], [2.,4.,1.],  quaternion.Quaternion(axis=[0, 0, 1], angle=np.pi/2, name=0, token='test'))
            #]

            # print(lidar2ego_rotation, sensor2ego_rotation)
            boxes_pred = get_predicted_data(
                camera_intrinsic,
                lidar2ego_translation,
                lidar2ego_rotation,
                sensor2ego_translation,
                sensor2ego_rotation,
                box_vis_level=box_vis_level, pred_anns=boxes,
                cam_type = cam['type']
                )

            #_, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(cam_path)
            #mmcv.imwrite(np.array(data)[:,:,::-1], f'{cam}.png')
            # Init axes.

            # Show image.
            ax[j, ind].imshow(data)
            #ax[j + 2, ind].imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes_pred:
                    c = [1, 0, 0]  # np.array(get_color(box.name)) / 255.0
                    box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))
                #for box in boxes_gt:
                #    c = np.array(get_color(box.name)) / 255.0
                #    box.render(ax[j + 2, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)
            #ax[j + 2, ind].set_xlim(0, data.size[0])
            #ax[j + 2, ind].set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax[j, ind].axis('off')
        ax[j, ind].set_title('PRED:'.format())
        ax[j, ind].set_aspect('equal')

        #ax[j + 2, ind].axis('off')
        #ax[j + 2, ind].set_title('GT:{} {labels_type}'.format(
        #    sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        #ax[j + 2, ind].set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path+'_img', bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()

#import json
#bevformer = 'test/bevformer_v4_raw/Tue_Jan_11_11_50_33_2022/pts_bbox/results_nusc.json'
gt = 'train.pkl'
#detr3d = '/home/lzq/workspace/bev/bevformer/test/detr3d_res101_gridmask/Tue_Dec_14_14_14_59_2021/pts_bbox/results_nusc.json'
#data1 = json.load(open(bevformer, 'r'))
gts = mmcv.load(gt)['infos']

for ind, gt in enumerate(gts[10:]):
    # print(pred)
    # break
    render_sample_data(gt, out_path=str(ind))
    break

#for idx, pred in enumerate(data1['results']):
#    print(pred)
#    render_sample_data(None, pred_data=data1['results'][pred], out_path=str(idx))


#data2 = json.load(open(detr3d,'r'))
#sample_token_list1 = list(data1['results'].keys())
#sample_token_list2 = list(data2['results'].keys())
#for id in range(0, 1):
#    render_sample_data('206a62a42e2c4ec8b52f902d54fa2599', pred_data=data1, out_path=str(id))
#nusc.lidiar_render('206a62a42e2c4ec8b52f902d54fa2599', out_path='206a62a42e2c4ec8b52f902d54fa2599')
#idx = '3e8750f331d7499e9b5123e9eb70f2e2'
#render_sample_data(idx, pred_data=data1, out_path=str(idx))
#render_sample_data(sample_token_list2[id],pred_data=data2)
#my_sample = nusc.get('sample', idx)
#nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, out_path=idx+'_NB')


#my_sample = nusc.get('sample', 'a1311722c1de4bb3b4a9d36a6246da9d')
#nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True, out_path='a1311722c1de4bb3b4a9d36a6246da9d_lidar')