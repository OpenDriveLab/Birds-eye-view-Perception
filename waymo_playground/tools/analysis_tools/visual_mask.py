import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import json
import cv2
from io import BytesIO
import PIL

from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap
#nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
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


def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = None,
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('cams', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        print(i)
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)


from pyquaternion import Quaternion


def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False):
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
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


from pyquaternion import Quaternion


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
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
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

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
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


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


def lidiar_render(sample_token, data,out_path=None):
    bbox_gt_list = []
    bbox_pred_list = []
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        try:
            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category_to_detection_name(content['category_name']),
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))
        except:
            pass

    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(DetectionBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name']))
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    print('green is ground truth')
    print('blue is the predited result')
    visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, savepath=out_path+'_lidar')


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    #print(key)
    return [0, 0, 0]


def render_sample_data(
        samples,
        sample_toekn = None,
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
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    if ax is None:
        plt.subplots_adjust(left=0.0, bottom=0.0, top=0.1, right=0.1)
        _, ax = plt.subplots(2, 3, figsize=(10, 4))


    j = 0
    data_list = []
    for ind, cam in enumerate(cams):

        sample_data_token = samples['cams'][cam]['sample_data_token']

        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = 'camera'  # sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                         name=record['detection_name'], token='predicted') for record in
                     pred_data['results'][sample_toekn] if record['detection_score'] > 0.3]

            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                                                                         box_vis_level=box_vis_level, pred_anns=boxes)
            #_, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(data_path)
            #mmcv.imwrite(np.array(data)[:,:,::-1], f'{cam}.png')
            # Init axes.

            # Show image.
            ax[j, ind].imshow(data)
            #ax[j + 2, ind].imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes_pred:
                    c = np.array(get_color(box.name)) / 255.0
                    box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=0.5)
                #for box in boxes_gt:
                #    c = np.array(get_color(box.name)) / 255.0
                #    box.render(ax[j + 2, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)
            # buffer_ = BytesIO()  # using buffer,great way!
            # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
            #ax[j, ind].figure.savefig(buffer_, format='png')
            #buffer_.seek(0)
            # 用PIL或CV2从内存中读取
            #dataPIL = PIL.Image.open(buffer_)
            # 转换为nparrary，PIL转换就非常快了,data即为所需
            #data = np.asarray(dataPIL)
            # from IPython import embed
            # embed()
            # exit()
            # data_list.append(data)


        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax[j, ind].axis('off')
        #ax[j, ind].set_title('PRED: {} {labels_type}'.format(
        #    sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax[j, ind].set_aspect('equal')

        #ax[j + 2, ind].axis('off')
        #ax[j + 2, ind].set_title('GT:{} {labels_type}'.format(
        #    sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        #ax[j + 2, ind].set_aspect('equal')

    if out_path is not None:
        #from IPython import embed
        #embed()
        #exit()
        plt.savefig(out_path+'_img', bbox_inches='tight', pad_inches=0, dpi=200)
    #if verbose:
    #    plt.show()
    plt.close()

# import json
bevformer = 'test/bevformer_v4_raw/Fri_Jan_14_00_11_36_2022/pts_bbox/results_nusc.json'
bevformer_mask = 'test/bevformer_mask3_raw/Tue_Feb_22_15_04_02_2022/pts_bbox/results_nusc.json'
gt = '/home/lizhiqi/bevformer/data/nuscenes/nuscenes_infos_raw.pkl'
# detr3d = '/home/lzq/workspace/bev/bevformer/test/detr3d_res101_gridmask/Tue_Dec_14_14_14_59_2021/pts_bbox/results_nusc.json'
data1 = json.load(open(bevformer_mask, 'r'))
gts = mmcv.load(gt)['infos']

def make_rgba(probs, color):
    H, W = probs.shape
    return np.stack((
    np.full((H, W), color[0]),
    np.full((H, W), color[1]),
    np.full((H, W), color[2]),
    probs,
    ), 2)
import torch
class F():
    def __init__(self):

        self.xbound = [-51.2, 51.2, 0.5]
        self.ybound = [-51.2, 51.2, 0.5]
        self.zbound = [-10.0, 10.0, 20.0]
        self.dbound = [4.0, 45.0, 1.0]
    def gen_dx_bx(self):
        dx = torch.Tensor([row[2] for row in [self.xbound, self.ybound, self.zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [self.xbound, self.ybound, self.zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [self.xbound, self.ybound, self.zbound]])

        return dx, bx, nx

def add_ego(img, bx, dx):
    pts = np.array([
        [-4.084/2., 1.73/2.],
        [4.084/2., 1.73/2.],
        [4.084/2., -1.73/2.],
        [-4.084/2., -1.73/2.],
    ])
    pts = np.round(
        (pts - bx + dx/2.) / dx
        ).astype(np.int32)
    pts[:, [1, 0]] = pts[:, [0, 1]]
    cv2.fillPoly(img, [pts], (0.0, 0.0, 0.0, 0.5))
    return img

token = '713882a28f2442098451f77f1362d326'

#data = np.load('.cache/mask_gt_1000_1000_0.10_mini.npy')[7]
data = np.load(f"npy/1121.npy")

# map = np.ones([2000, d ..2000])

car = data[0]
verh = data[1]
ped = data[2]
div = data[3]
boundary = data[4]
road = data[5]
colors = [
        [255.00, 127, 79],
        [0, 255, 255],
        [0, 255, 255],
        [0, 255, 0],
        [1, 145, 247]
    ]
plt.figure()
    # plt.clf()
showimg = make_rgba(road, (1.00, 0.50, 0.31))
plt.imshow(showimg, origin='lower')
showimg = make_rgba(div + boundary, (0.0, 1.0, 1.0))
plt.imshow(showimg, origin='lower')
showimg = make_rgba(verh, (0.004, 0.569, 0.969))
plt.imshow(showimg, origin='lower')
add_ego(showimg, np.array([-49.75, -49.75]), np.array([0.1, 0.1]))
plt.imshow(showimg, origin='lower')
plt.xlim((0, showimg.shape[1]))
plt.ylim((0, showimg.shape[0]))
#plt.axis('on')
print(showimg.shape)



plt.savefig(f'tmp.png')
    # data = data[1:].transpose(1, 2, 0).argmax(-1)
    # for i, tmp in enumerate([road,  div, boundary, ped, verh]):
    #     for j in range(200):
    #         for k in range(200):
    #             color = colors[i]
    #             if tmp[j, k] > 0.3:
    #                 map[j, k] = [(tmp[j, k]/2+0.5) * a for a in color]
    #     # map[road > 0.5] =
    #     # map[boundary > 0.5] = [255, 0, 0]
    #     # map[div>0.5] =
    #     # map[ped>0.5] =
    #     # map[verh>0.5] =
    # map[95:105, 98:102] = [255, 0, 0]
    # map = np.flip(map, 0)
    # mmcv.imwrite(map[:, :, ::-1], f'npys/tmp_{ind}.png')
#print(ind, ' ok')
# for idx, pred in enumerate(data1['results']):
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