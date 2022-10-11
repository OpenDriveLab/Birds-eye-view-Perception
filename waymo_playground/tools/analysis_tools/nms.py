import mmcv
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.core.bbox import get_box_type
from pyquaternion import Quaternion
import torch
import numpy as np
from mmdet3d.core.post_processing import aligned_3d_nms


data_p1 = '/home/lizhiqi/bevformer/test/9/Wed_Apr_13_18_19_18_2022/pts_bbox/results_nusc.json'
data_p2 = '/home/lizhiqi/bevformer/test/1/Wed_Apr_13_18_18_17_2022/pts_bbox/results_nusc.json'
data_p3 = '/home/lizhiqi/bevformer/test/v4_encoder3_ms/Wed_Apr_13_18_08_09_2022/pts_bbox/results_nusc.json'

final_results = mmcv.load(data_p1)
data1 = mmcv.load(data_p1)['results']
data2 = mmcv.load(data_p2)['results']
data3 = mmcv.load(data_p3)['results']

keys = list(data1.keys())
box_type_3d = 'LiDAR'
box_type_3d, box_mode_3d = get_box_type(box_type_3d)

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
           'barrier')
cls2id = {name: _id for _id, name in enumerate(CLASSES)}

for key in keys:
    s1, s2, s3 = data1[key], data2[key], data3[key]
    from IPython import embed
    all_results = np.array(s1+s2+s3)

    s1_bbox = [each['translation'] + each['size'] + [Quaternion(each['rotation']).yaw_pitch_roll[0]] for each in s1]
    s1_label = [cls2id[each['detection_name']] for each in s1]
    s1_scores = [each['detection_score'] for each in s1]

    s2_bbox = [each['translation'] + each['size'] + [Quaternion(each['rotation']).yaw_pitch_roll[0]] for each in s2]
    s2_label = [cls2id[each['detection_name']] for each in s2]
    s2_scores = [each['detection_score'] for each in s2]

    s3_bbox = [each['translation'] + each['size'] + [Quaternion(each['rotation']).yaw_pitch_roll[0]] for each in s3]
    s3_label = [cls2id[each['detection_name']] for each in s3]
    s3_scores = [each['detection_score'] for each in s3]

    all_bbox = s1_bbox + s2_bbox + s3_bbox
    all_label = torch.tensor(s1_label + s2_label + s3_label)
    all_scores = torch.tensor(s1_scores + s2_scores + s3_scores)

    all_bboxes_3d = LiDARInstance3DBoxes(
        all_bbox,
        box_dim=7,
        origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)

    all_corner3d = all_bboxes_3d.corners
    minmax_box3d = all_corner3d.new(torch.Size((all_corner3d.shape[0], 6)))
    minmax_box3d[:, :3] = torch.min(all_corner3d, dim=1)[0]
    minmax_box3d[:, 3:] = torch.max(all_corner3d, dim=1)[0]

    nms_selected = aligned_3d_nms(minmax_box3d,
                                  all_scores,
                                  all_label,
                                  0.3)[:500]

    all_results = list(all_results[nms_selected.cpu().numpy()])
    from IPython import embed
    embed()
    exit()
    data1[key] = all_results
final_results['results'] = data1
mmcv.dump(final_results, 'nms.json')
