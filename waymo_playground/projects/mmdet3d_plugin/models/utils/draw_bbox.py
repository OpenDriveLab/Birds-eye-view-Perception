## copy-paste from mmdet3d. Used to debug
import copy
import os.path as osp
import numpy as np
import cv2
import torch
import mmcv
from mmdet3d.core.bbox.iou_calculators import BboxOverlaps3D
from mmdet3d.core.visualizer.image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img)

c_iou = BboxOverlaps3D(coordinate='lidar')


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1,
                       img_metas=None):
    """Plot the boundary lines of 3D rectangular on 2D images.
    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]

    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for j, (start, end) in enumerate(line_indices):
            try:
                #if j in [7, 8, 10, 11]:
                #    front_thickness = thickness + 1
                #    cv2.line(img, (corners[start, 0], corners[start, 1]),
                #             (corners[end, 0], corners[end, 1]), color, front_thickness,
                #             cv2.LINE_AA)
                #else:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                             (corners[end, 0], corners[end, 1]), color, thickness+2,
                             cv2.LINE_AA)
                if img_metas != 0 and j == 0:
                    text = img_metas[i]
                    cv2.putText(img, '%.1f %.1f %.1f' % (text[0], text[1], text[2]), (corners[start, 0], corners[start, 1]),
                                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
            except:
                pass

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.
    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness, img_metas)


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode='lidar',
                               img_metas=None,
                               show=True,
                               scores=None,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.
    Project the predicted 3D bbox to 2D image plane and visualize them.
    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in. Should be one of
           'depth', 'lidar' and 'camera'. Defaults to 'lidar'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    # embed()
    # exit()

    # mmcv.mkdir_or_exist(out_dir)
    if scores is not None:
        keep = scores > 0.1
        pred_bboxes = pred_bboxes[keep]
    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            text = [[bbox[0], bbox[1], bbox[6]] for bbox in gt_bboxes.tensor.cpu().numpy()]
                #list(c_iou(gt_bboxes.tensor, pred_bboxes.tensor).max(1).values.cpu().numpy())
            img_metas = text

            try:
                show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
            except:
                print('error')
                pass
        if pred_bboxes is not None:
            try:
                show_img = draw_bbox(
                    pred_bboxes,
                    show_img,
                    proj_mat,
                    None,
                    color=pred_bbox_color)
            except:
                pass

        mmcv.imwrite(show_img, result_path)

        # mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)
    # print()
    # embed()
    return