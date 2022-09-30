import time
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from projects.mmdet3d_plugin.models.utils.visual import save_tensor


def point_sampling(reference_points, pc_range, device, img_metas, gt_bboxes_3d=None, dataset_type='nuscenes'):
    file_names = img_metas[0]['filename']

    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    # for i, each in enumerate(lidar2img):
    #    print(i, each)
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()

    reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
    # if dataset_type == 'waymo':
    #    reference_points[..., 1:2] = pc_range[4]-reference_points[..., 1:2] * \
    #                             (pc_range[4] - pc_range[1])
    # else:
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]

    # print(reference_points)
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    # print('ref', reference_points)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    # print(lidar2img)
    # print('ref', reference_points[0])

    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    # print('ref', reference_points[0])
    # print('lidar', lidar2img[0])

    reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
    tmp2 = reference_points_cam[0, 0, 0].reshape(-1, 4)
    tmp = reference_points[0, 0, 0].reshape(-1, 4)
    # for i, (each1, each2) in enumerate(zip(tmp, tmp2)):
    #    print(i, each1, (each2[0]/each2[2]).item(), (each2[1]/each2[2]).item())
    # reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    # reference_points_cam = torch.matmul(lidar2img.cpu(), reference_points.cpu()).to(device).squeeze(-1)

    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    # print('reference_points_cam[..., 0:2]',reference_points_cam[..., 0:2])

    # TODO use ori_shape
    # print('reference_points_cam', reference_points_cam.shape)
    # print( img_metas[0]['ori_shape'])
    if len(list(set(img_metas[0]['ori_shape']))) > 1:  # if cams have different input shape
        for i, ori_shape in enumerate(img_metas[0]['ori_shape']):
            reference_points_cam[..., i, :, 0] /= ori_shape[1]
            reference_points_cam[..., i, :, 1] /= ori_shape[0]
    else:
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    # reference_points_cam = (reference_points_cam - 0.5) * 2

    mask = (mask & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0))
    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        mask = torch.nan_to_num(mask)
    else:
        mask = mask.new_tensor(np.nan_to_num(mask.cpu().numpy()))
    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    mask = mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    # tmp = mask.permute(0, 1, 3, 2).reshape(-1, 200, 200)
    # save_tensor(tmp, 'tmp_nusc.png')
    # exit()
    # print(mask.shape)
    # print(mask.shape, reference_points_cam.shape)
    # torch.Size([6, 1, 80000, 4]) torch.Size([6, 1, 80000, 4, 2])

    valid_mask = mask.permute(3, 0, 2, 1).view(D, num_cam, num_query, 1)
    valid_points = reference_points_cam.permute(3, 0, 2, 1, 4).view(D, num_cam, num_query, 2)
    # valid_lidar = lidar2img.view(D, num_cam, num_query, 4).permute(2, 0, 1, 3)
    valid_mask = valid_mask.permute(2, 0, 1, 3)
    valid_points = valid_points.permute(2, 0, 1, 3)

    # print(valid_mask.shape, valid_points.shape)

    def nothing(x):
        pass

    # lambda x: pass
    # cv.namedWindow('view', cv.WINDOW_KEEPRATIO)
    # cv.createTrackbar('x', 'view', 0, 50*2, nothing)
    # cv.createTrackbar('y', 'view', 0, 50*2, nothing)
    # cv.createTrackbar('z', 'view', 0, 30*2, nothing)
    # for i in range(0,num_query):
    imgs = []
    for each in file_names:
        img = mmcv.imread(each)
        imgs.append(img)

    # while True:
    # x = cv.getTrackbarPos('x', 'view')-50
    # y = cv.getTrackbarPos('y', 'view')-50
    # z = cv.getTrackbarPos('z', 'view')-30
    if gt_bboxes_3d is None:
        return reference_points_cam, mask
    # for center in gt_bboxes_3d[0].gravity_center:
    # tmp = reference_points.reshape(-1,4)
    # for i in range(len(tmp)):
    # x = i % 200 * 0.7488 - 74.88
    # y = i // 200 * 0.7488 - 74.88
    # t = tmp[i]
    # for t in [(10,0,0)]:
    # points = [t[0], t[1], t[2]]
    for center in gt_bboxes_3d[0].gravity_center:

        points = [center[0], center[1], center[2]]

        points.append(1)
        print(points)
        points = torch.tensor(points, device=lidar2img.device)
        points = points.view(1, 1, 1, 1, 4).repeat(
            1, 1, num_cam, num_query, 1).unsqueeze(-1)
        points = torch.matmul(lidar2img[0:1], points).squeeze(-1)

        eps = 1e-5

        points = points[..., 0:2] / \
                 torch.maximum(points[..., 2:3],
                               torch.ones_like(points[..., 2:3]) * eps)
        points = points.view(1, num_cam, num_query, 2)

        for k in range(num_cam):
            per_point = points[0, k, 0, :]
            per_point = per_point.cpu().numpy()
            try:
                per_point = (int(per_point[0]), int(per_point[1]))
            except:
                continue

            if 0 < per_point[0] < img_metas[0]['img_shape'][0][1] and 0 < per_point[1] < img_metas[0]['img_shape'][0][
                0]:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.circle(imgs[k], per_point, radius=5,
                          color=(255, 0, 0), thickness=4)
                text = '%.0f,%.0f' % (center[0].item(), center[1].item())
                print(text)
                cv.putText(imgs[k], text, per_point, font, 1.2, (255, 255, 255), 2)
    first_row = np.hstack([imgs[1], imgs[0], imgs[2]])
    sencond_row = np.hstack([imgs[3], imgs[4], imgs[4]])
    view = np.vstack([first_row, sencond_row])
    mmcv.imwrite(view, f'{time.time()}.png')
    exit(0)
    # cv.imshow('view', view)
    # k = cv.waitKey(1) & 0xFF
    # if k == 27:
    #    break
    # plt.figure(figsize=(9, 5))
    # plt.subplot(231)
    # plt.imshow(imgs[2])
    # plt.axis('off')
    # plt.subplot(232)
    # plt.imshow(imgs[0])
    # plt.axis('off')
    # plt.subplot(233)
    # plt.imshow(imgs[1])
    # plt.axis('off')
    # plt.subplot(234)
    # plt.imshow(imgs[4])
    # plt.axis('off')
    # plt.subplot(235)
    # plt.imshow(imgs[3])
    # plt.axis('off')
    # plt.subplot(236)
    # plt.imshow(imgs[5])
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # plt.close()
    # print('reference_points_cam[..., 0:2]',reference_points_cam[..., 0:2])

    return reference_points_cam, mask
