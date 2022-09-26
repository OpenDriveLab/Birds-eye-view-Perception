import numpy as np
import cv2

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def scale_image_multiple_view(imgs, lidar2img, rand_scale, interpolation='bilinear'):

    y_size = [int(img.shape[0] * rand_scale) for img in imgs]
    x_size = [int(img.shape[1] * rand_scale) for img in imgs]

    scale_factor = np.eye(4)
    scale_factor[0, 0] *= rand_scale
    scale_factor[1, 1] *= rand_scale
    imgs_new = [
        cv2.resize(img, (x_size[idx], y_size[idx]), interpolation=cv2_interp_codes[interpolation])
        for idx, img in enumerate(imgs)
    ]
    lidar2img_new = [scale_factor @ l2i for l2i in lidar2img]
    return imgs_new, lidar2img_new
