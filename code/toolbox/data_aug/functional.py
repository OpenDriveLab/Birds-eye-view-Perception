import numpy as np
import cv2
from typing import List, Tuple

#  Available interpolation modes (opencv)
cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def scale_image_multiple_view(imgs: List[np.ndarray],
                              lidar2img: List[np.ndarray],
                              rand_scale: float,
                              interpolation='bilinear') -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Resize the multiple-view images with the same scale selected randomly.
    Notably used in :class:`.transforms.RandomScaleImageMultiViewImage_naive
    Args:
        img (list of numpy.array): Multiple-view images to be resized.
        lidar2img (list of numpy.array): Transformations from lidar to different cameras.
        rand_scale (float): resize ratio
        interpolation (string): mode for interpolation in opencv.
    Returns:
        imgs_new (list of numpy.array): updated multiple-view images
        lidar2img_new (list of numpy.array): updated transformations from lidar to different 
        cameras.
    """

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
