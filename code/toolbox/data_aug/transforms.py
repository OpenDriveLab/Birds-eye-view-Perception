from typing import List, Tuple
import numpy as np
from .functional import scale_image_multiple_view


class RandomScaleImageMultiViewImage_naive(object):
    """Resize the multiple-view images with the same scale selected randomly.  .
    Args:
        scales (tuple of float): ratio for resizing the images. Every time, select one ratio 
        randomly.
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales
        self.seed = 0

    def forward(self,
                imgs: List[np.ndarray],
                lidar2img: List[np.ndarray],
                seed=None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Args:
            img (list of numpy.array): Multiple-view images to be resized.
            lidar2img (list of numpy.array): Transformations from lidar to different cameras.
        Returns:
            imgs_new (list of numpy.array): updated multiple-view images
            lidar2img_new (list of numpy.array): updated transformations from lidar to different 
            cameras.
        """
        if seed is not None:
            np.random.seed(int(seed))
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        imgs_new, lidar2img_new = scale_image_multiple_view(imgs, lidar2img, rand_scale)

        return imgs_new, lidar2img_new

    def __call__(self,
                 imgs: List[np.ndarray],
                 lidar2img: List[np.ndarray],
                 seed=None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.forward(imgs, lidar2img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str
