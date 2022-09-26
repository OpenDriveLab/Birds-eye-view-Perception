from typing import Dict
from mmdet.datasets.builder import PIPELINES
from ..data_aug.transforms import RandomScaleImageMultiViewImage_naive


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(RandomScaleImageMultiViewImage_naive):
    """Resize the multiple-view images with the same scale selected randomly.  .
    Wrapper for mmdet3d
    Args:
        scales (tuple of float): ratio for resizing the images. Every time, select one ratio randonly.
    """

    def __call__(self, results: Dict, seed=None) -> Dict:
        """Call function to randomly resize multiple-view images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        imgs, lidar2img = results['img'], results['lidar2img']
        imgs_new, lidar2img_new = self.forward(imgs, lidar2img, seed=seed)
        results['img'] = imgs_new
        results['lidar2img'] = lidar2img_new
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results
