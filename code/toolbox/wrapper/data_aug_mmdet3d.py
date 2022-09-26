from mmdet.datasets.builder import PIPELINES
from ..data_aug.transforms import RandomScaleImageMultiViewImage_naive


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(RandomScaleImageMultiViewImage_naive):
    """Random scale the image
    Args:
        scales
    """

    def __call__(self, results, seed=None):
        """Call function to pad images, masks, semantic segmentation maps.
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


# @PIPELINES.register_module()
# class HorizontalRandomFlipMultiViewImage(object):

#     def __init__(self, flip_ratio=0.5, dataset='nuScenes'):
#         self.flip_ratio = flip_ratio
#         self.seed = 0
#         self.dataset = dataset

#     def __call__(self, results, seed=None):
#         if seed is not None: np.random.seed(int(seed))
#         if np.random.rand() >= self.flip_ratio:
#             return results
#         else:
#             results['flip'] = True
#             xxx, xxx, xxx = self.flip_bbox(results['can_bus'])
#             results['xxx'], results['xxx'] = xxx, xxx, xxx
#             results = self.flip_cam_params(results)
#             results['img'] = self.flip_img(results['img'])
#             results['can_bus'] = self.flip_can_bus(results['can_bus'])
#         return results
