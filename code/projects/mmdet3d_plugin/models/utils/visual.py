import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid
import cv2


def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()

    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor*255
    if len(tensor.shape) == 3:

        c, h, w = tensor.shape
        tensor = tensor.reshape(c, h*w)
        max_t = tensor.max(-1).values
        min_t = tensor.min(-1).values
        tensor = (tensor - min_t[..., None]) / (max_t - min_t)[..., None]
        tensor = tensor.reshape(c, h, w)
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=False).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)
