import random as rd
import cv2 as cv
import numpy as np


# 保存视频
class RecordMovie(object):

    def __init__(self, img_width, img_height):
        self.video_writer = None  # 视频对象
        self.is_end = False  # 结束保存视频
        self.img_width = img_width  # 宽度
        self.img_height = img_height  # 高度

    # 创建 视频写入对象
    def start(self, file_name, freq):
        # 创建视频格式
        four_cc = cv.VideoWriter_fourcc(*'mp4v')
        img_size = (self.img_width, self.img_height)  # 视频尺寸

        # 创建视频写入对象
        self.video_writer = cv.VideoWriter()
        self.video_writer.open(file_name, four_cc, freq, img_size, True)

        # 写入图片帧
    def record(self, img):
        if self.is_end is False:
            self.video_writer.write(img)

    # 完成视频 释放资源
    def end(self):
        self.is_end = True
        self.video_writer.release()


def main():

    img_1 = cv.imread("npys2/0_img.png")
    row = img_1[0].sum(-1) != 765
    cul = img_1[:, 0].sum(-1) != 765
    #img_2 = cv.imread("video/1_lidar.png")
    #height1, width1 = img_1.shape[:2]
    #height2, width2 = img_2.shape[:2]
    #print(img_1.shape, img_2.shape)
    # (2790,1174)

    ##rm = RecordMovie(1400, 550)
    rm = RecordMovie(1920, 1080)
    rm.start("test5.mp4", 12)
    for i in range(30):
        im = cv.imread('./title.png')
        rm.record(im)

    for i in range(0, 465):
        full_frame = np.ones((1080, 1920, 3)) * 255
        im1 = cv.imread("npys2/{i}_img.png".format(i=i))
        im2 = cv.imread("npys2/tmp_{i}.png".format(i=i))
        im1 = im1[cul]
        im1 = im1[:, row]
        im1[256:, :456] = im1[256:, 456:0:-1]
        im1[256:, 456:456+456] = im1[256:, 455+456:455:-1]
        im1[256:, 456+456:] = im1[256:, -1:456+455:-1]


        im2 = cv.resize(im2, (512, 512))
        h, _, _ = im1.shape
        h2, w, _ = im2.shape
        padd = np.ones([h, w, 3]) * 255
        padd[:h2, :w, :] = im2
        im = np.hstack((padd, im1))
        if i == 0: print(im.shape)
        #im = cv.imread(f'{i}_img.png')
        #im = cv.imread(f'npy/tmp{i}.png')

        #cv.imwrite('{i}.png'.format(i=i), im)
        # #im = cv.imread('{i}.png'.format(i=i))
        # from IPython import embed
        # embed()
        # exit()
        im = cv.resize(im, (1920, 522))

        full_frame[279:279+522] = im
        full_frame = full_frame.astype(np.uint8)


        rm.record(full_frame)

    # 5.关闭视频文件
    rm.end()


if __name__ == '__main__':
    main()