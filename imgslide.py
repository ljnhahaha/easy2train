import os
import argparse

import matplotlib
matplotlib.use('tkagg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image


pre_split = None


def main(path1: str, path2: str):
    if not os.path.exists(path1):
        raise ValueError(f'{path1} not exists')
    if not os.path.exists(path2):
        raise ValueError(f'{path2} not exists')

    img1 = Image.open(path1)
    img1.convert('RGB')
    img1 = np.array(img1)
    img2 = Image.open(path2)
    img2.convert('RGB')
    img2 = np.array(img2)
    assert img1.shape == img2.shape, f'Input images dont have the same shape. Image1: {img1.shape}, Image2: {img2.shape}'

    h, w = img1.shape[0], img1.shape[1]
    global pre_split
    pre_split = w // 2

    show_img = img1.copy()
    show_img[:, pre_split:, :] = img2[:, pre_split:, :]
    fig = plt.figure()
    plt.subplots_adjust(bottom=0.25)
    ax = fig.subplots()
    im = ax.imshow(show_img)
    ax.axis(False)

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    s_factor = Slider(ax_slide, 'changing factor', 0, 1, valinit=0.5, valstep=0.0001)
    
    def update(val):
        global pre_split
        cur_v = s_factor.val
        split_w = int(w * cur_v)
        if pre_split >= split_w:
            show_img[:, split_w:pre_split, :] = img2[:, split_w:pre_split, :]
        else:
            show_img[:, pre_split:split_w, :] = img1[:, pre_split:split_w, :]
        pre_split = split_w

        im.set_data(show_img)
        # fig.canvas.draw_idle()
        fig.canvas.draw()
    
    s_factor.on_changed(update)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, required=True, help='Path of the first image')
    parser.add_argument('--path2', type=str, required=True, help='Path of the second image')
    args = parser.parse_args()
    main(args.path1, args.path2)
