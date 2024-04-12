import numpy as np

from .config import cfg
from .dropgenerator import generate_label, generateDrops_np


class RainDrop_Augmentor():

    def __init__(self, height, width):
        drops_list, label_map = generate_label(height, width, cfg)
        self.drops_list = drops_list
        self.label_map = label_map

    def generate_one(self, img_np, mode='rgb'):

        assert mode in ['gray', 'rgb']

        # requirement input [H, W, 3]
        if (mode == 'gray'):
            img_np = np.squeeze(img_np)
            img_np = np.expand_dims(img_np, axis=-1)
            img_np = np.repeat(img_np, 3, axis=-1)

        output_img, output_label, mask = generateDrops_np(img_np, cfg, self.drops_list)
        output_img = np.asarray(output_img)

        if (mode == 'gray'):
            output_img = output_img[:, :, 0]

        return output_img
