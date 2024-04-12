# change rainy drop func from
# https://github.com/EvoCargo/RaindropsOnWindshield/blob/main/raindrops_generator/raindrop/dropgenerator.py

import math
import random
from random import randint

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from skimage.measure import label as skimage_label

from .raindrop import Raindrop, make_bezier


def CheckCollision(DropList):
    """This function handle the collision of the drops.

    :param DropList: list of raindrop class objects
    """
    listFinalDrops = []
    Checked_list = []
    list_len = len(DropList)
    # because latter raindrops in raindrop list should has more colision information
    # so reverse list
    DropList.reverse()
    drop_key = 1
    for drop in DropList:
        # if the drop has not been handle
        if drop.getKey() not in Checked_list:
            # if drop has collision with other drops
            if drop.getIfColli():
                # get collision list
                collision_list = drop.getCollisionList()
                # first get radius and center to decide how  will the collision do
                final_x = drop.getCenters()[0] * drop.getRadius()
                final_y = drop.getCenters()[1] * drop.getRadius()
                tmp_devide = drop.getRadius()
                final_R = drop.getRadius() * drop.getRadius()
                for col_id in collision_list:
                    col_id = int(col_id)
                    Checked_list.append(col_id)
                    # list start from 0
                    final_x += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[0]
                    final_y += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[1]
                    tmp_devide += DropList[list_len - col_id].getRadius()
                    final_R += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getRadius()
                final_x = int(round(final_x / tmp_devide))
                final_y = int(round(final_y / tmp_devide))
                final_R = int(round(math.sqrt(final_R)))
                # rebuild drop after handled the collisions
                newDrop = Raindrop(drop_key, (final_x, final_y), final_R)
                drop_key = drop_key + 1
                listFinalDrops.append(newDrop)
            # no collision
            else:
                drop.setKey(drop_key)
                drop_key = drop_key + 1
                listFinalDrops.append(drop)

    return listFinalDrops


def generate_label(h, w, cfg):
    """This function generate list of raindrop class objects and label map of
    this drops in the image.

    :param h: image height
    :param w: image width
    :param cfg: config with global constants
    :param shape: int from 0 to 2 defining raindrop shape type
    """
    maxDrop = cfg['maxDrops']
    minDrop = cfg['minDrops']
    maxR = cfg['maxR']
    minR = cfg['minR']
    drop_num = randint(minDrop, maxDrop)
    imgh = h
    imgw = w
    # random drops position
    ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(drop_num)]
    listRainDrops = []
    listFinalDrops = []
    for key, pos in enumerate(ran_pos):
        key = key + 1
        radius = random.randint(minR, maxR)
        shape = random.randint(1, 1)
        drop = Raindrop(key, pos, radius, shape)
        listRainDrops.append(drop)
# to check if collision or not
    label_map = np.zeros([h, w])
    collisionNum = len(listRainDrops)
    listFinalDrops = list(listRainDrops)
    loop = 0
    while collisionNum > 0:
        loop = loop + 1
        listFinalDrops = list(listFinalDrops)
        collisionNum = len(listFinalDrops)
        label_map = np.zeros_like(label_map)
        # Check Collision
        for drop in listFinalDrops:
            # check the bounding
            (ix, iy) = drop.getCenters()
            radius = drop.getRadius()
            ROI_WL = 2 * radius
            ROI_WR = 2 * radius
            ROI_HU = 3 * radius
            ROI_HD = 2 * radius
            if (iy - 3 * radius) < 0:
                ROI_HU = iy
            if (iy + 2 * radius) > imgh:
                ROI_HD = imgh - iy
            if (ix - 2 * radius) < 0:
                ROI_WL = ix
            if (ix + 2 * radius) > imgw:
                ROI_WR = imgw - ix


# apply raindrop label map to Image's label map
            drop_label = drop.getLabelMap()
            # check if center has already has drops
            if (label_map[iy, ix] > 0):
                col_ids = np.unique(label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL:ix + ROI_WR])
                col_ids = col_ids[col_ids != 0]
                drop.setCollision(True, col_ids)
                label_map[iy - ROI_HU:iy + ROI_HD,
                          ix - ROI_WL:ix + ROI_WR] = drop_label[3 * radius - ROI_HU:3 * radius + ROI_HD, 2 * radius -
                                                                ROI_WL:2 * radius + ROI_WR] * drop.getKey()
            else:
                label_map[iy - ROI_HU:iy + ROI_HD,
                          ix - ROI_WL:ix + ROI_WR] = drop_label[3 * radius - ROI_HU:3 * radius + ROI_HD, 2 * radius -
                                                                ROI_WL:2 * radius + ROI_WR] * drop.getKey()
                # no collision
                collisionNum = collisionNum - 1

        if collisionNum > 0:
            listFinalDrops = CheckCollision(listFinalDrops)
    return listFinalDrops, label_map


def generateDrops(imagePath, cfg, listFinalDrops):
    """Generate raindrops on the image.

    :param imagePath: path to the image on which you want to generate drops
    :param cfg: config with global constants
    :param listFinalDrops: final list of raindrop class objects after handling collisions
    :param label_map: general label map of all drops in the image
    """
    ifReturnLabel = cfg['return_label']
    edge_ratio = cfg['edge_darkratio']

    PIL_bg_img = Image.open(imagePath).convert('RGB')
    bg_img = np.asarray(PIL_bg_img)
    label_map = np.zeros_like(bg_img)[:, :, 0]
    imgh, imgw, _ = bg_img.shape

    A = cfg['A']
    B = cfg['B']
    C = cfg['C']
    D = cfg['D']

    alpha_map = np.zeros_like(label_map).astype(np.float64)

    for drop in listFinalDrops:
        (ix, iy) = drop.getCenters()
        radius = drop.getRadius()
        ROI_WL = 2 * radius
        ROI_WR = 2 * radius
        ROI_HU = 3 * radius
        ROI_HD = 2 * radius
        if (iy - 3 * radius) < 0:
            ROI_HU = iy
        if (iy + 2 * radius) > imgh:
            ROI_HD = imgh - iy
        if (ix - 2 * radius) < 0:
            ROI_WL = ix
        if (ix + 2 * radius) > imgw:
            ROI_WR = imgw - ix

        drop_alpha = drop.getAlphaMap()
        alpha_map[iy - ROI_HU:iy + ROI_HD,
                  ix - ROI_WL:ix + ROI_WR] += drop_alpha[3 * radius - ROI_HU:3 * radius + ROI_HD,
                                                         2 * radius - ROI_WL:2 * radius + ROI_WR]

    alpha_map = alpha_map / np.max(alpha_map) * 255.0

    PIL_bg_img = Image.open(imagePath)
    for idx, drop in enumerate(listFinalDrops):
        (ix, iy) = drop.getCenters()
        radius = drop.getRadius()
        ROIU = iy - 3 * radius
        ROID = iy + 2 * radius
        ROIL = ix - 2 * radius
        ROIR = ix + 2 * radius
        if (iy - 3 * radius) < 0:
            ROIU = 0
            ROID = 5 * radius
        if (iy + 2 * radius) > imgh:
            ROIU = imgh - 5 * radius
            ROID = imgh
        if (ix - 2 * radius) < 0:
            ROIL = 0
            ROIR = 4 * radius
        if (ix + 2 * radius) > imgw:
            ROIL = imgw - 4 * radius
            ROIR = imgw

        tmp_bg = bg_img[ROIU:ROID, ROIL:ROIR, :]
        try:
            drop.updateTexture(tmp_bg)
        except Exception:
            del listFinalDrops[idx]
            continue
        tmp_alpha_map = alpha_map[ROIU:ROID, ROIL:ROIR]

        output = drop.getTexture()
        tmp_output = np.asarray(output).astype(np.float)[:, :, -1]
        tmp_alpha_map = tmp_alpha_map * (tmp_output / 255)
        tmp_alpha_map = Image.fromarray(tmp_alpha_map.astype('uint8'))

        edge = ImageEnhance.Brightness(output)
        edge = edge.enhance(edge_ratio)

        PIL_bg_img.paste(edge, (ix - 2 * radius, iy - 3 * radius), output)
        PIL_bg_img.paste(output, (ix - 2 * radius, iy - 3 * radius), output)

    mask = np.zeros_like(bg_img)

    if len(listFinalDrops) > 0:
        # make circles and elipses
        for drop in listFinalDrops:
            if (drop.shape == 0):
                cv2.circle(mask, drop.center, drop.radius, (255, 255, 255), -1)
            if (drop.shape == 1):
                cv2.circle(mask, drop.center, drop.radius, (255, 255, 255), -1)
                cv2.ellipse(mask, drop.center, (drop.radius, int(1.3 * math.sqrt(3) * drop.radius)), 0, 180, 360,
                            (255, 255, 255), -1)

        img = Image.fromarray(np.uint8(mask[:, :, 0]), 'L')
        # make beziers
        for drop in listFinalDrops:
            if (drop.shape == 2):
                img = Image.fromarray(np.uint8(img), 'L')
                draw = ImageDraw.Draw(img)
                ts = [t / 100.0 for t in range(101)]
                xys = [(drop.radius * C[0] - 2 * drop.radius + drop.center[0],
                        drop.radius * C[1] - 3 * drop.radius + drop.center[1]),
                       (drop.radius * B[0] - 2 * drop.radius + drop.center[0],
                        drop.radius * B[1] - 3 * drop.radius + drop.center[1]),
                       (drop.radius * D[0] - 2 * drop.radius + drop.center[0],
                        drop.radius * D[1] - 3 * drop.radius + drop.center[1])]
                bezier = make_bezier(xys)
                points = bezier(ts)

                xys = [(drop.radius * C[0] - 2 * drop.radius + drop.center[0],
                        drop.radius * C[1] - 3 * drop.radius + drop.center[1]),
                       (drop.radius * A[0] - 2 * drop.radius + drop.center[0],
                        drop.radius * A[1] - 3 * drop.radius + drop.center[1]),
                       (drop.radius * D[0] - 2 * drop.radius + drop.center[0],
                        drop.radius * D[1] - 3 * drop.radius + drop.center[1])]
                bezier = make_bezier(xys)
                points.extend(bezier(ts))
                draw.polygon(points, fill='white')
                mask = np.array(img)

    im_mask = Image.fromarray(mask.astype('uint8'))

    if ifReturnLabel:
        output_label = np.array(alpha_map)
        output_label.flags.writeable = True
        output_label[output_label > 0] = 1
        output_label = Image.fromarray(output_label.astype('uint8'))
        return PIL_bg_img, output_label, im_mask

    return PIL_bg_img


def generateDrops_np(img_np, cfg, listFinalDrops):
    """Generate raindrops on the image.

    :param imgs: numpy imgs shape -> [B, H, W, C], type -> np.uint8
    :param cfg: config with global constants
    :param listFinalDrops: final list of raindrop class objects after handling collisions
    :param label_map: general label map of all drops in the image
    """
    ifReturnLabel = cfg['return_label']
    edge_ratio = cfg['edge_darkratio']

    # PIL_bg_img = Image.open(imagePath)
    # label_map = np.zeros_like(bg_img)[:,:,0]
    # imgh, imgw, _ = bg_img.shape
    bg_img = img_np
    label_map = np.zeros_like(bg_img)[:, :, 0]  # [H, W]
    imgh, imgw, _ = bg_img.shape

    A = cfg['A']
    B = cfg['B']
    C = cfg['C']
    D = cfg['D']

    # 0. generate alpha change map by generated list raindrops
    alpha_map = np.zeros_like(label_map).astype(np.float64)  # [H, W]

    for drop in listFinalDrops:
        (ix, iy) = drop.getCenters()
        radius = drop.getRadius()
        ROI_WL = 2 * radius
        ROI_WR = 2 * radius
        ROI_HU = 3 * radius
        ROI_HD = 2 * radius
        if (iy - 3 * radius) < 0:
            ROI_HU = iy
        if (iy + 2 * radius) > imgh:
            ROI_HD = imgh - iy
        if (ix - 2 * radius) < 0:
            ROI_WL = ix
        if (ix + 2 * radius) > imgw:
            ROI_WR = imgw - ix

        drop_alpha = drop.getAlphaMap()

        alpha_map[iy - ROI_HU:iy + ROI_HD,
                  ix - ROI_WL:ix + ROI_WR] += drop_alpha[3 * radius - ROI_HU:3 * radius + ROI_HD,
                                                         2 * radius - ROI_WL:2 * radius + ROI_WR]

    alpha_map = alpha_map / np.max(alpha_map) * 255.0

    PIL_bg_img = Image.fromarray(np.uint8(bg_img)).convert('RGB')
    # convert
    for idx, drop in enumerate(listFinalDrops):
        (ix, iy) = drop.getCenters()
        radius = drop.getRadius()
        ROIU = iy - 3 * radius
        ROID = iy + 2 * radius
        ROIL = ix - 2 * radius
        ROIR = ix + 2 * radius
        if (iy - 3 * radius) < 0:
            ROIU = 0
            ROID = 5 * radius
        if (iy + 2 * radius) > imgh:
            ROIU = imgh - 5 * radius
            ROID = imgh
        if (ix - 2 * radius) < 0:
            ROIL = 0
            ROIR = 4 * radius
        if (ix + 2 * radius) > imgw:
            ROIL = imgw - 4 * radius
            ROIR = imgw

        tmp_bg = bg_img[ROIU:ROID, ROIL:ROIR]
        try:
            drop.updateTexture(tmp_bg)
        except Exception:
            del listFinalDrops[idx]
            continue
        tmp_alpha_map = alpha_map[ROIU:ROID, ROIL:ROIR]

        output = drop.getTexture()
        tmp_output = np.asarray(output).astype(np.float)[:, :, -1]
        tmp_alpha_map = tmp_alpha_map * (tmp_output / 255)
        tmp_alpha_map = Image.fromarray(tmp_alpha_map.astype('uint8'))

        edge = ImageEnhance.Brightness(output)
        edge = edge.enhance(edge_ratio)

        # PIL_bg_img.paste(edge, (ix-2*radius, iy-3*radius), output)
        # PIL_bg_img.paste(output, (ix-2*radius, iy-3*radius), output)
        PIL_bg_img.paste(edge, (ROIL, ROIU), output)
        PIL_bg_img.paste(output, (ROIL, ROIU), output)


# mask process part
    mask = np.zeros_like(bg_img)

    if len(listFinalDrops) > 0:
        # make circles and elipses
        for drop in listFinalDrops:
            if (drop.shape == 0):
                cv2.circle(mask, drop.center, drop.radius, (255, 255, 255), -1)
            if (drop.shape == 1):
                cv2.circle(mask, drop.center, drop.radius, (255, 255, 255), -1)
                cv2.ellipse(mask, drop.center, (drop.radius, int(1.3 * math.sqrt(3) * drop.radius)), 0, 180, 360,
                            (255, 255, 255), -1)

        img = Image.fromarray(np.uint8(mask[:, :, 0]), 'L')
        # make beziers
        for drop in listFinalDrops:
            if (drop.shape == 2):
                img = Image.fromarray(np.uint8(img), 'L')
                draw = ImageDraw.Draw(img)
                ts = [t / 100.0 for t in range(101)]
                A0, A1 = drop.control_point['A'][0], drop.control_point['A'][1]
                B0, B1 = drop.control_point['B'][0], drop.control_point['B'][1]
                C0, C1 = drop.control_point['C'][0], drop.control_point['C'][1]
                D0, D1 = drop.control_point['D'][0], drop.control_point['D'][1]
                xys = [(drop.radius * C0 - 2 * drop.radius + drop.center[0],
                        drop.radius * C1 - 3 * drop.radius + drop.center[1]),
                       (drop.radius * B0 - 2 * drop.radius + drop.center[0],
                        drop.radius * B1 - 3 * drop.radius + drop.center[1]),
                       (drop.radius * D0 - 2 * drop.radius + drop.center[0],
                        drop.radius * D1 - 3 * drop.radius + drop.center[1])]
                bezier = make_bezier(xys)
                points = bezier(ts)

                xys = [(drop.radius * C0 - 2 * drop.radius + drop.center[0],
                        drop.radius * C1 - 3 * drop.radius + drop.center[1]),
                       (drop.radius * A0 - 2 * drop.radius + drop.center[0],
                        drop.radius * A1 - 3 * drop.radius + drop.center[1]),
                       (drop.radius * D0 - 2 * drop.radius + drop.center[0],
                        drop.radius * D1 - 3 * drop.radius + drop.center[1])]
                bezier = make_bezier(xys)
                points.extend(bezier(ts))
                draw.polygon(points, fill='white')
                mask = np.array(img)

    im_mask = Image.fromarray(mask.astype('uint8'))

    if ifReturnLabel:
        output_label = np.array(alpha_map)
        output_label.flags.writeable = True
        output_label[output_label > 0] = 1
        output_label = Image.fromarray(output_label.astype('uint8'))
        return PIL_bg_img, output_label, im_mask

    return PIL_bg_img
