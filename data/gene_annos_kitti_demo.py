if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import json

    code_root = '/mnt/nas/share/home/xugk/MetricDepth_test/'

    data_root = osp.join(code_root, 'data/kitti_demo')
    split_root = code_root

    files = []
    rgb_root = osp.join(data_root, 'rgb')
    depth_root = osp.join(data_root, 'depth')
    for rgb_file in os.listdir(rgb_root):
        rgb_path = osp.join(rgb_root, rgb_file).split(split_root)[-1]
        depth_path = rgb_path.replace('/rgb/', '/depth/')
        cam_in = [707.0493, 707.0493, 604.0814, 180.5066]
        depth_scale = 256.

        meta_data = {}
        meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path
        meta_data['depth'] = depth_path
        meta_data['depth_scale'] = depth_scale
        files.append(meta_data)
    files_dict = dict(files=files)

    with open(osp.join(code_root, 'data/kitti_demo/test_annotations.json'), 'w') as f:
        json.dump(files_dict, f)
        