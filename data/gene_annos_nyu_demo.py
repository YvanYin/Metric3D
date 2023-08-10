if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import json

    code_root = '/mnt/nas/share/home/xugk/MetricDepth_test/'

    data_root = osp.join(code_root, 'data/nyu_demo')
    split_root = code_root

    files = []
    rgb_root = osp.join(data_root, 'rgb')
    depth_root = osp.join(data_root, 'depth')
    for rgb_file in os.listdir(rgb_root):
        rgb_path = osp.join(rgb_root, rgb_file).split(split_root)[-1]
        depth_path = rgb_path.replace('.jpg', '.png').replace('/rgb_', '/sync_depth_').replace('/rgb/', '/depth/')
        cam_in = [518.8579, 519.46961, 325.58245, 253.73617]
        depth_scale = 1000.

        meta_data = {}
        meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path
        meta_data['depth'] = depth_path
        meta_data['depth_scale'] = depth_scale
        files.append(meta_data)
    files_dict = dict(files=files)

    with open(osp.join(code_root, 'data/nyu_demo/test_annotations.json'), 'w') as f:
        json.dump(files_dict, f)