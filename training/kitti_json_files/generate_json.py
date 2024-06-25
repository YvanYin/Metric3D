import os
import json

common_root = 'd:/Datasets'
depth_root = 'kitti/kitti_depth/depth/data_depth_annotated'
raw_root = 'kitti_raw/kitti_raw'

#print(os.listdir(os.path.join(common_root, raw_root)))

mid = 'proj_depth/groundtruth'
mid_raw = 'data'

test_file_dict = {}
test_file_list = []
train_file_dict = {}


with open('D:/Datasets/eigen_train.txt') as f:
    lines_train = f.readlines()

cnt = 0
invalid_cnt = 0

if True:
    lines = lines_train

    for l in lines:
        l_ls = l.split(' ')
        scene = l_ls[0]
        date = scene.split('/')[0]
        scene_no_date = scene.split('/')[1]
        frame = l_ls[1]
        frame = frame.zfill(10)

        if 'l' in l_ls[2]:
            cam = 'image_02'
            P_str = 'P_rect_02'
        elif 'r' in l_ls[2]:
            cam = 'image_03'
            P_str = 'P_rect_03'
        else:
            raise NotImplementedError()

        depth_train = os.path.join(depth_root, 'train', scene_no_date, mid, cam , frame + '.png')
        depth_val = os.path.join(depth_root, 'val', scene_no_date, mid, cam, frame+'.png')
        rgb = os.path.join(raw_root, scene, cam, mid_raw, frame+'.png')

        with open(os.path.join(common_root, raw_root, date, 'calib_cam_to_cam.txt')) as c:
            lines_c = c.readlines()

            for l_c in lines_c:
                if P_str in l_c:
                    k_str = l_c.split(':')[1:]
                    k = k_str[0].split(' ')
                    cam_in = [float(k[1]), float(k[6]), float(k[3]), float(k[7])]

        rgb_path = os.path.join(common_root, rgb)
        assert  os.path.join(common_root, rgb_path)

        if os.path.exists(os.path.join(common_root, depth_train)):
            depth_path = os.path.join(common_root, depth_train)
            depth_rel = depth_train

        else:
            depth_path = os.path.join(common_root, depth_val)
            depth_rel = depth_val

        try:
            assert os.path.exists(depth_path)
            cnt += 1
        except:
            invalid_cnt += 1
            continue

        curr_file = [{'rgb':rgb.replace("\\", '/'), 'depth':depth_rel.replace("\\", '/'), 'cam_in':cam_in}]
        test_file_list = test_file_list + curr_file

        if ((cnt + invalid_cnt) % 1000 == 0):
            print(cnt + invalid_cnt)

print(cnt, invalid_cnt)

train_file_dict['files'] = test_file_list
with open('eigen_train.json', 'w') as fj:
    json.dump(train_file_dict, fj)