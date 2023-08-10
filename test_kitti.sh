python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/test_kitti_convlarge_hourglass_0.3_150.py' \
    --load-from ./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth \
    --test_data_path ./data/kitti_demo/test_annotations.json \
    --launcher None