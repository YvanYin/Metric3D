# ConvNeXt Large
python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/convlarge.0.3_150.py' \
    --load-from ./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth \
    --test_data_path ./data/wild_demo \
    --launcher None \
    --batch_size 2

# ConvNeXt Tiny, note: only trained on outdoor data, perform better in outdoor scenes, such as kitti
python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/convtiny.0.3_150.py' \
    --load-from ./weight/convtiny_hourglass_v1.pth \
    --test_data_path ./data/wild_demo \
    --launcher None \
    --batch_size 2