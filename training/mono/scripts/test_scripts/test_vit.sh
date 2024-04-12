cd ../../../

python  mono/tools/test.py \
        mono/configs/test_configs_vit_small/ibims.vit.dpt.raft.py \
        --load-from vit_small_step00800000.pth
