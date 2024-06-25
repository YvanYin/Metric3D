cd ../../../

python  mono/tools/train.py \
        mono/configs/RAFTDecoder/vit.raft5.large.kitti.py \
        --use-tensorboard \
        --launcher slurm \
        --load-from Path_to_Checkpoint.pth \
        --experiment_name set1
