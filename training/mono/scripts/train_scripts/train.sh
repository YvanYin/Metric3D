cd ../../../

python  mono/tools/train.py \
        mono/configs/RAFTDecoder/vit.raft5.small.sanity_check.py \
        --use-tensorboard \
        --launcher slurm \
        --experiment_name set1
