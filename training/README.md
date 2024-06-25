# Training

**Re-implemented training codes in public environments by @JUGGHM**   

This is an re-implemented and verified version of the original training codes in private environments. Codes for overall framework, dataloaders, and losses are kept.
However, we cannot provide the annotations ```json``` currently due to IP issues. 

You can either integrate our framework into your own codes (Recommanded), or prepare the datasets as following (Needs many efforts). 

### Config the pretrained checkpoints for ConvNeXt and DINOv2
Download the checkpoints and config the paths in ```data_server_info/pretrained_weight.py```

### Prepare the json files
Prepare json files for different datasets in ```data_server_info/public_datasets.py```. Some tiny examples are also provided in ```data_server_info/annos*.json```. 

### Train
```bash mono/scripts/training_scripts/train.sh```

### Example: Finetune on KITTI
We just re-implemented the [json generating script](./kitti_json_files/generate_json.py) in public environments. Users can finetune the pretrained models on KITTI now with [eigen_split json](./kitti_json_files/eigen_train.json).

Step1: Configure the path for KITTI Raw dataset, KITTI Depth dataset, and the provided json file in ```data_server_info/public_datasets.py``` correctly.

Step2: Start training by 
```bash mono/scripts/training_scripts/train_kitti.sh```

