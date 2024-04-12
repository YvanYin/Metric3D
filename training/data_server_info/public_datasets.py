
db_info={}


#### DDAD Dataset
# RGBD, consecutive frames, and ring cameras annotations
db_info['DDAD']={
    'db_root': 'tbd_data_root',  # Config your data root!
    'data_root': 'DDAD',
    'semantic_root': 'DDAD',
    'meta_data_root': 'DDAD',
    'train_annotations_path': 'DDAD/DDAD/annotations/train.json',
    'test_annotations_path': 'DDAD/DDAD/annotations/test.json',
    'val_annotations_path': 'DDAD/DDAD/annotations/val.json',
}

#### Mapillary Planet Scale Dataset
# Single frame RGBD annotations
db_info['Mapillary_PSD']={
    'db_root': 'tbd_data_root',
    'data_root': 'Mapillary_PSD',
    'semantic_root': 'Mapillary_PSD',
    'train_annotations_path': 'Mapillary_PSD/Mapillary_PSD/annotations/train.json',
    'test_annotations_path': 'Mapillary_PSD/Mapillary_PSD/annotations/test.json',
    'val_annotations_path': 'Mapillary_PSD/Mapillary_PSD/annotations/val.json',
}

#### Cityscapes dataset
# Cityscapes sequence dataset, RGBD and consecutive frames annotations
db_info['Cityscapes_sequence'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Cityscapes_sequence',
    'semantic_root': 'Cityscapes_sequence',
    'train_annotations_path': 'Cityscapes_sequence/Cityscapes_sequence/annotations/train.json',
    'test_annotations_path': 'Cityscapes_sequence/Cityscapes_sequence/annotations/test.json',
    'val_annotations_path': 'Cityscapes_sequence/Cityscapes_sequence/annotations/val.json',
}
# Cityscapes extra dataset, RGBD annotations
db_info['Cityscapes_trainextra'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Cityscapes_trainextra',
    'train_annotations_path': 'Cityscapes_trainextra/Cityscapes_trainextra/annotations/train.json',
    'test_annotations_path': 'Cityscapes_trainextra/Cityscapes_trainextra/annotations/test.json',
    'val_annotations_path': 'Cityscapes_trainextra/Cityscapes_trainextra/annotations/val.json',
}
db_info['Cityscapes_sequence_test'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Cityscapes_sequence',
    'train_annotations_path': 'Cityscapes_sequence/Cityscapes_sequence/annotations/train.json',
    'test_annotations_path': 'Cityscapes_sequence/Cityscapes_sequence/annotations/test.json',
    'val_annotations_path': 'Cityscapes_sequence/Cityscapes_sequence/annotations/test.json',
}

#### Lyft dataset
# Lyft dataset, RGBD, neighbouring cameras, and consecutive frames annotations
db_info['Lyft'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Lyft',
    'depth_root': 'Lyft',
    'meta_data_root': 'Lyft',
    'semantic_root': 'Lyft', 
    'train_annotations_path': 'Lyft/Lyft/annotations/train.json',
    'test_annotations_path': 'Lyft/Lyft/annotations/test.json',
    'val_annotations_path': 'Lyft/Lyft/annotations/val.json',
}
# Lyft dataset, RGBD for ring cameras
db_info['Lyft_ring'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Lyft',
    'depth_root': 'Lyft',
    'meta_data_root': 'Lyft',
    'train_annotations_path': 'Lyft/Lyft/annotations/train.json',
    'test_annotations_path': 'Lyft/Lyft/annotations/test.json',
    'val_annotations_path': 'Lyft/Lyft/annotations/val.json',
}

#### DSEC dataset
# DSEC dataset, RGBD and consecutive frames annotaitons
db_info['DSEC'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'DSEC',
    'semantic_root': 'DSEC', 
    'train_annotations_path': 'DSEC/DSEC/annotations/train.json',
    'test_annotations_path': 'DSEC/DSEC/annotations/test.json',
    'val_annotations_path': 'DSEC/DSEC/annotations/val.json',
}

#### Argovers2 Dataset
# Argovers2 dataset, RGBD and neighbouring cameras annotaitons
db_info['Argovers2'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Argovers2',
    'depth_root': 'Argovers2',
    'meta_data_root': 'Argovers2',
    'train_annotations_path': 'Argovers2/Argovers2/annotations/train.json',
    'test_annotations_path': 'Argovers2/Argovers2/annotations/test.json',
    'val_annotations_path': 'Argovers2/Argovers2/annotations/val.json',
} 
# Argovers2 dataset, RGBD and consecutive cameras annotaitons
db_info['Argovers2_tmpl'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Argovers2',
    'depth_root': 'Argovers2',
    'meta_data_root': 'Argovers2',
    'train_annotations_path': 'Argovers2/Argovers2/annotations/train.json',
    'test_annotations_path': 'Argovers2/Argovers2/annotations/test.json',
    'val_annotations_path': 'Argovers2/Argovers2/annotations/val.json',
} 

#### DrivingStereo Dataset
# DrivingStereo dataset, RGBD annotaitons for stereo data
db_info['DrivingStereo'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'DrivingStereo',
    'semantic_root': 'DrivingStereo',
    'train_annotations_path': 'DrivingStereo/DrivingStereo/annotations/train.json',
    'test_annotations_path': 'DrivingStereo/DrivingStereo/annotations/test.json',
    'val_annotations_path': 'DrivingStereo/DrivingStereo/annotations/val.json',
} 
# DrivingStereo dataset, RGBD and consecutive frames annotaitons for stereo data
db_info['DrivingStereo_tmpl'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'DrivingStereo',
    'semantic_root': 'DrivingStereo',
    'train_annotations_path': 'DrivingStereo/DrivingStereo/annotations/train.json',
    'test_annotations_path': 'DrivingStereo/DrivingStereo/annotations/test.json',
    'val_annotations_path': 'DrivingStereo/DrivingStereo/annotations/val.json',
} 

#### DIML Dataset
# DIML dataset, RGBD annotaitons for stereo data
db_info['DIML'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'DIML',
    'semantic_root': 'DIML',
    'train_annotations_path': 'DIML/DIML/anotation/train.json',
    'test_annotations_path': 'DIML/DIML/anotation/test.json',
    'val_annotations_path': 'DIML/DIML/anotation/val.json',
} 

db_info['NuScenes'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'NuScenes',
    'train_annotations_path': 'NuScenes/NuScenes/annotations/train.json',
    'test_annotations_path': 'NuScenes/NuScenes/annotations/test.json',
    'val_annotations_path': 'NuScenes/NuScenes/annotations/val.json',
} 
db_info['NuScenes_tmpl'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'NuScenes',
    'train_annotations_path': 'NuScenes/NuScenes/annotations/train.json',
    'test_annotations_path': 'NuScenes/NuScenes/annotations/test.json',
    'val_annotations_path': 'NuScenes/NuScenes/annotations/val.json',
} 


# Pandaset, RGBD + tmpl dataset
db_info['Pandaset'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Pandaset',
    'meta_data_root': 'Pandaset',
    'semantic_root': 'Pandaset',
    'train_annotations_path': 'Pandaset/Pandaset/annotations/train.json',
    'test_annotations_path': 'Pandaset/Pandaset/annotations/test.json',
    'val_annotations_path': 'Pandaset/Pandaset/annotations/val.json',
}
db_info['Pandaset_ring'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Pandaset',
    'meta_data_root': 'Pandaset',
    'semantic_root': 'Pandaset',
    'train_annotations_path': 'Pandaset/Pandaset/annotations/train.json',
    'test_annotations_path': 'Pandaset/Pandaset/annotations/test.json',
    'val_annotations_path': 'Pandaset/Pandaset/annotations/val.json',
}

# UASOL, RGBD + tmpl dataset
db_info['UASOL'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'UASOL_data',
    'meta_data_root': 'UASOL_data',
    'semantic_root': 'UASOL_data',
    'train_annotations_path': 'UASOL_data/UASOL_data/annotations/train.json', 
    'test_annotations_path': 'UASOL_data/UASOL_data/annotations/test.json', 
    'val_annotations_path': 'UASOL_data/UASOL_data/annotations/test.json',
}

# Taskonomy, RGBD dataset
db_info['Taskonomy'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Taskonomy',
    'meta_data_root': 'Taskonomy',
    'semantic_root': 'Taskonomy',
    'normal_root': 'Taskonomy',

    'train_annotations_path': 'Taskonomy/Taskonomy/annotations/train.json',
    'test_annotations_path': 'Taskonomy/Taskonomy/annotations/test.json',
    'val_annotations_path': 'Taskonomy/Taskonomy/annotations/test.json',
}

### WebStereo Datasets
# HRWSI/Holopix dataset, RGBD and sky masks annotations
db_info['HRWSI_Holopix'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'WebStereo',
    'train_annotations_path': 'WebStereo/annotations/train.json',
    'test_annotations_path': 'WebStereo/annotations/test.json',
    'val_annotations_path': 'WebStereo/annotations/val.json',
}

### Waymo Datasets
db_info['Waymo'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Waymo',
    'meta_data_root': 'Waymo',
    'semantic_root': 'Waymo',
    'train_annotations_path': 'Waymo/Waymo/annotations/training_annos_all_filter.json',
    'test_annotations_path': 'Waymo/Waymo/annotations/testing_annos_all_filter.json',
    'val_annotations_path': 'Waymo/Waymo/annotations/validation_annos_all_filter.json',
}


# DIODE, RGBD dataset
db_info['DIODE'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'DIODE',
    'depth_mask_root': 'DIODE',
    'normal_root': 'DIODE',
    'train_annotations_path': 'DIODE/DIODE/annotations/train.json',
    'test_annotations_path': 'DIODE/DIODE/annotations/test.json',
    'val_annotations_path': 'DIODE/DIODE/annotations/val.json',
}
db_info['DIODE_indoor'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'DIODE',
    'depth_mask_root': 'DIODE',
    'train_annotations_path': 'DIODE/DIODE/annotations/train.json',
    'test_annotations_path': 'DIODE/DIODE/annotations/test.json',
    'val_annotations_path': 'DIODE/DIODE/annotations/val.json',
}
db_info['DIODE_outdoor'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'DIODE',
    'depth_mask_root': 'DIODE',
    'normal_root': 'DIODE',
    'train_annotations_path': 'DIODE/DIODE/annotations/train.json',
    'test_annotations_path': 'DIODE/DIODE/annotations/test.json',
    'val_annotations_path': 'DIODE/DIODE/annotations/val.json',
}
db_info['ETH3D'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'ETH3D',
    'depth_mask_root': 'ETH3D',
    'train_annotations_path': 'ETH3D/ETH3D/annotations/test.json',
    'test_annotations_path': 'ETH3D/ETH3D/annotations/test.json',
    'val_annotations_path': 'ETH3D/ETH3D/annotations/test.json',
}
# NYU, RGBD dataset
db_info['NYU'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'NYU',
    'normal_root': 'NYU',
    #'train_annotations_path': 'NYU/NYU/annotations/train.json',
    'train_annotations_path': 'NYU/NYU/annotations/train_normal.json',
    #'test_annotations_path': 'NYU/NYU/annotations/test.json',
    'test_annotations_path': 'NYU/NYU/annotations/test_normal.json',
    'val_annotations_path': 'NYU/NYU/annotations/test.json',
}
# ScanNet, RGBD dataset
db_info['ScanNet'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'ScanNet',
    'train_annotations_path': 'ScanNet/ScanNet/annotations/train.json',
    'test_annotations_path': 'ScanNet/ScanNet/annotations/test.json',
    'val_annotations_path': 'ScanNet/ScanNet/annotations/test.json',
}
# KITTI, RGBD dataset
db_info['KITTI'] = {
    'db_root': 'tbd_data_root',
    'data_root': '',
    'train_annotations_path': 'KITTI/KITTI/annotations/eigen_train.json',
    'test_annotations_path': 'KITTI/KITTI/annotations/eigen_test.json',
    'val_annotations_path': 'KITTI/KITTI/annotations/eigen_test.json',
}


########### new training data
# Blended_mvg, RGBD dataset
db_info['BlendedMVG_omni'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Blended_mvg',
    'meta_data_root': 'Blended_mvg',
    'normal_root': 'Blended_mvg',
    'train_annotations_path': 'Blended_mvg/Blended_mvg/annotations/train.json',
    'test_annotations_path': 'Blended_mvg/Blended_mvg/annotations/test.json',
    'val_annotations_path': 'Blended_mvg/Blended_mvg/annotations/val.json',
}

# HM3D, RGBD dataset
db_info['HM3D'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'HM3D',
    'meta_data_root': 'HM3D',
    'normal_root': 'HM3D',
    'train_annotations_path': 'HM3D/HM3d_omnidata/annotations/train.json', #',
    'test_annotations_path': 'HM3D/HM3d_omnidata/annotations/val.json',
    'val_annotations_path': 'HM3D/HM3d_omnidata/annotations/test.json',
}

# LeddarPixSet, RGBD dataset, some errors in the data
db_info['LeddarPixSet'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'LeddarPixSet',
    'meta_data_root': 'LeddarPixSet',
    'train_annotations_path': 'LeddarPixSet/LeddarPixSet/annotations/train.json',
    'test_annotations_path': 'LeddarPixSet/LeddarPixSet/annotations/test.json',
    'val_annotations_path': 'LeddarPixSet/LeddarPixSet/annotations/val.json',
}

# RGBD dataset
db_info['Replica'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Replica',
    'meta_data_root': 'Replica',
    'normal_root': 'Replica',
    'train_annotations_path': 'Replica/replica/annotations/train.json',
    'test_annotations_path': 'Replica/replica/annotations/test.json',
    'val_annotations_path': 'Replica/replica/annotations/val.json',
}

db_info['Replica_gso'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Replica',
    'meta_data_root': 'Replica',
    'normal_root': 'Replica',
    'train_annotations_path': 'Replica/replica_gso/annotations/train.json',
    'test_annotations_path': 'Replica/replica_gso/annotations/test.json',
    'val_annotations_path': 'Replica/replica_gso/annotations/val.json',
}

db_info['Matterport3D'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'Matterport3D',
    'meta_data_root': 'Matterport3D',
    'normal_root': 'Matterport3D',
    'train_annotations_path': 'Matterport3D/Matterport3D/annotations/train.json',
    'test_annotations_path': 'Matterport3D/Matterport3D/annotations/test.json',
    'val_annotations_path': 'Matterport3D/Matterport3D/annotations/test.json',
}

db_info['S3DIS'] = {
    'db_root': 'tbd_data_root',
    'data_root': 's3dis',
    'meta_data_root': 's3dis',
    'normal_root': 's3dis',
    'train_annotations_path': 's3dis/s3dis/annotations/train.json',
    'test_annotations_path': 's3dis/s3dis/annotations/test.json',
    'val_annotations_path': 's3dis/s3dis/annotations/test.json',
}

db_info['Seasons4'] = {
    'db_root': 'tbd_data_root',
    'data_root': '4seasons/4seasons',
    'meta_data_root': '4seasons/4seasons',
    'train_annotations_path': '4seasons/4seasons/annotations/train.json',
    'test_annotations_path': '4seasons/4seasons/annotations/test.json',
    'val_annotations_path': '4seasons/4seasons/annotations/test.json',
}

db_info['Virtual_KITTI'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'virtual_kitti',
    'meta_data_root': 'virtual_kitti',
    'semantic_root': 'virtual_kitti',
    'train_annotations_path': 'virtual_kitti/virtual_kitti/annotations/train.json',
    'test_annotations_path': 'virtual_kitti/virtual_kitti/annotations/test.json',
    'val_annotations_path': 'virtual_kitti/virtual_kitti/annotations/test.json',
}

db_info['IBIMS'] = {
    'db_root': 'tbd_data_root',
    'data_root': '',
    'train_annotations_path': 'iBims-1/annotations/train.json',
    'test_annotations_path': 'iBims-1/annotations/test.json',
    'val_annotations_path': 'iBims-1/annotations/test.json',
}

db_info['ScanNetAll'] = {
    'db_root': 'tbd_data_root',
    'data_root': 'scannet',
    'normal_root': 'scannet',
    'meta_data_root': 'scannet',
    'train_annotations_path': 'scannet/scannet/annotations/train.json',
    'test_annotations_path': 'scannet/scannet/annotations/test.json',
    'val_annotations_path': 'scannet/scannet/annotations/test.json',
}

db_info['Hypersim'] = {
    'db_root': 'tbd_data_root',
    'data_root': '',
    'meta_data_root': '',
    'normal_root': '',
    # 'semantic_root': '', # Semantic tags without sky, see https://github.com/apple/ml-hypersim/blob/main/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv
    'train_annotations_path': 'Hypersim/annotations/train.json',
    'test_annotations_path': 'Hypersim/annotations/test.json',
    'val_annotations_path': 'Hypersim/annotations/test.json',
}

db_info['DIML_indoor'] = {
    'db_root': 'tbd_data_root',
    'data_root': '',
    # 'semantic_root': '',
    'train_annotations_path': 'DIML_indoor_new/annotations/train.json',
    'test_annotations_path': 'DIML_indoor_new/annotations/test.json',
    'val_annotations_path': 'DIML_indoor_new/annotations/test.json',
} 