# 🚀 Metric3D Project 🚀

**Official PyTorch implementation of "Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image" and "Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation"**

<a href='https://jugghm.github.io/Metric3Dv2'><img src='https://img.shields.io/badge/project%20page-@Metric3D-yellow.svg'></a>
<a href='https://arxiv.org/abs/2307.08695'><img src='https://img.shields.io/badge/arxiv-@Metric3Dv1-green'></a>
<a href='https:'><img src='https://img.shields.io/badge/arxiv (on hold)-@Metric3Dv2-red'></a>
<a href='https://huggingface.co/spaces/JUGGHM/Metric3D'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

[//]: # (### [Project Page]&#40;https://arxiv.org/abs/2307.08695&#41; | [v2 Paper]&#40;https://arxiv.org/abs/2307.10984&#41; | [v1 Arxiv]&#40;https://arxiv.org/abs/2307.10984&#41; | [Video]&#40;https://www.youtube.com/playlist?list=PLEuyXJsWqUNd04nwfm9gFBw5FVbcaQPl3&#41; | [Hugging Face 🤗]&#40;https://huggingface.co/spaces/JUGGHM/Metric3D&#41; )

## News and TO DO LIST

[//]: # (- [ ] Release training codes)
- [ ] Droid slam codes
- [ ] Release the ViT-giant2 model
- [ ] Focal length free mode
- [ ] Floating noise removing mode
- [ ] Improving HuggingFace Demo and Visualization 

- `[2024/3/18]` HuggingFace GPU version updated!
- `[2024/3/18]` [Project page](https://jugghm.github.io/Metric3Dv2/) released!
- `[2024/3/18]` Metric3D V2 models released, supporting metric depth and surface normal now!
- `[2023/8/10]` Inference codes, pretrained weights, and demo released.
- `[2023/7]` Metric3D accepted by ICCV 2023!
- `[2023/4]` The Champion of [2nd Monocular Depth Estimation Challenge](https://jspenmar.github.io/MDEC) in CVPR 2023

##  🌼 Abstract
We present Metric3Dv2, a versatile geometric foundation model for zero-shot metric depth and surface normal estimation.

![page2](media/screenshots/page2.png)



##  📝 Benchmarks 

### Metric Depth

[//]: # (#### Zero-shot Testing)

[//]: # (Our models work well on both indoor and outdoor scenarios, compared with other zero-shot metric depth estimation methods.)

[//]: # ()
[//]: # (|                 | Backbone   | KITTI $\delta 1$ ↑ | KITTI $\delta 2$  ↑ | KITTI $\delta 3$ ↑ | KITTI AbsRel  ↓ | KITTI RMSE  ↓ | KITTI RMS_log  ↓ | NYU $\delta 1$ ↑ | NYU $\delta 2$ ↑ | NYU $\delta 3$ ↑ | NYU AbsRel  ↓ | NYU RMSE  ↓ | NYU log10  ↓ |)

[//]: # (|-----------------|------------|--------------------|---------------------|--------------------|-----------------|---------------|------------------|------------------|------------------|------------------|---------------|-------------|--------------|)

[//]: # (| ZeroDepth       | ResNet-18 | 0.910              | 0.980               | 0.996              | 0.057           | 4.044         | 0.083            | 0.901            | 0.961            | -                | 0.100         | 0.380       | -            |)

[//]: # (| PolyMax         | ConvNeXt-L    | -                  | -                   | -                  | -               | -             | -                | 0.969            | 0.996            | 0.999            | 0.067         | 0.250       | 0.033        |)

[//]: # (| Ours | ViT-L     | 0.985              | 0.995               | 0.999              | 0.052           | 2.511         | 0.074            | 0.975            | 0.994            | 0.998            | 0.063         | 0.251       | 0.028        |)

[//]: # (| Ours | ViT-g2    | 0.989              | 0.996               | 0.999              | 0.051           | 2.403         | 0.080            | 0.980            | 0.997            | 0.999            | 0.067         | 0.260       | 0.030        |)

[//]: # ()
[//]: # ([//]: # &#40;| Adabins | Efficient-B5 | 0.964 | 0.995 | 0.999 | 0.058  |  2.360 | 0.088            | 0.903  | 0.984  | 0.997  | 0.103  | 0.0444  | 0.364 |&#41;)
[//]: # ([//]: # &#40;| NewCRFs | SwinT-L | 0.974 | 0.997 | 0.999 | 0.052  |  2.129 | 0.079            | 0.922  | 0.983  | 0.994  | 0.095  | 0.041  | 0.334 |&#41;)
[//]: # ([//]: # &#40;| Ours &#40;CSTM_label&#41; | ConvNeXt-L |      0.964      | 0.993   | 0.998  | 0.058 | 2.770  | 0.092            | 0.944  |  0.986 | 0.995   | 0.083  |  0.035 |  0.310 |&#41;)

[//]: # (#### Finetuned)
Our models rank 1st on the routing KITTI and NYU benchmarks.

|               | Backbone    | KITTI δ1 ↑ | KITTI δ2  ↑  | KITTI AbsRel  ↓ | KITTI RMSE  ↓ | KITTI RMS_log  ↓ | NYU δ1 ↑ | NYU δ2 ↑  | NYU AbsRel  ↓ | NYU RMSE  ↓ | NYU log10  ↓ |
|---------------|-------------|------------|-------------|-----------------|---------------|------------------|----------|----------|---------------|-------------|--------------|
| ZoeDepth      | ViT-Large   | 0.971      | 0.995                  | 0.053           | 2.281         | 0.082            | 0.953    | 0.995        | 0.077         | 0.277       | 0.033        |
| ZeroDepth     | ResNet-18   | 0.968      | 0.996                   | 0.057           | 2.087         | 0.083            | 0.954    | 0.995           | 0.074         | 0.269       | 0.103        |
| IEBins        | SwinT-Large | 0.978      | 0.998                  | 0.050           | 2.011         | 0.075            | 0.936    | 0.992           | 0.087         | 0.314       | 0.031        |
| DepthAnything | ViT-Large   | 0.982      | 0.998                  | 0.046           | 1.985         | 0.069            | 0.984    | 0.998           | 0.056         | 0.206       | 0.024        |
| Ours          | ViT-Large   | 0.985      | 0.998       | 0.999                        | 1.985         | 0.064            | 0.989    | 0.998           | 0.047         | 0.183       | 0.020        |
| Ours          | ViT-giant2  | 0.989      | 0.998       | 1.000                        | 1.766         | 0.060            | 0.987    | 0.997           | 0.045         | 0.187       | 0.015        |

### Affine-invariant Depth
Even compared to recent affine-invariant depth methods (Marigold and Depth Anything), our metric-depth (and normal) models still show superior performance. 

|                       | #Data for Pretrain and Train                 | KITTI Absrel ↓ | KITTI δ1 ↑ | NYUv2 AbsRel  ↓ | NYUv2 δ1 ↑ | DIODE-Full AbsRel ↓ | DIODE-Full δ1 ↑ | Eth3d AbsRel  ↓ | Eth3d δ1 ↑ |
|-----------------------|----------------------------------------------|----------------|------------|-----------------|------------|---------------------|-----------------|----------------------|------------|
| OmniData (v2, ViT-L)       | 1.3M + 12.2M                                 | 0.069          | 0.948      | 0.074           | 0.945      | 0.149               | 0.835           | 0.166                | 0.778      | 
| MariGold  (LDMv2)     | 5B + 74K                                     | 0.099          | 0.916      | 0.055           | 0.961      | 0.308               | 0.773           | 0.127                | 0.960      | 
| DepthAnything (ViT-L) | 142M + 63M                                   | 0.076          | 0.947      | 0.043           | 0.981      | 0.277               | 0.759           | 0.065                | 0.882      | 
| Ours (ViT-L)          | 142M + 16M                                   | 0.042          | 0.979      | 0.042           | 0.980      | 0.141               | 0.882           | 0.042                | 0.987      | 
| Ours (ViT-g)          | 142M + 16M                                   | 0.043          | 0.982      | 0.043           | 0.981      | 0.136               | 0.895           | 0.042                | 0.983      | 


### Surface Normal
Our models also show powerful performance on normal benchmarks.

|              | NYU 11.25° ↑ | NYU Mean ↓ | NYU RMS ↓ | ScanNet 11.25° ↑ | ScanNet Mean ↓ | ScanNet RMS ↓ | iBims 11.25° ↑ | iBims Mean ↓ | iBims RMS ↓ | 
|--------------|----------|----------|-----------|-----------------|----------------|--------------|---------------|--------------|-------------|
| EESNU        | 0.597    | 16.0     | 24.7      | 0.711           | 11.8           | 20.3         | 0.585         | 20.0         | -           | 
| IronDepth    | -        | -        | -         | -               | -              | -            | 0.431         | 25.3         | 37.4        | 
| PolyMax      | 0.656    | 13.1     | 20.4      | -               | -              | -            | -             | -            | -           |
| Ours (ViT-L) | 0.688    | 12.0     | 19.2      | 0.760           | 9.9            | 16.4         | 0.694         | 19.4         | 34.9        | 
| Ours (ViT-g)   | 0.662    | 13.2     | 20.2      | 0.778           | 9.2            | 15.3         | 0.697         | 19.6         | 35.2        |

[//]: # (## 🌈 3D Reconstruction DEMOs)

[//]: # (### Monocular reconstruction for a Sequence)

[//]: # ()
[//]: # (### In-the-wild 3D reconstruction)

[//]: # ()
[//]: # (|           | Image | Reconstruction | Pointcloud File |)

[//]: # (|:---------:|:------------------:|:------------------:|:--------:|)

[//]: # (|    room   |    <img src="data/wild_demo/jonathan-borba-CnthDZXCdoY-unsplash.jpg" width="300" height="335">     |     <img src="media/gifs/room.gif" width="300" height="335">            |  [Download]&#40;https://drive.google.com/file/d/1P1izSegH2c4LUrXGiUksw037PVb0hjZr/view?usp=drive_link&#41;        |)

[//]: # (| Colosseum |    <img src="data/wild_demo/david-kohler-VFRTXGw1VjU-unsplash.jpg" width="300" height="169">     |     <img src="media/gifs/colo.gif" width="300" height="169">         |     [Download]&#40;https://drive.google.com/file/d/1jJCXe5IpxBhHDr0TZtNZhjxKTRUz56Hg/view?usp=drive_link&#41;     |)

[//]: # (|   chess   |    <img src="data/wild_demo/randy-fath-G1yhU1Ej-9A-unsplash.jpg" width="300" height="169" align=center>     |     <img src="media/gifs/chess.gif" width="300" height="169">            |      [Download]&#40;https://drive.google.com/file/d/1oV_Foq25_p-tTDRTcyO2AzXEdFJQz-Wm/view?usp=drive_link&#41;    |)

[//]: # ()
[//]: # (All three images are downloaded from [unplash]&#40;https://unsplash.com/&#41; and put in the data/wild_demo directory.)

[//]: # ()
[//]: # (### 3D metric reconstruction, Metric3D × DroidSLAM)

[//]: # (Metric3D can also provide scale information for DroidSLAM, help to solve the scale drift problem for better trajectories. )

[//]: # ()
[//]: # (#### Bird Eyes' View &#40;Left: Droid-SLAM &#40;mono&#41;. Right: Droid-SLAM with Metric-3D&#41;)

[//]: # ()
[//]: # (<div align=center>)

[//]: # (<img src="media/gifs/0028.gif"> )

[//]: # (</div>)

[//]: # ()
[//]: # (### Front View)

[//]: # ()
[//]: # (<div align=center>)

[//]: # (<img src="media/gifs/0028_fv.gif"> )

[//]: # (</div>)

[//]: # ()
[//]: # (#### KITTI odemetry evaluation &#40;Translational RMS drift &#40;t_rel, ↓&#41; / Rotational RMS drift &#40;r_rel, ↓&#41;&#41;)

[//]: # (|            | Modality |   seq 00   |   seq 02   |   seq 05  |   seq 06   |   seq 08   |   seq 09  |   seq 10  |)

[//]: # (|:----------:|:--------:|:----------:|:----------:|:---------:|:----------:|:----------:|:---------:|:---------:|)

[//]: # (|  ORB-SLAM2 |   Mono   | 11.43/0.58 | 10.34/0.26 | 9.04/0.26 | 14.56/0.26 | 11.46/0.28 |  9.3/0.26 | 2.57/0.32 |)

[//]: # (| Droid-SLAM |   Mono   |  33.9/0.29 | 34.88/0.27 | 23.4/0.27 |  17.2/0.26 |  39.6/0.31 | 21.7/0.23 |   7/0.25  |)

[//]: # (| Droid+Ours |   Mono   |  1.44/0.37 |  2.64/0.29 | 1.44/0.25 |   0.6/0.2  |   2.2/0.3  | 1.63/0.22 | 2.73/0.23 |)

[//]: # (|  ORB-SLAM2 |  Stereo  |  0.88/0.31 |  0.77/0.28 | 0.62/0.26 |  0.89/0.27 |  1.03/0.31 | 0.86/0.25 | 0.62/0.29 |)

[//]: # ()
[//]: # (Metric3D makes the mono-SLAM scale-aware, like stereo systems.)

[//]: # ()
[//]: # (#### KITTI sequence videos - Youtube)

[//]: # ([2011_09_30_drive_0028]&#40;https://youtu.be/gcTB4MgVCLQ&#41; /)

[//]: # ([2011_09_30_drive_0033]&#40;https://youtu.be/He581fmoPP4&#41; /)

[//]: # ([2011_09_30_drive_0034]&#40;https://youtu.be/I3PkukQ3_F8&#41;)

[//]: # ()
[//]: # (#### Estimated pose)

[//]: # ([2011_09_30_drive_0033]&#40;https://drive.google.com/file/d/1SMXWzLYrEdmBe6uYMR9ShtDXeFDewChv/view?usp=drive_link&#41; / )

[//]: # ([2011_09_30_drive_0034]&#40;https://drive.google.com/file/d/1ONU4GxpvTlgW0TjReF1R2i-WFxbbjQPG/view?usp=drive_link&#41; /)

[//]: # ([2011_10_03_drive_0042]&#40;https://drive.google.com/file/d/19fweg6p1Q6TjJD2KlD7EMA_aV4FIeQUD/view?usp=drive_link&#41;)

[//]: # ()
[//]: # (#### Pointcloud files)

[//]: # ([2011_09_30_drive_0033]&#40;https://drive.google.com/file/d/1K0o8DpUmLf-f_rue0OX1VaHlldpHBAfw/view?usp=drive_link&#41; /)

[//]: # ([2011_09_30_drive_0034]&#40;https://drive.google.com/file/d/1bvZ6JwMRyvi07H7Z2VD_0NX1Im8qraZo/view?usp=drive_link&#41; /)

[//]: # ([2011_10_03_drive_0042]&#40;https://drive.google.com/file/d/1Vw59F8nN5ApWdLeGKXvYgyS9SNKHKy4x/view?usp=drive_link&#41;)

## 🔨 Installation
### One-line Installation
For the ViT models, use the following environment：
```bash
pip install -r requirements_v2.txt
```

For ConvNeXt-L, it is 
```bash
pip install -r requirements.txt
```

### dataset annotation components
With off-the-shelf depth datasets, we need to generate json annotaions in compatible with this dataset, which is organized by:
```
dict(
	'files':list(
		dict(
			'rgb': 'data/kitti_demo/rgb/xxx.png',
			'depth': 'data/kitti_demo/depth/xxx.png',
			'depth_scale': 1000.0 # the depth scale of gt depth img.
			'cam_in': [fx, fy, cx, cy],
		),

		dict(
			...
		),

		...
	)
)
```
To generate such annotations, please refer to the "Inference" section.

### configs
In ```mono/configs``` we provide different config setups. 

Intrinsics of the canonical camera is set bellow: 
```
    canonical_space = dict(
        img_size=(512, 960),
        focal_length=1000.0,
    ),
```
where cx and cy is set to be half of the image size.

Inference settings are defined as
```
    depth_range=(0, 1),
    depth_normalize=(0.3, 150),
    crop_size = (512, 1088),
```
where the images will be first resized as the ```crop_size``` and then fed into the model.

## ✈️ Inference
### Download Checkpoint
|      |       Encoder       |      Decoder      |                                               Link                                                |
|:----:|:-------------------:|:-----------------:|:-------------------------------------------------------------------------------------------------:|
| v1-T |    ConvNeXt-Tiny    | Hourglass-Decoder |                                            Coming soon                                            |
| v1-L |   ConvNeXt-Large    | Hourglass-Decoder | [Download](https://drive.google.com/file/d/1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN/view?usp=drive_link) |
| v2-S | DINO2reg-ViT-Small  |    RAFT-4iter     | [Download](https://drive.google.com/file/d/1YfmvXwpWmhLg3jSxnhT7LvY0yawlXcr_/view?usp=drive_link) |
| v2-L | DINO2reg-ViT-Large  |    RAFT-8iter     | [Download](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link) |
| v2-g | DINO2reg-ViT-giant2 |    RAFT-8iter     | Coming soon |

### Dataset Mode
1. put the trained ckpt file ```model.pth``` in ```weight/```.
2. generate data annotation by following the code ```data/gene_annos_kitti_demo.py```, which includes 'rgb', (optional) 'intrinsic', (optional) 'depth', (optional) 'depth_scale'.
3. change the 'test_data_path' in ```test_*.sh``` to the ```*.json``` path. 
4. run ```source test_kitti.sh``` or ```source test_nyu.sh```.

### In-the-Wild Mode
1. put the trained ckpt file ```model.pth``` in ```weight/```.
2. change the 'test_data_path' in ```test.sh``` to the image folder path. 
3. run ```source test_vit.sh``` for transformers and ```source test.sh``` for convnets.
As no intrinsics are provided, we provided by default 9 settings of focal length.

## ❓ Q & A
### Q1: Why depth maps look good but pointclouds are distorted?
Because the focal length is not properly set! Please find a proper focal length by modifying codes [here](mono/utils/do_test.py#309) yourself.  

### Q2: Why the pointclouds are too slow to be generated?
Because the images are too large! Use smaller ones instead. 

### Q3: Why predicted depth maps are not satisfactory?
First be sure all black padding regions at image boundaries are cropped out. Then please try again.
Besides, metric 3D is not almighty. Some objects (chandeliers, drones...) / camera views (aerial view, bev...) do not occur frequently in the training datasets. We will going deeper into this and release more powerful solutions.

## 📧 Citation
```
@article{hu2024metric3dv2,
  title={A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation},
  author={Hu, Mu and Yin, Wei, and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao, and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
  booktitle={arXiv},
  year={2024}
}
```
```
@article{yin2023metric,
  title={Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image},
  author={Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen},
  booktitle={ICCV},
  year={2023}
}
```

## License and Contact

The *Metric 3D* code is under a 2-clause BSD License for non-commercial usage. For further questions, contact Dr. yvan.yin  [yvanwy@outlook.com] and Mr. mu.hu [mhuam@connect.ust.hk].
