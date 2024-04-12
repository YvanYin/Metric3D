#import collections
import collections.abc as collections
import cv2
import math
import numpy as np
import numbers
import random
import torch
from imgaug import augmenters as iaa
import matplotlib
import matplotlib.cm
import mono.utils.weather_aug_utils as wa

"""
Provides a set of Pytorch transforms that use OpenCV instead of PIL (Pytorch default)
for image manipulation.
"""

class Compose(object):
    # Composes transforms: transforms.Compose([transforms.RandScale([0.5, 2.0]), transforms.ToTensor()])
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        for t in self.transforms:
            images, labels, intrinsics, cam_models, normals, other_labels, transform_paras = t(images, labels, intrinsics, cam_models, normals, other_labels, transform_paras)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras

class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __init__(self,  **kwargs):
        return
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        if not isinstance(images, list) or not isinstance(labels, list) or not isinstance(intrinsics, list):
            raise (RuntimeError("transform.ToTensor() only handle inputs/labels/intrinsics lists."))
        if len(images) != len(intrinsics):
            raise (RuntimeError("Numbers of images and intrinsics are not matched."))
        if not isinstance(images[0], np.ndarray) or not isinstance(labels[0], np.ndarray):
            raise (RuntimeError("transform.ToTensor() only handle np.ndarray for the input and label."
                                "[eg: data readed by cv2.imread()].\n"))
        if  not isinstance(intrinsics[0], list):
            raise (RuntimeError("transform.ToTensor() only handle list for the camera intrinsics"))

        if len(images[0].shape) > 3 or len(images[0].shape) < 2:
            raise (RuntimeError("transform.ToTensor() only handle image(np.ndarray) with 3 dims or 2 dims.\n"))
        if len(labels[0].shape) > 3 or len(labels[0].shape) < 2:
            raise (RuntimeError("transform.ToTensor() only handle label(np.ndarray) with 3 dims or 2 dims.\n"))

        if len(intrinsics[0]) >4 or len(intrinsics[0]) < 3:
            raise (RuntimeError("transform.ToTensor() only handle intrinsic(list) with 3 sizes or 4 sizes.\n"))
        
        for i, img in enumerate(images):
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            images[i] = torch.from_numpy(img.transpose((2, 0, 1))).float()
        for i, lab in enumerate(labels):
            if len(lab.shape) == 2:
                lab = np.expand_dims(lab, axis=0)
            labels[i] = torch.from_numpy(lab).float()
        for i, intrinsic in enumerate(intrinsics):
            if len(intrinsic) == 3:
                intrinsic = [intrinsic[0],] + intrinsic
            intrinsics[i] = torch.tensor(intrinsic, dtype=torch.float)
        if cam_models is not None:
            for i, cam_model in enumerate(cam_models):
                cam_models[i] = torch.from_numpy(cam_model.transpose((2, 0, 1))).float() if cam_model is not None else None
        if normals is not None:
            for i, normal in enumerate(normals):
                normals[i] = torch.from_numpy(normal.transpose((2, 0, 1))).float()
        if other_labels is not None:
            for i, lab in enumerate(other_labels):
                if len(lab.shape) == 2:
                    lab = np.expand_dims(lab, axis=0)
                other_labels[i] = torch.from_numpy(lab).float()
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras

class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None, **kwargs):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = torch.tensor(mean).float()[:, None, None]
        self.std = torch.tensor(std).float()[:, None, None] if std is not None \
            else torch.tensor([1.0, 1.0, 1.0]).float()[:, None, None]

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        # if self.std is None:
        #     # for t, m in zip(image, self.mean):
        #     #     t.sub(m)
        #     image = image - self.mean
        #     if ref_images is not None:
        #         for i, ref_i in enumerate(ref_images):
        #             ref_images[i] =  ref_i - self.mean
        # else:
        #     # for t, m, s in zip(image, self.mean, self.std):
        #     #     t.sub(m).div(s)
        #     image = (image - self.mean) / self.std
        #     if ref_images is not None:
        #         for i, ref_i in enumerate(ref_images):
        #             ref_images[i] =  (ref_i - self.mean) / self.std
        for i, img in enumerate(images):
            img = torch.div((img - self.mean), self.std)
            images[i] = img
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras

class ResizeCanonical(object):
    """
    Resize the input to the canonical space first, then resize the input with random sampled size.
    In the first stage, we assume the distance holds while the camera model varies. 
    In the second stage, we aim to simulate the observation in different distance. The camera will move along the optical axis.
    Args:
        images: list of RGB images.
        labels: list of depth/disparity labels.
        other labels: other labels, such as instance segmentations, semantic segmentations...
    """
    def __init__(self, **kwargs):
        self.ratio_range = kwargs['ratio_range']
        self.canonical_focal = kwargs['focal_length']
        self.crop_size = kwargs['crop_size']
    
    def random_on_canonical_transform(self, image, label, intrinsic, cam_model, to_random_ratio):
        ori_h, ori_w, _ = image.shape
        ori_focal = (intrinsic[0] + intrinsic[1]) / 2.0

        to_canonical_ratio = self.canonical_focal / ori_focal
        to_scale_ratio = to_random_ratio
        resize_ratio = to_canonical_ratio * to_random_ratio
        reshape_h = int(ori_h * resize_ratio + 0.5)
        reshape_w = int(ori_w * resize_ratio + 0.5)

        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        if intrinsic is not None:
            intrinsic = [self.canonical_focal, self.canonical_focal, intrinsic[2]*resize_ratio, intrinsic[3]*resize_ratio]
        if label is not None:
            # number of other labels may be less than that of image
            label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            # scale the label and camera intrinsics
            label = label / to_scale_ratio
        
        if cam_model is not None:
            # Should not directly resize the cam_model.
            # Camera model should be resized in 'to canonical' stage, while it holds in 'random resizing' stage. 
            # cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
        
        return image, label, intrinsic, cam_model, to_scale_ratio       
    
    def random_on_crop_transform(self, image, label, intrinsic, cam_model, to_random_ratio):
        ori_h, ori_w, _ = image.shape
        crop_h, crop_w = self.crop_size
        ori_focal = (intrinsic[0] + intrinsic[1]) / 2.0

        to_canonical_ratio = self.canonical_focal / ori_focal
        
        # random resize based on the last crop size
        proposal_reshape_h = int(crop_h * to_random_ratio + 0.5) 
        proposal_reshape_w = int(crop_w * to_random_ratio + 0.5)
        resize_ratio_h = proposal_reshape_h / ori_h
        resize_ratio_w = proposal_reshape_w / ori_w
        resize_ratio = min(resize_ratio_h, resize_ratio_w) # resize based on the long edge
        reshape_h = int(ori_h * resize_ratio + 0.5)
        reshape_w = int(ori_w * resize_ratio + 0.5)
        
        to_scale_ratio = resize_ratio / to_canonical_ratio        

        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        if intrinsic is not None:
            intrinsic = [self.canonical_focal, self.canonical_focal, intrinsic[2]*resize_ratio, intrinsic[3]*resize_ratio]
        if label is not None:
            # number of other labels may be less than that of image
            label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            # scale the label and camera intrinsics
            label = label / to_scale_ratio
        
        if cam_model is not None:
            # Should not directly resize the cam_model.
            # Camera model should be resized in 'to canonical' stage, while it holds in 'random resizing' stage. 
            # cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
        return image, label, intrinsic, cam_model, to_scale_ratio       

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        assert len(images[0].shape) == 3 and len(labels[0].shape) == 2
        assert labels[0].dtype == np.float
        target_focal = (intrinsics[0][0] + intrinsics[0][1]) / 2.0
        target_to_canonical_ratio = self.canonical_focal / target_focal
        target_img_shape = images[0].shape
        to_random_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        to_scale_ratio = 0.0
        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i] if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model, to_scale_ratio = self.random_on_canonical_transform(
                img, label, intrinsic, cam_model, to_random_ratio)
                
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model 

        if normals != None:
            reshape_h, reshape_w, _ = images[0].shape
            for i, normal in enumerate(normals):
                normals[i] = cv2.resize(normal, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
                
        if other_labels != None: 
            # other labels are like semantic segmentations, instance segmentations, instance planes segmentations...           
            #resize_ratio = target_to_canonical_ratio * to_scale_ratio
            #reshape_h = int(target_img_shape[0] * resize_ratio + 0.5)
            #reshape_w = int(target_img_shape[1] * resize_ratio + 0.5)
            reshape_h, reshape_w, _ = images[0].shape
            for i, other_label_i in enumerate(other_labels):
                other_labels[i] = cv2.resize(other_label_i, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
        
        if transform_paras is not None:
            transform_paras.update(label_scale_factor = 1.0/to_scale_ratio)

        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras


class LabelScaleCononical(object):
    """
    To solve the ambiguity observation for the mono branch, i.e. different focal length (object size) with the same depth, cameras are
    mapped to a cononical space. To mimic this, we set the focal length to a canonical one and scale the depth value. NOTE: resize the image based on the ratio can also solve this ambiguity.
    Args:
        images: list of RGB images.
        labels: list of depth/disparity labels.
        other labels: other labels, such as instance segmentations, semantic segmentations...
    """
    def __init__(self, **kwargs):
        self.canonical_focal = kwargs['focal_length']
    
    def _get_scale_ratio(self, intrinsic):
        target_focal_x = intrinsic[0]
        label_scale_ratio = self.canonical_focal / target_focal_x
        pose_scale_ratio = 1.0
        return label_scale_ratio, pose_scale_ratio
        
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        assert len(images[0].shape) == 3 and len(labels[0].shape) == 2
        #assert labels[0].dtype == np.float

        label_scale_ratio = None
        pose_scale_ratio = None
        
        for i in range(len(intrinsics)): 
            img_i = images[i]
            label_i = labels[i] if  i < len(labels) else None
            intrinsic_i = intrinsics[i].copy()
            cam_model_i = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            
            label_scale_ratio, pose_scale_ratio = self._get_scale_ratio(intrinsic_i)
            
            # adjust the focal length, map the current camera to the canonical space
            intrinsics[i] = [intrinsic_i[0]*label_scale_ratio, intrinsic_i[1]*label_scale_ratio, intrinsic_i[2], intrinsic_i[3]]
            
            # scale the label to the canonical space
            if label_i is not None:
                labels[i] = label_i * label_scale_ratio
                        
            if cam_model_i is not None:
                # As the focal length is adjusted (canonical focal length), the camera model should be re-built.
                ori_h, ori_w, _ = img_i.shape
                cam_models[i] = build_camera_model(ori_h, ori_w, intrinsics[i])
            
        
        if transform_paras is not None:
            transform_paras.update(label_scale_factor = label_scale_ratio)

        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras



class ResizeKeepRatio(object):
    """
    Resize and pad to a given size. Hold the aspect ratio.
    This resizing assumes that the camera model remains unchanged.
    Args:
        resize_size: predefined output size.
    """
    def __init__(self, resize_size, padding=None, ignore_label=-1, **kwargs):
        if isinstance(resize_size, int):
            self.resize_h = resize_size
            self.resize_w = resize_size
        elif isinstance(resize_size, collections.Iterable) and len(resize_size) == 2 \
                and isinstance(resize_size[0], int) and isinstance(resize_size[1], int) \
                and resize_size[0] > 0 and resize_size[1] > 0:
            self.resize_h = resize_size[0]
            self.resize_w = resize_size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))
        self.crop_size = kwargs['crop_size']
        self.canonical_focal = kwargs['focal_length']

    def main_data_transform(self, image, label, intrinsic, cam_model, resize_ratio, padding, to_scale_ratio):
        """
        Resize data first and then do the padding.
        'label' will be scaled.
        """
        h, w, _ = image.shape
        reshape_h = int(resize_ratio * h)
        reshape_w = int(resize_ratio * w)

        pad_h, pad_w, pad_h_half, pad_w_half = padding
        
        # resize
        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        # padding
        image = cv2.copyMakeBorder(
            image, 
            pad_h_half, 
            pad_h - pad_h_half, 
            pad_w_half, 
            pad_w - pad_w_half, 
            cv2.BORDER_CONSTANT, 
            value=self.padding)

        if label is not None:
            # label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            label = resize_depth_preserve(label, (reshape_h, reshape_w))
            label = cv2.copyMakeBorder(
                label, 
                pad_h_half, 
                pad_h - pad_h_half, 
                pad_w_half, 
                pad_w - pad_w_half, 
                cv2.BORDER_CONSTANT, 
                value=self.ignore_label)
            # scale the label
            label = label / to_scale_ratio
        
        # Resize, adjust principle point
        if intrinsic is not None:
            intrinsic[2] = intrinsic[2] * resize_ratio
            intrinsic[3] = intrinsic[3] * resize_ratio

        if cam_model is not None:
            #cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
            cam_model = cv2.copyMakeBorder(
                cam_model, 
                pad_h_half, 
                pad_h - pad_h_half, 
                pad_w_half, 
                pad_w - pad_w_half, 
                cv2.BORDER_CONSTANT, 
                value=self.ignore_label)

        # Pad, adjust the principle point
        if intrinsic is not None:
            intrinsic[2] = intrinsic[2] + pad_w_half
            intrinsic[3] = intrinsic[3] + pad_h_half
        return image, label, intrinsic, cam_model
    
    def get_label_scale_factor(self, image, intrinsic, resize_ratio):
        ori_h, ori_w, _ = image.shape
        crop_h, crop_w = self.crop_size
        ori_focal = (intrinsic[0] + intrinsic[1]) / 2.0 #intrinsic[0] #

        to_canonical_ratio = self.canonical_focal / ori_focal
        to_scale_ratio = resize_ratio / to_canonical_ratio 
        return to_scale_ratio
        
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        target_h, target_w, _ = images[0].shape
        resize_ratio_h = self.resize_h / target_h
        resize_ratio_w = self.resize_w / target_w
        resize_ratio = min(resize_ratio_h, resize_ratio_w)
        reshape_h = int(resize_ratio * target_h)
        reshape_w = int(resize_ratio * target_w)
        pad_h = max(self.resize_h - reshape_h, 0)
        pad_w = max(self.resize_w - reshape_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        
        pad_info = [pad_h, pad_w, pad_h_half, pad_w_half]
        to_scale_ratio = self.get_label_scale_factor(images[0], intrinsics[0], resize_ratio)

        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i] if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model = self.main_data_transform(
                img, label, intrinsic, cam_model, resize_ratio, pad_info, to_scale_ratio)
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model 
        
        if normals is not None:
            for i, normal in enumerate(normals):
                normal =  cv2.resize(normal, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
                # pad
                normals[i] =  cv2.copyMakeBorder(
                    normal, 
                    pad_h_half, 
                    pad_h - pad_h_half, 
                    pad_w_half, 
                    pad_w - pad_w_half, 
                    cv2.BORDER_CONSTANT, 
                    value=0)

        if other_labels is not None:
            
            for i, other_lab in enumerate(other_labels):
                # resize
                other_lab =  cv2.resize(other_lab, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
                # pad
                other_labels[i] =  cv2.copyMakeBorder(
                    other_lab, 
                    pad_h_half, 
                    pad_h - pad_h_half, 
                    pad_w_half, 
                    pad_w - pad_w_half, 
                    cv2.BORDER_CONSTANT, 
                    value=self.ignore_label)


        if transform_paras is not None:
            transform_paras.update(pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half])
            if 'label_scale_factor' in transform_paras:
                transform_paras['label_scale_factor'] = transform_paras['label_scale_factor'] * 1.0 / to_scale_ratio
            else:
                transform_paras.update(label_scale_factor=1.0/to_scale_ratio)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras
    
class KeepResizeCanoSize(object):
    """
    Resize and pad to a given size. Hold the aspect ratio.
    This resizing assumes that the camera model remains unchanged.
    Args:
        resize_size: predefined output size.
    """
    def __init__(self, resize_size, padding=None, ignore_label=-1, **kwargs):
        if isinstance(resize_size, int):
            self.resize_h = resize_size
            self.resize_w = resize_size
        elif isinstance(resize_size, collections.Iterable) and len(resize_size) == 2 \
                and isinstance(resize_size[0], int) and isinstance(resize_size[1], int) \
                and resize_size[0] > 0 and resize_size[1] > 0:
            self.resize_h = resize_size[0]
            self.resize_w = resize_size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))
        self.crop_size = kwargs['crop_size']
        self.canonical_focal = kwargs['focal_length']

    def main_data_transform(self, image, label, intrinsic, cam_model, resize_ratio, padding, to_scale_ratio):
        """
        Resize data first and then do the padding.
        'label' will be scaled.
        """
        h, w, _ = image.shape
        reshape_h = int(resize_ratio * h)
        reshape_w = int(resize_ratio * w)

        pad_h, pad_w, pad_h_half, pad_w_half = padding
        
        # resize
        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        # padding
        image = cv2.copyMakeBorder(
            image, 
            pad_h_half, 
            pad_h - pad_h_half, 
            pad_w_half, 
            pad_w - pad_w_half, 
            cv2.BORDER_CONSTANT, 
            value=self.padding)

        if label is not None:
            # label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            label = resize_depth_preserve(label, (reshape_h, reshape_w))
            label = cv2.copyMakeBorder(
                label, 
                pad_h_half, 
                pad_h - pad_h_half, 
                pad_w_half, 
                pad_w - pad_w_half, 
                cv2.BORDER_CONSTANT, 
                value=self.ignore_label)
            # scale the label
            label = label / to_scale_ratio
        
        # Resize, adjust principle point
        if intrinsic is not None:
            intrinsic[2] = intrinsic[2] * resize_ratio
            intrinsic[3] = intrinsic[3] * resize_ratio

        if cam_model is not None:
            #cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
            cam_model = cv2.copyMakeBorder(
                cam_model, 
                pad_h_half, 
                pad_h - pad_h_half, 
                pad_w_half, 
                pad_w - pad_w_half, 
                cv2.BORDER_CONSTANT, 
                value=self.ignore_label)

        # Pad, adjust the principle point
        if intrinsic is not None:
            intrinsic[2] = intrinsic[2] + pad_w_half
            intrinsic[3] = intrinsic[3] + pad_h_half
        return image, label, intrinsic, cam_model
    
    # def get_label_scale_factor(self, image, intrinsic, resize_ratio):
    #     ori_h, ori_w, _ = image.shape
    #     crop_h, crop_w = self.crop_size
    #     ori_focal = intrinsic[0] #(intrinsic[0] + intrinsic[1]) / 2.0

    #     to_canonical_ratio = self.canonical_focal / ori_focal
    #     to_scale_ratio = resize_ratio / to_canonical_ratio 
    #     return to_scale_ratio
        
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        target_h, target_w, _ = images[0].shape
        ori_focal = intrinsics[0][0]
        to_canonical_ratio = self.canonical_focal / ori_focal 
        
        resize_ratio = to_canonical_ratio
        reshape_h = int(resize_ratio * target_h)
        reshape_w = int(resize_ratio * target_w)
        
        pad_h = 32 - reshape_h % 32
        pad_w = 32 - reshape_w % 32
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        
        pad_info = [pad_h, pad_w, pad_h_half, pad_w_half]
        to_scale_ratio = 1.0

        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i] if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model = self.main_data_transform(
                img, label, intrinsic, cam_model, resize_ratio, pad_info, to_scale_ratio)
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model 
        
        if normals is not None:
            
            for i, normal in enumerate(normals):
                # resize
                normal =  cv2.resize(normal, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
                # pad
                normals[i] =  cv2.copyMakeBorder(
                    normal, 
                    pad_h_half, 
                    pad_h - pad_h_half, 
                    pad_w_half, 
                    pad_w - pad_w_half, 
                    cv2.BORDER_CONSTANT, 
                    value=0)

        if other_labels is not None:
            
            for i, other_lab in enumerate(other_labels):
                # resize
                other_lab =  cv2.resize(other_lab, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
                # pad
                other_labels[i] =  cv2.copyMakeBorder(
                    other_lab, 
                    pad_h_half, 
                    pad_h - pad_h_half, 
                    pad_w_half, 
                    pad_w - pad_w_half, 
                    cv2.BORDER_CONSTANT, 
                    value=self.ignore_label)


        if transform_paras is not None:
            transform_paras.update(pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half])
            if 'label_scale_factor' in transform_paras:
                transform_paras['label_scale_factor'] = transform_paras['label_scale_factor'] * 1.0 / to_scale_ratio
            else:
                transform_paras.update(label_scale_factor=1.0/to_scale_ratio)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras


class RandomCrop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, crop_size, crop_type='center', padding=None, ignore_label=-1, **kwargs):
        if isinstance(crop_size, int):
            self.crop_h = crop_size
            self.crop_w = crop_size
        elif isinstance(crop_size, collections.Iterable) and len(crop_size) == 2 \
                and isinstance(crop_size[0], int) and isinstance(crop_size[1], int) \
                and crop_size[0] > 0 and crop_size[1] > 0:
            self.crop_h = crop_size[0]
            self.crop_w = crop_size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand' or crop_type=='rand_in_field':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center | rand_in_field \n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))
        

    def cal_padding_paras(self, h, w):
        # padding if current size is not satisfied
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        return pad_h, pad_w, pad_h_half, pad_w_half

    def cal_cropping_paras(self, h, w, intrinsic):
        u0 = intrinsic[2]
        v0 = intrinsic[3]
        if self.crop_type == 'rand':
            h_min = 0
            h_max = h - self.crop_h
            w_min = 0
            w_max = w - self.crop_w 
        elif self.crop_type == 'center':
            h_min = (h - self.crop_h) / 2
            h_max = (h - self.crop_h) / 2
            w_min = (w - self.crop_w) / 2
            w_max = (w - self.crop_w) / 2
        else: # rand in field
            h_min = min(max(0, v0 - 0.75*self.crop_h), h-self.crop_h)
            h_max = min(max(v0 - 0.25*self.crop_h, 0), h-self.crop_h)
            w_min = min(max(0, u0 - 0.75*self.crop_w), w-self.crop_w)
            w_max = min(max(u0 - 0.25*self.crop_w, 0), w-self.crop_w)
        
        h_off = random.randint(int(h_min), int(h_max))
        w_off = random.randint(int(w_min), int(w_max))
        return h_off, w_off
    
    def main_data_transform(self, image, label, intrinsic, cam_model,
        pad_h, pad_w, pad_h_half, pad_w_half, h_off, w_off):

        # padding if current size is not satisfied
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("depthtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            if label is not None:
                label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
            if cam_model is not None:
                cam_model = cv2.copyMakeBorder(cam_model, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        
        # cropping
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if label is not None:
            label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if cam_model is not None:
            cam_model = cam_model[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]

        if intrinsic is not None:
            intrinsic[2] = intrinsic[2] + pad_w_half - w_off
            intrinsic[3] = intrinsic[3] + pad_h_half - h_off    
        return image, label, intrinsic, cam_model

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        if 'random_crop_size' in transform_paras and transform_paras['random_crop_size'] is not None \
            and (transform_paras['random_crop_size'][0] + transform_paras['random_crop_size'][1] > 500):
            self.crop_h = int(transform_paras['random_crop_size'][0].item())
            self.crop_w = int(transform_paras['random_crop_size'][1].item())
        target_img = images[0]
        target_h, target_w, _ = target_img.shape
        target_intrinsic = intrinsics[0]
        pad_h, pad_w, pad_h_half, pad_w_half = self.cal_padding_paras(target_h, target_w)
        h_off, w_off = self.cal_cropping_paras(target_h+pad_h, target_w+pad_w, target_intrinsic)

        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i].copy() if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model = self.main_data_transform(
                img, label, intrinsic, cam_model,
                pad_h, pad_w, pad_h_half, pad_w_half, h_off, w_off)
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model 
        pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        if normals is not None:           
            for i, normal  in enumerate(normals):
                # padding if current size is not satisfied
                normal = cv2.copyMakeBorder(normal, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=0)
                normals[i] = normal[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if other_labels is not None:           
            for i, other_lab  in enumerate(other_labels):
                # padding if current size is not satisfied
                other_lab = cv2.copyMakeBorder(other_lab, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
                other_labels[i] = other_lab[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if transform_paras is not None:
            transform_paras.update(dict(pad=pad))
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras
    

class RandomResize(object):
    """
    Random resize the image. During this process, the camera model is hold, and thus the depth label is scaled.
    Args:
        images: list of RGB images.
        labels: list of depth/disparity labels.
        other labels: other labels, such as instance segmentations, semantic segmentations...
    """
    def __init__(self, ratio_range=(0.85, 1.15), prob=0.5, is_lidar=True, **kwargs):
        self.ratio_range = ratio_range
        self.is_lidar = is_lidar
        self.prob = prob
    
    def random_resize(self, image, label, intrinsic, cam_model, to_random_ratio):
        ori_h, ori_w, _ = image.shape
        
        resize_ratio = to_random_ratio
        label_scale_ratio = 1.0 / resize_ratio
        reshape_h = int(ori_h * resize_ratio + 0.5)
        reshape_w = int(ori_w * resize_ratio + 0.5)

        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        if intrinsic is not None:
            intrinsic = [intrinsic[0], intrinsic[1], intrinsic[2]*resize_ratio, intrinsic[3]*resize_ratio]
        if label is not None:
            if self.is_lidar:
                label = resize_depth_preserve(label, (reshape_h, reshape_w))
            else:
                label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            # scale the label
            label = label * label_scale_ratio
        
        if cam_model is not None:
            # Should not directly resize the cam_model.
            # Camera model should be resized in 'to canonical' stage, while it holds in 'random resizing' stage. 
            # cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
        
        return image, label, intrinsic, cam_model, label_scale_ratio       

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        assert len(images[0].shape) == 3 and len(labels[0].shape) == 2
        assert labels[0].dtype == np.float
        # target_focal = (intrinsics[0][0] + intrinsics[0][1]) / 2.0
        # target_to_canonical_ratio = self.canonical_focal / target_focal
        # target_img_shape = images[0].shape
        prob = random.uniform(0, 1)
        if prob < self.prob:
            to_random_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        else:
            to_random_ratio = 1.0
        label_scale_ratio = 0.0
        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i].copy() if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model, label_scale_ratio = self.random_resize(
                img, label, intrinsic, cam_model, to_random_ratio)
                
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic.copy()
            if cam_model is not None:
                cam_models[i] = cam_model 

        if normals != None: 
            reshape_h, reshape_w, _ = images[0].shape
            for i, norm in enumerate(normals):
                normals[i] = cv2.resize(norm, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)


        if other_labels != None: 
            # other labels are like semantic segmentations, instance segmentations, instance planes segmentations...           
            #resize_ratio = target_to_canonical_ratio * to_scale_ratio
            #reshape_h = int(target_img_shape[0] * resize_ratio + 0.5)
            #reshape_w = int(target_img_shape[1] * resize_ratio + 0.5)
            reshape_h, reshape_w, _ = images[0].shape
            for i, other_label_i in enumerate(other_labels):
                other_labels[i] = cv2.resize(other_label_i, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
        
        if transform_paras is not None:
            if 'label_scale_factor' in transform_paras:
                transform_paras['label_scale_factor'] = transform_paras['label_scale_factor'] * label_scale_ratio
            else:
                transform_paras.update(label_scale_factor = label_scale_ratio)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras

class RandomEdgeMask(object):
    """
    Random mask the input and labels.
    Args:
        images: list of RGB images.
        labels: list of depth/disparity labels.
        other labels: other labels, such as instance segmentations, semantic segmentations...
    """
    def __init__(self, mask_maxsize=32, prob=0.5, rgb_invalid=[0,0,0], label_invalid=-1,**kwargs):
        self.mask_maxsize = mask_maxsize
        self.prob = prob
        self.rgb_invalid = rgb_invalid
        self.label_invalid = label_invalid
    
    def mask_edge(self, image, mask_edgesize, mask_value):
        H, W = image.shape[0], image.shape[1]
        # up
        image[0:mask_edgesize[0], :, ...] = mask_value 
        # down
        image[H-mask_edgesize[1]:H, :, ...] = mask_value
        # left
        image[:, 0:mask_edgesize[2], ...] = mask_value
        # right
        image[:, W-mask_edgesize[3]:W, ...] = mask_value
        
        return image

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        assert len(images[0].shape) == 3 and len(labels[0].shape) == 2
        assert labels[0].dtype == np.float
        
        prob = random.uniform(0, 1)
        if prob > self.prob:
            return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras
        
        mask_edgesize = random.sample(range(self.mask_maxsize), 4) #[up, down, left, right]
        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            img = self.mask_edge(img, mask_edgesize, self.rgb_invalid)
                
            images[i] = img
            if label is not None:
                label = self.mask_edge(label, mask_edgesize, self.label_invalid)
                labels[i] = label
        
        if normals != None:       
            for i, normal in enumerate(normals):
                normals[i] = self.mask_edge(normal, mask_edgesize, mask_value=0)

        if other_labels != None: 
            # other labels are like semantic segmentations, instance segmentations, instance planes segmentations...           
            for i, other_label_i in enumerate(other_labels):
                other_labels[i] = self.mask_edge(other_label_i, mask_edgesize, self.label_invalid)

        if transform_paras is not None:
            pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
            new_pad = [max(mask_edgesize[0], pad[0]), max(mask_edgesize[1], pad[1]), max(mask_edgesize[2], pad[2]), max(mask_edgesize[3], pad[3])]
            transform_paras.update(dict(pad=new_pad)) 
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras


class AdjustSize(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, padding=None, ignore_label=-1, **kwargs):
        
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))
        
    def get_pad_paras(self, h, w):
        pad_h = 32 - h % 32 if h %32 != 0 else 0
        pad_w = 32 - w % 32 if w %32 != 0 else 0
        pad_h_half = int(pad_h // 2)
        pad_w_half = int(pad_w // 2)
        return pad_h, pad_w, pad_h_half, pad_w_half

    def main_data_transform(self, image, label, intrinsic, cam_model):
        h, w, _ = image.shape
        pad_h, pad_w, pad_h_half, pad_w_half = self.get_pad_paras(h=h, w=w)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("depthtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            if label is not None:
                label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
            if cam_model is not None:
                cam_model = cv2.copyMakeBorder(cam_model, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)

        if intrinsic is not None:
            intrinsic[2] = intrinsic[2] + pad_w_half
            intrinsic[3] = intrinsic[3] + pad_h_half
        pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        return image, label, intrinsic, cam_model, pad
    
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        target_img = images[0]
        target_h, target_w, _ = target_img.shape
        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i] if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model, pad = self.main_data_transform(
                img, label, intrinsic, cam_model)
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model 

            if transform_paras is not None:
                transform_paras.update(dict(pad=pad))
        if normals is not None:
            pad_h, pad_w, pad_h_half, pad_w_half = self.get_pad_paras(h=target_h, w=target_w)
            for i, normal  in enumerate(normals):
                normals[i] = cv2.copyMakeBorder(normal, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=0)

        if other_labels is not None:
            pad_h, pad_w, pad_h_half, pad_w_half = self.get_pad_paras(h=target_h, w=target_w)
            for i, other_lab  in enumerate(other_labels):
                other_labels[i] = cv2.copyMakeBorder(other_lab, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5, **kwargs):
        self.p = prob

    def main_data_transform(self, image, label, intrinsic, cam_model, rotate):
        if rotate:
            image = cv2.flip(image, 1)
            if label is not None:
                label = cv2.flip(label, 1)
            if intrinsic is not None:
                h, w, _ = image.shape
                intrinsic[2] = w - intrinsic[2]
                intrinsic[3] = h - intrinsic[3]
            if cam_model is not None:
                cam_model = cv2.flip(cam_model, 1)
                cam_model[:, :, 0] = cam_model[:, :, 0] * -1
                cam_model[:, :, 2] = cam_model[:, :, 2] * -1
        return image, label, intrinsic, cam_model
    
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        rotate = random.random() > self.p 

        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i] if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model = self.main_data_transform(
                img, label, intrinsic, cam_model, rotate)
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model 
        if normals is not None:
            for i, normal in enumerate(normals):
                if rotate:
                    normal = cv2.flip(normal, 1)
                    normal[:, :, 0] = -normal[:, :, 0] # NOTE: check the direction of normal coordinates axis, this is used in https://github.com/baegwangbin/surface_normal_uncertainty
                normals[i] = normal

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                if rotate:
                    other_lab = cv2.flip(other_lab, 1)
                other_labels[i] = other_lab
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras

class RandomBlur(object):
    def __init__(self, 
                 aver_kernal=(2, 10), 
                 motion_kernal=(5, 15), 
                 angle=[-80, 80], 
                 prob=0.3,
                 **kwargs):

        gaussian_blur = iaa.AverageBlur(k=aver_kernal)
        motion_blur = iaa.MotionBlur(k=motion_kernal, angle=angle)
        zoom_blur = iaa.imgcorruptlike.ZoomBlur(severity=1)
        self.prob = prob
        self.blurs = [gaussian_blur, motion_blur, zoom_blur]

    def blur(self, imgs, id):
        blur_mtd = self.blurs[id]
        return blur_mtd(images=imgs)

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        prob = random.random()
        if prob < self.prob:
            id = random.randint(0, len(self.blurs)-1)
            images = self.blur(images, id)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras

class RGBCompresion(object):
    def __init__(self, prob=0.1, compression=(0, 50), **kwargs):
        self.rgb_compress = iaa.Sequential(
            [
                iaa.JpegCompression(compression=compression),
            ],
            random_order=True,
        )
        self.prob = prob

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        if random.random() < self.prob:
            images = self.rgb_compress(images=images)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __init__(self,  **kwargs):
        return
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        for i, img in enumerate(images):
            images[i] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __init__(self,  **kwargs):
        return
    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        for i, img in enumerate(images):
            images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras


class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 to_gray_prob=0.3,
                 distortion_prob=0.3,
                 **kwargs):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.gray_aug = iaa.Grayscale(alpha=(0.8, 1.0))
        self.to_gray_prob = to_gray_prob
        self.distortion_prob = distortion_prob

    def convert(self, img, alpha=1.0, beta=0.0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img, beta, do):
        """Brightness distortion."""
        if do:
            # beta = random.uniform(-self.brightness_delta,
            #                         self.brightness_delta)
            img = self.convert(
                img,
                beta=beta)
        return img

    def contrast(self, img, alpha, do):
        """Contrast distortion."""
        if do:
            #alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img = self.convert(
                img,
                alpha=alpha)
        return img

    def saturation(self, img, alpha, do):
        """Saturation distortion."""
        if do:
            # alpha = random.uniform(self.saturation_lower,
            #                         self.saturation_upper)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=alpha)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)        
        return img

    def hue(self, img, rand_hue, do):
        """Hue distortion."""
        if do:
            # rand_hue = random.randint(-self.hue_delta, self.hue_delta)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img
    
    def rgb2gray(self, img):
        img = self.gray_aug(image=img)
        return img

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        brightness_beta = random.uniform(-self.brightness_delta, self.brightness_delta)
        brightness_do = random.random() < self.distortion_prob

        contrast_alpha = random.uniform(self.contrast_lower, self.contrast_upper)
        contrast_do = random.random() < self.distortion_prob

        saturate_alpha = random.uniform(self.saturation_lower, self.saturation_upper)
        saturate_do = random.random() < self.distortion_prob

        rand_hue = random.randint(-self.hue_delta, self.hue_delta)
        rand_hue_do = random.random() < self.distortion_prob

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = 1 if random.random() > 0.5 else 2
        for i, img in enumerate(images):
            if random.random() < self.to_gray_prob:
                img = self.rgb2gray(img)
            else:
                # random brightness
                img = self.brightness(img, brightness_beta, brightness_do)

                if mode == 1:
                    img = self.contrast(img, contrast_alpha, contrast_do)

                # random saturation
                img = self.saturation(img, saturate_alpha, saturate_do)

                # random hue
                img = self.hue(img, rand_hue, rand_hue_do)

                # random contrast
                if mode == 0:
                    img = self.contrast(img, contrast_alpha, contrast_do)
            images[i] = img
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras

class Weather(object):
    """Apply the following weather augmentations to data.
    Args:
        prob (float): probability to enforce the weather augmentation.
    """

    def __init__(self,
                 prob=0.3,
                 **kwargs):
        snow = iaa.FastSnowyLandscape(
            lightness_threshold=[50, 100],
            lightness_multiplier=(1.2, 2)
            )
        cloud = iaa.Clouds()
        fog = iaa.Fog()
        snow_flakes = iaa.Snowflakes(flake_size=(0.2, 0.4), speed=(0.001, 0.03)) #iaa.imgcorruptlike.Snow(severity=2)# 
        rain = iaa.Rain(speed=(0.1, 0.3), drop_size=(0.1, 0.3))
        # rain_drops = RainDrop_Augmentor()
        self.aug_list = [
            snow, cloud, fog, snow_flakes, rain, 
            #wa.add_sun_flare, wa.darken, wa.random_brightness,
        ]
        self.prob = prob
    
    def aug_with_weather(self, imgs, id):
        weather = self.aug_list[id]
        if id <5:
            return weather(images=imgs)
        else:
            return weather(imgs)

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if random.random() < self.prob:
            select_id = np.random.randint(0, high=len(self.aug_list))  
            images = self.aug_with_weather(images, select_id)        
        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras
        
    
def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h) + 0.5).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w) + 0.5).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return depth


def gray_to_colormap(img, cmap='rainbow', max_value=None):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max_value == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max_value + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap


class LiDarResizeCanonical(object):
    """
    Resize the input to the canonical space first, then resize the input with random sampled size.
    In the first stage, we assume the distance holds while the camera model varies. 
    In the second stage, we aim to simulate the observation in different distance. The camera will move along the optical axis.
    """
    def __init__(self, **kwargs):
        self.ratio_range = kwargs['ratio_range']
        self.canonical_focal = kwargs['focal_length']
        self.crop_size = kwargs['crop_size']
    
    def random_on_canonical_transform(self, image, label, intrinsic, cam_model, to_random_ratio):
        ori_h, ori_w, _ = image.shape
        ori_focal = (intrinsic[0] + intrinsic[1]) / 2.0

        to_canonical_ratio = self.canonical_focal / ori_focal
        to_scale_ratio = to_random_ratio
        resize_ratio = to_canonical_ratio * to_random_ratio
        reshape_h = int(ori_h * resize_ratio + 0.5)
        reshape_w = int(ori_w * resize_ratio + 0.5)

        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        if intrinsic is not None:
            intrinsic = [self.canonical_focal, self.canonical_focal, intrinsic[2]*resize_ratio, intrinsic[3]*resize_ratio]
        if label is not None:
            # number of other labels may be less than that of image
            #label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            label = resize_depth_preserve(label, (reshape_h, reshape_w))
            # scale the label and camera intrinsics
            label = label / to_scale_ratio
        
        if cam_model is not None:
            # Should not directly resize the cam_model.
            # Camera model should be resized in 'to canonical' stage, while it holds in 'random resizing' stage. 
            # cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
        return image, label, intrinsic, cam_model, to_scale_ratio    
    
    def random_on_crop_transform(self, image, label, intrinsic, cam_model, to_random_ratio):
        ori_h, ori_w, _ = image.shape
        crop_h, crop_w = self.crop_size
        ori_focal = (intrinsic[0] + intrinsic[1]) / 2.0

        to_canonical_ratio = self.canonical_focal / ori_focal
        
        # random resize based on the last crop size
        proposal_reshape_h = int(crop_h * to_random_ratio + 0.5) 
        proposal_reshape_w = int(crop_w * to_random_ratio + 0.5)
        resize_ratio_h = proposal_reshape_h / ori_h
        resize_ratio_w = proposal_reshape_w / ori_w
        resize_ratio = min(resize_ratio_h, resize_ratio_w) # resize based on the long edge
        reshape_h = int(ori_h * resize_ratio + 0.5)
        reshape_w = int(ori_w * resize_ratio + 0.5)
        
        to_scale_ratio = resize_ratio / to_canonical_ratio        

        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        if intrinsic is not None:
            intrinsic = [self.canonical_focal, self.canonical_focal, intrinsic[2]*resize_ratio, intrinsic[3]*resize_ratio]
        if label is not None:
            # number of other labels may be less than that of image
            # label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            label = resize_depth_preserve(label, (reshape_h, reshape_w))
            # scale the label and camera intrinsics
            label = label / to_scale_ratio
        
        if cam_model is not None:
            # Should not directly resize the cam_model.
            # Camera model should be resized in 'to canonical' stage, while it holds in 'random resizing' stage. 
            # cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
        return image, label, intrinsic, cam_model, to_scale_ratio          

    def __call__(self, images, labels, intrinsics, cam_models=None, normals=None, other_labels=None, transform_paras=None):
        assert len(images[0].shape) == 3 and len(labels[0].shape) == 2
        assert labels[0].dtype == np.float
        target_focal = (intrinsics[0][0] + intrinsics[0][1]) / 2.0
        target_to_canonical_ratio = self.canonical_focal / target_focal
        target_img_shape = images[0].shape
        to_random_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        to_scale_ratio = 0
        for i in range(len(images)):
            img = images[i]
            label = labels[i] if i < len(labels) else None
            intrinsic = intrinsics[i] if i < len(intrinsics) else None
            cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
            img, label, intrinsic, cam_model, to_scale_ratio = self.random_on_canonical_transform(
                img, label, intrinsic, cam_model, to_random_ratio)
                
            images[i] = img
            if label is not None:
                labels[i] = label
            if intrinsic is not None:
                intrinsics[i] = intrinsic
            if cam_model is not None:
                cam_models[i] = cam_model 
        if normals != None: 
            reshape_h, reshape_w, _ = images[0].shape
            for i, normal in enumerate(normals):
                normals[i] = cv2.resize(normal, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
                
        if other_labels != None: 
            # other labels are like semantic segmentations, instance segmentations, instance planes segmentations...           
            # resize_ratio = target_to_canonical_ratio * to_random_ratio
            # reshape_h = int(target_img_shape[0] * resize_ratio + 0.5)
            # reshape_w = int(target_img_shape[1] * resize_ratio + 0.5)
            reshape_h, reshape_w, _ = images[0].shape
            for i, other_label_i in enumerate(other_labels):
                other_labels[i] = cv2.resize(other_label_i, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
              
        if transform_paras is not None:
            transform_paras.update(label_scale_factor = 1.0/to_scale_ratio)

        return images, labels, intrinsics, cam_models, normals, other_labels, transform_paras



def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map. 
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y =  np.arctan(y_center/ (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model


if __name__ == '__main__':
    img = cv2.imread('/mnt/mldb/raw/62b3ed3455e805efcb28c74b/NuScenes/data_test/samples/CAM_FRONT/n008-2018-08-01-15-34-25-0400__CAM_FRONT__1533152214512404.jpg', -1)
    H, W, _ = img.shape
    label = img[:, :, 0]
    intrinsic = [1000, 1000, W//2, H//2]
    for i in range(20):
        weather_aug = Weather(prob=1.0)
        img_aug,  label, intrinsic, cam_model, ref_images, transform_paras = weather_aug([img, ], [label,], [intrinsic,])
        cv2.imwrite(f'test_aug_{i}.jpg', img_aug[0])
    
    print('Done')
