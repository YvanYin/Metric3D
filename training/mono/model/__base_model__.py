import torch
import torch.nn as nn
from mono.utils.comm import get_func
import numpy as np
import torch.nn.functional as F


class BaseDepthModel(nn.Module):
    def __init__(self, cfg, criterions, **kwards):
        super(BaseDepthModel, self).__init__()   
        model_type = cfg.model.type
        self.depth_model = get_func('mono.model.model_pipelines.' + model_type)(cfg)

        self.criterions_main = criterions['decoder_losses'] if criterions and 'decoder_losses' in criterions else None
        self.criterions_auxi = criterions['auxi_losses'] if criterions and 'auxi_losses' in criterions else None
        self.criterions_pose = criterions['pose_losses'] if criterions and 'pose_losses' in criterions else None
        self.criterions_gru = criterions['gru_losses'] if criterions and 'gru_losses' in criterions else None
        try:
            self.downsample = cfg.prediction_downsample
        except:
            self.downsample = None

        self.training = True

    def forward(self, data):
        if self.downsample != None:
            self.label_downsample(self.downsample, data)
        
        output = self.depth_model(**data)

        losses_dict = {}
        if self.training:
            output.update(data)
            losses_dict = self.get_loss(output)

        if self.downsample != None:
            self.pred_upsample(self.downsample, output)

        return output['prediction'], losses_dict, output['confidence']
    
    def inference(self, data):
        with torch.no_grad():
            output = self.depth_model(**data)
            output.update(data)

        if self.downsample != None:
            self.pred_upsample(self.downsample, output)

            output['dataset'] = 'wild'
        return output

    def get_loss(self, paras):
        losses_dict = {}
        # Losses for training
        if self.training:
            # decode branch
            losses_dict.update(self.compute_decoder_loss(paras))
            # auxilary branch
            losses_dict.update(self.compute_auxi_loss(paras))
            # pose branch
            losses_dict.update(self.compute_pose_loss(paras))
            # GRU sequence branch
            losses_dict.update(self.compute_gru_loss(paras))

            total_loss = sum(losses_dict.values())
            losses_dict['total_loss'] = total_loss
        return losses_dict
    
    def compute_gru_loss(self, paras_):
        losses_dict = {}
        if self.criterions_gru is None or len(self.criterions_gru) == 0:
            return losses_dict
        paras = {k:v for k,v in paras_.items() if k!='prediction' and k!='prediction_normal'}
        n_predictions = len(paras['predictions_list'])
        for i, pre in enumerate(paras['predictions_list']):
            if i == n_predictions-1:
                break
            #if i % 3 != 0:
                #continue
            if 'normal_out_list' in paras.keys():
                pre_normal = paras['normal_out_list'][i]
            else:
                pre_normal = None
            iter_dict = self.branch_loss(
                prediction=pre,
                prediction_normal=pre_normal,
                criterions=self.criterions_gru,
                branch=f'gru_{i}',
                **paras
            )
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = 0.9**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            iter_dict = {k:v*i_weight for k,v in iter_dict.items()}
            losses_dict.update(iter_dict)
        return losses_dict

    def compute_decoder_loss(self, paras):
        losses_dict = {}
        decode_losses_dict = self.branch_loss(
            criterions=self.criterions_main, 
            branch='decode',
            **paras
            )
        return decode_losses_dict
    
    def compute_auxi_loss(self, paras):
        losses_dict = {}
        if len(self.criterions_auxi) == 0:
            return losses_dict
        args = dict(
            target=paras['target'],
            data_type=paras['data_type'],
            sem_mask=paras['sem_mask'],
        )
        for i, auxi_logit in enumerate(paras['auxi_logit_list']): 
            auxi_losses_dict = self.branch_loss(
                prediction=paras['auxi_pred'][i],
                criterions=self.criterions_auxi,
                pred_logit=auxi_logit,     
                branch=f'auxi_{i}',
                **args
                )
            losses_dict.update(auxi_losses_dict)
        return losses_dict
    
    def compute_pose_loss(self, paras):
        losses_dict = {}
        if self.criterions_pose is None or len(self.criterions_pose) == 0:
            return losses_dict
        # valid_flg = paras['tmpl_flg']
        # if torch.sum(valid_flg) == 0:
        #     return losses_dict
        # else:
        #     # sample valid batch
        #     samples = {}
        #     for k, v in paras.items():
        #         if isinstance(v, torch.Tensor):
        #             samples.update({k: v[valid_flg]})
        #         elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
        #             samples.update({k: [i[valid_flg] for i in v]})
        for loss_method in self.criterions_pose:
            loss_tmp = loss_method(**paras)
            losses_dict['pose_' + loss_method._get_name()] = loss_tmp
        return losses_dict

    def branch_loss(self, prediction, pred_logit, criterions, branch='decode', **kwargs):   
        B, _, _, _ = prediction.shape
        losses_dict = {}
        args = dict(pred_logit=pred_logit)
        
        target = kwargs.pop('target')
        args.update(kwargs)

        # data type for each batch
        batches_data_type = np.array(kwargs['data_type']) 
        # batches_data_names = np.array(kwargs['dataset']) 

        # resize the target
        # if target.shape[2] != prediction.shape[2] and target.shape[3] != prediction.shape[3]:
        #     _, _, H, W = prediction.shape
        #     target = nn.functional.interpolate(target, (H,W), mode='nearest')

        mask = target > 1e-8
        for loss_method in criterions:
            # sample batches, which satisfy the loss requirement for data types
            new_mask = self.create_mask_as_loss(loss_method, mask, batches_data_type)

            loss_tmp = loss_method(
                prediction=prediction, 
                target=target, 
                mask=new_mask, 
                **args)                
            losses_dict[branch + '_' + loss_method._get_name()] = loss_tmp
        return losses_dict
    
    def create_mask_as_loss(self, loss_method, mask, batches_data_type):
        data_type_req = np.array(loss_method.data_type)[:, None]
        batch_mask = torch.tensor(np.any(data_type_req == batches_data_type, axis=0), device="cuda") #torch.from_numpy(np.any(data_type_req == batches_data_type, axis=0)).cuda()
        new_mask = mask * batch_mask[:, None, None, None]
        return new_mask
    
    def label_downsample(self, downsample_factor, data_dict):
        scale_factor = float(1.0 / downsample_factor)
        downsample_target = F.interpolate(data_dict['target'], scale_factor=scale_factor)
        downsample_stereo_depth = F.interpolate(data_dict['stereo_depth'], scale_factor=scale_factor)

        data_dict['target'] = downsample_target
        data_dict['stereo_depth'] = downsample_stereo_depth

        return data_dict

    def pred_upsample(self, downsample_factor, data_dict):
        scale_factor = float(downsample_factor)
        upsample_prediction = F.interpolate(data_dict['prediction'], scale_factor=scale_factor).detach()
        upsample_confidence = F.interpolate(data_dict['confidence'], scale_factor=scale_factor).detach()

        data_dict['prediction'] = upsample_prediction
        data_dict['confidence'] = upsample_confidence

        return data_dict




    # def mask_batches(self, prediction, target, mask, batches_data_names, data_type_req):
    #     """
    #     Mask the data samples that satify the loss requirement.
    #     Args:
    #         data_type_req (str): the data type required by a loss. 
    #         batches_data_names (list): the list of data types in a batch. 
    #     """
    #     batch_mask = np.any(data_type_req == batches_data_names, axis=0)
    #     prediction = prediction[batch_mask]
    #     target = target[batch_mask]
    #     mask = mask[batch_mask]
    #     return prediction, target, mask, batch_mask

    # def update_mask_g8(self, target, mask,  prediction, batches_data_names, absRel=0.5):
    #     data_type_req=np.array(['Golf8_others'])[:, None]
        
    #     pred, target, mask_sample, batch_mask = self.mask_batches(prediction, target, mask, batches_data_names, data_type_req)
    #     if pred.numel() == 0:
    #         return mask
    #     scale_batch = []
    #     for i in range(mask_sample.shape[0]):
    #         scale = torch.median(target[mask_sample]) / (torch.median(pred[mask_sample]) + 1e-8)
    #         abs_rel = torch.abs(pred[i:i+1, ...] * scale - target[i:i+1, ...]) / (pred[i:i+1, ...] * scale + 1e-6)
    #         if target[i, ...][target[i, ...]>0].min() < 0.041:
    #             mask_valid_i = ((abs_rel < absRel) | ((target[i:i+1, ...]<0.02) & (target[i:i+1, ...]>1e-6)))  & mask_sample[i:i+1, ...]
    #         else:
    #             mask_valid_i = mask_sample[i:i+1, ...]
    #         mask_sample[i:i+1, ...] = mask_valid_i
    #         # print(target.max(), target[target>0].min())
    #         # self.visual_g8(target, mask_valid_i)
    #     mask[batch_mask] = mask_sample
    #     return mask
    
    # def update_mask_g8_v2(self, target, mask,  prediction, batches_data_names,):
    #     data_type_req=np.array(['Golf8_others'])[:, None]
        
    #     pred, target, mask_sample, batch_mask = self.mask_batches(prediction, target, mask, batches_data_names, data_type_req)
    #     if pred.numel() == 0:
    #         return mask
        
    #     raw_invalid_mask = target < 1e-8
    #     target[raw_invalid_mask] = 1e8
    #     kernal = 31
    #     pool = min_pool2d(target, kernal)
    #     diff = target- pool
    #     valid_mask = (diff < 0.02)  &  mask_sample & (target<0.3)
    #     target_min = target.view(target.shape[0], -1).min(dim=1)[0]
    #     w_close = target_min < 0.04
    #     valid_mask[~w_close] = mask_sample[~w_close]
    #     mask[batch_mask]= valid_mask

    #     target[raw_invalid_mask] = -1
    #     #self.visual_g8(target, mask[batch_mask])
    #     return mask
        
    # def visual_g8(self, gt, mask):
    #     import matplotlib.pyplot as plt
    #     from mono.utils.transform import gray_to_colormap
    #     gt = gt.cpu().numpy().squeeze()
    #     mask = mask.cpu().numpy().squeeze()
    #     if gt.ndim >2:
    #         gt = gt[0, ...]
    #         mask = mask[0, ...]
    #     name = np.random.randint(1000000)
    #     print(gt.max(), gt[gt>0].min(), name)
    #     gt_filter = gt.copy()
    #     gt_filter[~mask] = 0
    #     out = np.concatenate([gt, gt_filter], axis=0)
    #     out[out<0] = 0
    #     o = gray_to_colormap(out)
    #     o[out<1e-8]=0
        
    #     plt.imsave(f'./tmp/{name}.png', o)     
        
        
        


def min_pool2d(tensor, kernel, stride=1):
    tensor = tensor * -1.0
    tensor = F.max_pool2d(tensor, kernel, padding=kernel//2, stride=stride)
    tensor = -1.0 * tensor
    return tensor