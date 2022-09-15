import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads, patch_encoder, patch_point_encoder
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils

import numpy as np


class AziNormSecond(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module',
            'backbone_2d', 'dense_head',
            'patch_encoder','patch_point_encoder', 'patch_backbone_3d', 'patch_map_to_bev_module',
            'patch_backbone_2d', 'patch_dense_head'
        ]
        
        self.module_list = self.build_networks()
        self.verbose = self.model_cfg.get('verbose', False)

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict


    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict


    def build_patch_encoder(self, model_info_dict):
        if self.model_cfg.get('PATCH_ENCODER', None) is None:
            return None, model_info_dict

        patch_encoder_module = patch_encoder.__all__[self.model_cfg.PATCH_ENCODER.NAME](
            model_cfg=self.model_cfg.PATCH_ENCODER,
            # num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['patch_size'] = np.array(self.model_cfg.PATCH_ENCODER.PATCH_SIZE)
        model_info_dict['module_list'].append(patch_encoder_module)
        return patch_encoder_module, model_info_dict

    def build_patch_point_encoder(self, model_info_dict):
        if self.model_cfg.get('PATCH_POINT_ENCODER', None) is None:
            return None, model_info_dict

        patch_point_encoder_module = patch_point_encoder.__all__[self.model_cfg.PATCH_POINT_ENCODER.NAME](
            model_cfg=self.model_cfg.PATCH_POINT_ENCODER,
            # num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['patch_size'] = np.array(self.model_cfg.PATCH_POINT_ENCODER.PATCH_SIZE)
        model_info_dict['module_list'].append(patch_point_encoder_module)
        return patch_point_encoder_module, model_info_dict


    def build_patch_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('PATCH_BACKBONE_3D', None) is None:
            return None, model_info_dict
        patch_backbone_3d_module = backbones_3d.__all__[self.model_cfg.PATCH_BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.PATCH_BACKBONE_3D,
            input_channels=self.dataset.point_feature_encoder.num_point_features, #model_info_dict['num_point_features'],
            patch_size=model_info_dict['patch_size'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(patch_backbone_3d_module)
        model_info_dict['num_point_features'] = patch_backbone_3d_module.num_point_features
        return patch_backbone_3d_module, model_info_dict

    def build_patch_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('PATCH_MAP_TO_BEV', None) is None:
            return None, model_info_dict

        patch_map_to_bev_module = map_to_bev.__all__[self.model_cfg.PATCH_MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.PATCH_MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(patch_map_to_bev_module)
        model_info_dict['num_bev_features'] = patch_map_to_bev_module.num_bev_features
        return patch_map_to_bev_module, model_info_dict

    def build_patch_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('PATCH_BACKBONE_2D', None) is None:
            return None, model_info_dict

        patch_backbone_2d_module = backbones_2d.__all__[self.model_cfg.PATCH_BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.PATCH_BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(patch_backbone_2d_module)
        model_info_dict['num_bev_features'] = patch_backbone_2d_module.num_bev_features
        return patch_backbone_2d_module, model_info_dict

    def build_patch_dense_head(self, model_info_dict):
        if self.model_cfg.get('PATCH_DENSE_HEAD', None) is None:
            return None, model_info_dict
        patch_dense_head_module = dense_heads.__all__[self.model_cfg.PATCH_DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.PATCH_DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.PATCH_DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            patch_size=model_info_dict['patch_size'],
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )
        model_info_dict['module_list'].append(patch_dense_head_module)
        return patch_dense_head_module, model_info_dict


    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """

        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):

            index_start = batch_dict['patch_num_list'][:index].sum()
            index_end = batch_dict['patch_num_list'][:index+1].sum()
            box_preds = batch_dict['batch_box_preds'][index_start:index_end].reshape(-1, batch_dict['batch_box_preds'].shape[-1])
            cls_preds = batch_dict['batch_cls_preds'][index_start:index_end].reshape(-1, batch_dict['batch_cls_preds'].shape[-1])
            src_box_preds = box_preds
            src_cls_preds = cls_preds


            assert cls_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise Exception
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
   
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }

            if 'first_stage_xy' in batch_dict.keys():
                record_dict.update({'first_stage_xy_per_scene': batch_dict['first_stage_xy'][index]})
            
            if 'gt_boxes' in batch_dict.keys():
                record_dict.update({'gt_boxes_per_scene': batch_dict['gt_boxes'][index]})

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict



    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            recall_dict['gt_bin'] = np.zeros(72, dtype=np.int)
            recall_dict['gt_dis30_bin'] = np.zeros(72, dtype=np.int)
            recall_dict['gt_dis50_bin'] = np.zeros(72, dtype=np.int)
            recall_dict['gt_disinf_bin'] = np.zeros(72, dtype=np.int)

            recall_dict['gt_ped_bin'] = np.zeros(72, dtype=np.int)
            recall_dict['gt_cyc_bin'] = np.zeros(72, dtype=np.int)

            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

                recall_dict['rcnn_%s_bin' % (str(cur_thresh))] = np.zeros(72, dtype=np.int)
                recall_dict['rcnn_%s_dis30_bin' % (str(cur_thresh))] = np.zeros(72, dtype=np.int)
                recall_dict['rcnn_%s_dis50_bin' % (str(cur_thresh))] = np.zeros(72, dtype=np.int)
                recall_dict['rcnn_%s_disinf_bin' % (str(cur_thresh))] = np.zeros(72, dtype=np.int)

                recall_dict['rcnn_%s_ped_bin' % (str(cur_thresh))] = np.zeros(72, dtype=np.int)
                recall_dict['rcnn_%s_cyc_bin' % (str(cur_thresh))] = np.zeros(72, dtype=np.int)

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            gt_bbox_azimuths = torch.atan2(cur_gt[:,1], cur_gt[:,0])
            gt_bin_idx = ((gt_bbox_azimuths + np.pi) / (2 * np.pi / 72)).int()
            recall_dict['gt_bin'] += torch.bincount(gt_bin_idx, minlength=72).cpu().numpy()
            dis30_mask = torch.norm(cur_gt[:,:2], dim=-1) < 30.
            recall_dict['gt_dis30_bin'] += torch.bincount(gt_bin_idx[dis30_mask], minlength=72).cpu().numpy()
            dis50_mask = (torch.norm(cur_gt[:,:2], dim=-1) >= 30.) * (torch.norm(cur_gt[:,:2], dim=-1) < 50.)
            recall_dict['gt_dis50_bin'] += torch.bincount(gt_bin_idx[dis50_mask], minlength=72).cpu().numpy()
            disinf_mask = torch.norm(cur_gt[:,:2], dim=-1) >= 50. 
            recall_dict['gt_disinf_bin'] += torch.bincount(gt_bin_idx[disinf_mask], minlength=72).cpu().numpy()

            ped_mask = cur_gt[:, -1] == 2
            recall_dict['gt_ped_bin'] += torch.bincount(gt_bin_idx[ped_mask], minlength=72).cpu().numpy()
            cyc_mask = cur_gt[:, -1] == 3
            recall_dict['gt_cyc_bin'] += torch.bincount(gt_bin_idx[cyc_mask], minlength=72).cpu().numpy()

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled

                    # dis specific
                    recalled_bin_idx = gt_bin_idx[(iou3d_rcnn.max(dim=0)[0] > cur_thresh)]
                    recall_dict['rcnn_%s_bin' % (str(cur_thresh))] += torch.bincount(recalled_bin_idx, minlength=72).cpu().numpy()
    
                    recalled_dis30_bin_idx = gt_bin_idx[(iou3d_rcnn.max(dim=0)[0] > cur_thresh) * dis30_mask]
                    recall_dict['rcnn_%s_dis30_bin' % (str(cur_thresh))] += torch.bincount(recalled_dis30_bin_idx, minlength=72).cpu().numpy()
                    recalled_dis50_bin_idx = gt_bin_idx[(iou3d_rcnn.max(dim=0)[0] > cur_thresh) * dis50_mask]
                    recall_dict['rcnn_%s_dis50_bin' % (str(cur_thresh))] += torch.bincount(recalled_dis50_bin_idx, minlength=72).cpu().numpy()
                    recalled_disinf_bin_idx = gt_bin_idx[(iou3d_rcnn.max(dim=0)[0] > cur_thresh) * disinf_mask]
                    recall_dict['rcnn_%s_disinf_bin' % (str(cur_thresh))] += torch.bincount(recalled_disinf_bin_idx, minlength=72).cpu().numpy()

                    # cls specific
                    recalled_ped_bin_idx = gt_bin_idx[(iou3d_rcnn.max(dim=0)[0] > cur_thresh) * ped_mask]
                    recall_dict['rcnn_%s_ped_bin' % (str(cur_thresh))] += torch.bincount(recalled_ped_bin_idx, minlength=72).cpu().numpy()
                    recalled_cyc_bin_idx = gt_bin_idx[(iou3d_rcnn.max(dim=0)[0] > cur_thresh) * cyc_mask]
                    recall_dict['rcnn_%s_cyc_bin' % (str(cur_thresh))] += torch.bincount(recalled_cyc_bin_idx, minlength=72).cpu().numpy()

                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict


    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None

        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None

        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    def forward(self, batch_dict):
        for _module_index, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)

            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_patch_head, tb_dict_patch_head = self.patch_dense_head.get_loss()

        tb_dict = {
            'loss_patch_head': loss_patch_head.item(),
            **tb_dict_patch_head,
        }
        loss = loss_patch_head

        return loss, tb_dict, disp_dict
