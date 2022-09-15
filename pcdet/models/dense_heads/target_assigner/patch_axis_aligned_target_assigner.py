import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils

from ....utils import common_utils

class PatchAxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
         
        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        self.seperate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        if self.seperate_multihead:
            rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
            self.gt_remapping = {}
            for rpn_head_cfg in rpn_head_cfgs:
                for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
                    self.gt_remapping[name] = idx + 1


        self.model_cfg = model_cfg
        self.rotate_patch = model_cfg.get('rotate_patch', True)
        self.maskout_margin = model_cfg.get('maskout_margin', True)
        self.margin_thre = model_cfg.get('margin_thre', 7.)

    def assign_targets(self, all_anchors, patch_num_list, patch_center_xyz_list, gt_boxes_with_classes, cls_preds, box_preds):

        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []
        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    if self.seperate_multihead:
                        selected_classes = cur_gt_classes[mask].clone()
                        if len(selected_classes) > 0:
                            new_cls_id = self.gt_remapping[anchor_class_name]
                            selected_classes[:] = new_cls_id
                    else:
                        selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]

                single_target = self.assign_targets_single(anchors, 
                    patch_num_list[k], 
                    patch_center_xyz_list[k], 
                    cur_gt[mask], 
                    selected_classes, 
                    None, 
                    None,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name])

                target_list.append(single_target)
            if self.use_multihead:
                raise Exception
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(patch_num_list[k], *feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(patch_num_list[k], *feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(patch_num_list[k], *feature_map_size, -1) for t in target_list]
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(patch_num_list[k], -1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(patch_num_list[k], -1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(patch_num_list[k], -1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_labels = torch.cat(cls_labels, dim=0)
        reg_weights = torch.cat(reg_weights, dim=0)

        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        return all_targets_dict

    def assign_targets_single(self, 
                         anchors,
                         patch_num,
                         patch_center_xyz,
                         gt_boxes,
                         gt_classes,
                         cls_preds, #useless
                         box_preds, #useless
                         matched_threshold=0.6,
                         unmatched_threshold=0.45
                        ):
        # duplicate anchors for every patch
        ori_anchors = anchors.clone()

        if self.maskout_margin:
            _maskout_idx = ori_anchors.unsqueeze(0).repeat(patch_num, 1, 1)[:,:,:3].norm(dim=-1).reshape(-1) > self.margin_thre
        else:
            _maskout_idx = None

        anchors = anchors.unsqueeze(0).repeat(patch_num, 1, 1) 

        if self.rotate_patch:
            _pos_patch_angle = torch.atan2(patch_center_xyz[:,1].detach(), patch_center_xyz[:,0].detach())
            anchors[:, :, :3] = common_utils.rotate_points_along_z(anchors[:, :, :3], _pos_patch_angle)
            anchors[:, :, 6] += _pos_patch_angle[:, None]
            anchors[:, :, :3] += patch_center_xyz[:, None, :]
        else:
            anchors[:, :, :3] += patch_center_xyz[:, None, :]

        anchors = anchors.reshape(-1, ori_anchors.shape[-1])
        
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if _maskout_idx is not None:
            anchors[_maskout_idx, 3:6] = 0

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou_bev(anchors[:, 0:7], gt_boxes[:, 0:7])

            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ]

            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        if _maskout_idx is not None:
            labels[_maskout_idx] = -1

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]

            if self.rotate_patch:
                _fg_patch_center_xyz = patch_center_xyz[(fg_inds/ori_anchors.shape[0]).long()]
                _neg_patch_angle = - torch.atan2(_fg_patch_center_xyz[:,1].detach(), _fg_patch_center_xyz[:,0].detach())
                fg_gt_boxes[:, :3] = common_utils.rotate_points_along_z(fg_gt_boxes[:, None, :3], _neg_patch_angle).squeeze(1)
                fg_anchors[:, :3] = common_utils.rotate_points_along_z(fg_anchors[:, None, :3], _neg_patch_angle).squeeze(1)
            else:
                pass

            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)
    
        reg_weights = anchors.new_zeros((num_anchors,))

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
