import torch
import torch.nn as nn

from ...utils import common_utils
import numpy as np
import copy

class RotatePatchLightEncoderForSniper(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size, **kwargs):
        super().__init__()

        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.patch_size = torch.tensor(model_cfg.PATCH_SIZE).cuda()
        self.model_cfg = model_cfg

        config = model_cfg

        self.using_points = getattr(model_cfg, 'using_points', False)


    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
           
            **kwargs:

        Returns:
        """

        batch_dict['patch_size'] = self.patch_size
        batch_dict['voxel_size'] = self.voxel_size
        batch_dict['point_cloud_range'] = self.point_cloud_range



        assert 'patch_voxels' in batch_dict.keys() and \
            'patch_voxel_coords' in batch_dict.keys() and \
            'patch_voxel_num_points' in batch_dict.keys() and \
            'valid_patch_center_array' in  batch_dict.keys() and \
            'valid_patch_num_array'  in  batch_dict.keys()


        # function as vfe
        patch_voxel_features, patch_voxel_num_points = batch_dict['patch_voxels'], batch_dict['patch_voxel_num_points']
        points_mean = patch_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(patch_voxel_num_points.view(-1, 1), min=1.0).type_as(patch_voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['patch_voxel_features'] = points_mean.contiguous()


        _voxel_idx_accu = batch_dict['patch_voxel_num_array'].reshape(-1).cumsum(0)
        batch_dict['patch_voxel_coords'][:, 0] = 0
        for i in range(1, batch_dict['patch_voxel_num_array'].numel()):
            batch_dict['patch_voxel_coords'][_voxel_idx_accu[i - 1].int():, 0] = i
        

        if self.using_points:

            _point_idx_accu = batch_dict['patch_point_num_array'].reshape(-1).cumsum(0)
            batch_dict['patch_points'][:, 0] = 0
            for i in range(1, batch_dict['patch_point_num_array'].numel()):
                batch_dict['patch_points'][_point_idx_accu[i - 1].int():, 0] = i
        
            batch_dict['points'] = batch_dict['patch_points']


        batch_dict['voxel_coords'] = batch_dict['patch_voxel_coords']
        batch_dict['voxel_features'] = batch_dict['patch_voxel_features']


        batch_dict['patch_num_list'] =  batch_dict['valid_patch_num_array'].int()
        batch_dict['patch_center_xyz_list'] = []
        _start = 0; _end = 0
        for _idx, _valid_patch_num in enumerate(batch_dict['valid_patch_num_array']):
            
            _end += _valid_patch_num.int().item()

            _temp = batch_dict['valid_patch_center_array'][_start: _end]
            _z_padding = torch.zeros((_temp.shape[0], 1), dtype=torch.float32).cuda()
            _temp = torch.cat([_temp, _z_padding], dim=-1)
            batch_dict['patch_center_xyz_list'].append(_temp)

            _start = _end
        


        if not self.training and getattr(self.model_cfg, 'split_infer', False):

            assert batch_dict['batch_size'] == 1



            _half_idx = int(batch_dict['patch_num_list'][0].item() / 2)


            batch_dict_1st = copy.deepcopy(batch_dict)
            batch_dict_2nd = copy.deepcopy(batch_dict)


            _1st_mask = batch_dict['voxel_coords'][:, 0] < _half_idx
            _2nd_mask = ~_1st_mask

            batch_dict_1st['voxel_coords'] =  batch_dict['voxel_coords'][_1st_mask]
            batch_dict_1st['voxel_features'] = batch_dict['voxel_features'][_1st_mask]

            batch_dict_2nd['voxel_coords'] =  batch_dict['voxel_coords'][_2nd_mask]
            batch_dict_2nd['voxel_coords'][:,0] -= _half_idx
            batch_dict_2nd['voxel_features'] = batch_dict['voxel_features'][_2nd_mask]


            batch_dict_1st['patch_num_list'] = batch_dict['patch_num_list'] * 0 + _half_idx
            batch_dict_2nd['patch_num_list'] = batch_dict['patch_num_list'] - _half_idx

            batch_dict_1st['patch_center_xyz_list'] = [batch_dict['patch_center_xyz_list'][0][:_half_idx]]
            batch_dict_2nd['patch_center_xyz_list'] = [batch_dict['patch_center_xyz_list'][0][_half_idx:]]

            if self.using_points:
                _1st_point_mask = batch_dict['points'][:, 0] < _half_idx
                _2nd_point_mask = ~_1st_point_mask
                batch_dict_1st['points'] = batch_dict['points'][_1st_point_mask]
                batch_dict_2nd['points'] = batch_dict['points'][_2nd_point_mask]
                batch_dict_2nd['points'][:,0] -= _half_idx


            return batch_dict_1st, batch_dict_2nd, batch_dict_1st

        return batch_dict
