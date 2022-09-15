from functools import partial

import numpy as np

from ...utils import box_utils, common_utils

class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def mask_points_and_boxes_outside_range_including_height(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range_including_height, config=config)
        mask = common_utils.mask_points_by_range_including_height(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
    
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict


    def transform_points_to_voxels_waymo_sniper(self, data_dict=None, config=None, voxel_generator=None):

        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=config.sub_point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            self.voxel_size = config.VOXEL_SIZE
            self.sub_point_cloud_range = config.sub_point_cloud_range
            _x = (np.arange(config.center_size[0]).astype(np.float) + 0.5) * self.voxel_size[0] * config.region_select_stride  + config.center_lower_bound
            _y = (np.arange(config.center_size[1]).astype(np.float) + 0.5) * self.voxel_size[1] * config.region_select_stride  + config.center_lower_bound
            grid_x, grid_y = np.meshgrid(_x, _y)
            self.fixed_patch_center = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
            
            self.training_patch_maximum = config.training_patch_maximum
            self.in_patch_point_minimum = getattr(config, 'in_patch_point_minimum', 5)
            self.rotate_patch = getattr(config, 'rotate_patch', True)
            self.patch_shape =  getattr(config, 'patch_shape', 'circle')
            self.sub_patch_center = getattr(config, 'sub_patch_center', True)
            return partial(self.transform_points_to_voxels_waymo_sniper, voxel_generator=voxel_generator)

        points = data_dict['points']
        ret_list = []
        points_list = []
        per_patch_point_num_list = []
            
        voxels_list = []
        coordinates_list = []
        per_voxel_point_num_list = []
        per_patch_voxel_num_list = []
        valid_patch_center_list = []

        # TODO naive implementation with for loop, too slow, reimplemented with DALI or CUDA to speed up
        for _fixed_patch_center in self.fixed_patch_center[np.random.permutation(len(self.fixed_patch_center))]:

            if self.training and len(valid_patch_center_list) >= self.training_patch_maximum:
                break
        
            if self.patch_shape == 'circle':
                _in_patch_idx = np.linalg.norm(points[:, :2] - _fixed_patch_center, axis=1) < self.sub_point_cloud_range[3]

            elif self.patch_shape == 'square':
                dx = points[:,0] - _fixed_patch_center[0]
                dy = points[:,1] - _fixed_patch_center[1]
                _pos_patch_angle = np.arctan2(_fixed_patch_center[1], _fixed_patch_center[0])

                _patch_cos = np.cos(_pos_patch_angle)
                _patch_sin = np.sin(_pos_patch_angle)
                y_ = dy * _patch_cos - dx * _patch_sin #in patch coord
                x_ = dy * _patch_sin + dx * _patch_cos
                _in_patch_idx = (x_ > self.sub_point_cloud_range[0]) * (x_ < self.sub_point_cloud_range[3]) \
                        * (y_ > self.sub_point_cloud_range[1]) * (y_ < self.sub_point_cloud_range[4]) 
            else:
                raise Exception

            if _in_patch_idx.sum() < self.in_patch_point_minimum:
                continue
            else:
                _points = points[_in_patch_idx]
                _points[:, :2] -= _fixed_patch_center

            if self.rotate_patch:
                _neg_patch_angle = - np.arctan2(_fixed_patch_center[1], _fixed_patch_center[0])
                _points = common_utils.rotate_points_along_z(_points[np.newaxis],  np.array([_neg_patch_angle]))[0]
  
            voxel_output = voxel_generator.generate(_points)

            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            if self.sub_patch_center == False: # add back
                # ((voxels.reshape(-1,5)[:,:2])[voxels.reshape(-1,5)[:,:2].sum(-1)!=0]+_fixed_patch_center).max(0)
                voxels[:, :, :2] += (voxels[:, :, :2].sum(-1, keepdims=True) != 0) * _fixed_patch_center


            if not data_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
            
            points_list.append(_points)
            per_patch_point_num_list.append(_points.shape[0])
            
            voxels_list.append(voxels)
            coordinates_list.append(coordinates)
            per_voxel_point_num_list.append(num_points)
            per_patch_voxel_num_list.append(voxels.shape[0])
            valid_patch_center_list.append(_fixed_patch_center)

        data_dict['patch_points'] = np.concatenate(points_list, axis=0)
        data_dict['patch_point_num_array'] = np.array(per_patch_point_num_list, dtype=np.int)

        data_dict['patch_voxels'] = np.concatenate(voxels_list, axis=0)
        data_dict['patch_voxel_num_array'] = np.array(per_patch_voxel_num_list, dtype=np.int)

        data_dict['patch_voxel_coords'] = np.concatenate(coordinates_list, axis=0)
        data_dict['patch_voxel_num_points'] = np.concatenate(per_voxel_point_num_list, axis=0)

        data_dict['valid_patch_center_array'] = np.array(valid_patch_center_list)
        data_dict['valid_patch_num_array'] =  np.array([data_dict['valid_patch_center_array'].shape[0]], dtype=np.int)

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
