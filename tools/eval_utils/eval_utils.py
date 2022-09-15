import pickle
import time

import numpy as np
import torch
import tqdm
import os 

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
 
        
        metric['recall_rcnn_%s_bin' % str(cur_thresh)] += ret_dict.get('rcnn_%s_bin' % str(cur_thresh), 0)
        metric['recall_rcnn_%s_dis30_bin' % str(cur_thresh)] += ret_dict.get('rcnn_%s_dis30_bin' % str(cur_thresh), 0)
        metric['recall_rcnn_%s_dis50_bin' % str(cur_thresh)] += ret_dict.get('rcnn_%s_dis50_bin' % str(cur_thresh), 0)
        metric['recall_rcnn_%s_disinf_bin' % str(cur_thresh)] += ret_dict.get('rcnn_%s_disinf_bin' % str(cur_thresh), 0)
        metric['recall_rcnn_%s_ped_bin' % str(cur_thresh)] += ret_dict.get('rcnn_%s_ped_bin' % str(cur_thresh), 0)
        metric['recall_rcnn_%s_cyc_bin' % str(cur_thresh)] += ret_dict.get('rcnn_%s_cyc_bin' % str(cur_thresh), 0)


    metric['gt_num'] += ret_dict.get('gt', 0)
    metric['gt_bin_num'] += ret_dict.get('gt_bin', 0) 
    metric['gt_dis30_bin_num'] += ret_dict.get('gt_dis30_bin', 0)
    metric['gt_dis50_bin_num'] += ret_dict.get('gt_dis50_bin', 0)
    metric['gt_disinf_bin_num'] += ret_dict.get('gt_disinf_bin', 0)
    metric['gt_ped_bin_num'] += ret_dict.get('gt_ped_bin', 0)
    metric['gt_cyc_bin_num'] += ret_dict.get('gt_cyc_bin', 0)


    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def statistics_infov2(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['center_%s' % str(cur_thresh)] += ret_dict.get('center_%s' % str(cur_thresh), 0)
        for clses in range(3):
            metric['center_cls%d_%s' % (clses, str(cur_thresh))] += ret_dict.get('center_cls{}_{}'.format(clses, str(cur_thresh)), 0)
    for clses in range(3):
        metric['gt_cls{}'.format(clses)] += ret_dict.get('gt_cls{}'.format(clses), 0)

    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d) / %d' % (metric['center_%s' % str(min_thresh)], metric['gt_num'])


def statistics_infov3(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        for clses in range(3):
            metric['tp_cls%d_%s' % (clses, str(cur_thresh))] += ret_dict.get('tp_cls{}_{}'.format(clses, str(cur_thresh)), 0)
            metric['fp_cls%d_%s' % (clses, str(cur_thresh))] += ret_dict.get('fp_cls{}_{}'.format(clses, str(cur_thresh)), 0)
            metric['detected_cls%d_%s' % (clses, str(cur_thresh))] += ret_dict.get('detected_cls{}_{}'.format(clses, str(cur_thresh)), 0)
        metric['detected_%s' % str(cur_thresh)] += sum(ret_dict.get('detected_cls{}_{}'.format(cc, str(cur_thresh)), 0) for cc in range(3))
    for clses in range(3):
        metric['gt_cls{}'.format(clses)] += ret_dict.get('gt_cls{}'.format(clses), 0)

    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d) / %d' % (metric['detected_%s' % str(min_thresh)], metric['gt_num'])


def eval_statistic(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, mode='pr'):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0, "gt_cls0": 0, "gt_cls1": 0, "gt_cls2": 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        if mode == 'recall':
            metric['center_%s' % str(cur_thresh)] = 0
            for clses in range(3):
                metric['center_cls%d_%s' % (clses, str(cur_thresh))] = 0
        else:
            metric['detected_%s' % str(cur_thresh)] = 0
            for clses in range(3):
                metric['tp_cls%d_%s' % (clses, str(cur_thresh))] = 0
                metric['fp_cls%d_%s' % (clses, str(cur_thresh))] = 0
                metric['detected_cls%d_%s' % (clses, str(cur_thresh))] = 0
    
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        if mode == 'recall':
            statistics_infov2(cfg, ret_dict, metric, disp_dict)
        else:
            statistics_infov3(cfg, ret_dict, metric, disp_dict)
        
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        if mode == 'recall':
            recall = metric['center_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            for clses in range(3):
                this_recall =  metric['center_cls{}_{}'.format(clses, cur_thresh)] / metric['gt_cls{}'.format(clses)]
                logger.info('recall/cls%d/%s: %f' % (clses, cur_thresh, this_recall))
            logger.info('recall/%s: %f' % (cur_thresh, recall))
        else:
            for clses in range(3):
                this_precision =  metric['tp_cls{}_{}'.format(clses, cur_thresh)] / (metric['tp_cls{}_{}'.format(clses, cur_thresh)] + metric['fp_cls{}_{}'.format(clses, cur_thresh)] + 1e-3)
                this_recall1 = metric['detected_cls{}_{}'.format(clses, cur_thresh)] / metric['gt_cls{}'.format(clses)]
                this_recall2 = metric['tp_cls{}_{}'.format(clses, cur_thresh)] / metric['gt_cls{}'.format(clses)]
                logger.info('%dcls%d/%s: R (%.4f, %.4f) P %.4f' % (metric['gt_cls{}'.format(clses)], clses, cur_thresh, this_recall1, this_recall2, this_precision))
    


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)


    metric = {
        'gt_num': 0,
        'gt_bin_num': 0, 
        'gt_dis30_bin_num': 0, 
        'gt_dis50_bin_num': 0, 
        'gt_disinf_bin_num': 0, 
        'gt_ped_bin_num': 0, 
        'gt_cyc_bin_num': 0, 

    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        metric['recall_rcnn_%s_bin' % str(cur_thresh)] = np.zeros(72, dtype=np.int)
        metric['recall_rcnn_%s_dis30_bin' % str(cur_thresh)] = np.zeros(72, dtype=np.int)
        metric['recall_rcnn_%s_dis50_bin' % str(cur_thresh)] = np.zeros(72, dtype=np.int)
        metric['recall_rcnn_%s_disinf_bin' % str(cur_thresh)] = np.zeros(72, dtype=np.int)
        metric['recall_rcnn_%s_ped_bin' % str(cur_thresh)] = np.zeros(72, dtype=np.int)
        metric['recall_rcnn_%s_cyc_bin' % str(cur_thresh)] = np.zeros(72, dtype=np.int)


    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    
    for i, batch_dict in enumerate(dataloader):
    
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
    
        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )

        bs = len(annos)
        det_annos += annos

        if cfg.LOCAL_RANK == 0:
            if i % 1 == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)


    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
