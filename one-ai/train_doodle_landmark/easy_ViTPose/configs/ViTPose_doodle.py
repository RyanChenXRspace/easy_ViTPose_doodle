from .ViTPose_common import *

# Channel configuration
channel_cfg = dict(
    num_output_channels=25,
    dataset_joints=25,
    dataset_channel=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24], ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24])

data_cfg:dict = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)

data_root = './datasets/Doodle'
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/train.json',
        # img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/train.json',
        # img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/train.json',
        # img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg)
)

# Set models channels
data_cfg['num_output_channels'] = channel_cfg['num_output_channels']
data_cfg['num_joints'] = channel_cfg['dataset_joints']
data_cfg['dataset_channel'] = channel_cfg['dataset_channel']
data_cfg['inference_channel'] = channel_cfg['inference_channel']

names = ['small', 'base', 'large', 'huge']
for name in names:
    globals()[f'model_{name}']['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
