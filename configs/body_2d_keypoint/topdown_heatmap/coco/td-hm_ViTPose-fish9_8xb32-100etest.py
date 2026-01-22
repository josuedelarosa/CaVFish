_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=200, val_interval=-1)

custom_imports = dict(
    imports=['mmpretrain.models', 'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=512)

default_hooks = dict(
    checkpoint=dict(save_best='AP', rule='greater', max_keep_ckpts=1))

codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# Dataset metadata
# Used by the visualizer and evaluation tools

dataset_info = dict(
    dataset_name='fish_9kpt',
    paper_info={},
    keypoint_info={
        0: dict(name='mouth', id=0, color=[255, 0, 0], type='upper', swap=''),
        1: dict(name='eye', id=1, color=[255, 0, 0], type='upper', swap=''),
        2: dict(name='pectoral', id=2, color=[255, 0, 0], type='upper', swap=''),
        3: dict(name='pelvic', id=3, color=[255, 0, 0], type='upper', swap=''),
        4: dict(name='dorsal', id=4, color=[255, 0, 0], type='upper', swap=''),
        5: dict(name='tailroot', id=5, color=[255, 0, 0], type='lower', swap=''),
        6: dict(name='tailtop', id=6, color=[255, 0, 0], type='lower', swap=''),
        7: dict(name='tailcenter', id=7, color=[255, 0, 0], type='lower', swap=''),
        8: dict(name='tailbottom', id=8, color=[255, 0, 0], type='lower', swap=''),
    },
    skeleton_info={
        0: dict(link=('mouth', 'eye'), id=0, color=[0, 255, 0]),
        1: dict(link=('eye', 'pectoral'), id=1, color=[0, 255, 0]),
        2: dict(link=('pectoral', 'pelvic'), id=2, color=[0, 255, 0]),
        3: dict(link=('pectoral', 'dorsal'), id=3, color=[0, 255, 0]),
        4: dict(link=('pectoral', 'tailroot'), id=4, color=[0, 255, 0]),
        5: dict(link=('tailroot', 'tailtop'), id=5, color=[0, 255, 0]),
        6: dict(link=('tailroot', 'tailcenter'), id=6, color=[0, 255, 0]),
        7: dict(link=('tailroot', 'tailbottom'), id=7, color=[0, 255, 0])
    },
    joint_weights=[1.0] * 9,
    sigmas=[0.025] * 9,
    flip_pairs=[],
    num_keypoints=9,
    upper_body_ids=[0, 1, 2, 3, 4],
    lower_body_ids=[5, 6, 7, 8],
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mae_pretrain_vit_base_20230913.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=9,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

data_root = '/home/arumota_pupils/Josue/1PECES/Josue/FIB/Data/'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]


train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))

# no validation or testing
val_dataloader = None
val_evaluator = None
val_cfg = None
test_evaluator = None
test_cfg = None

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    radius=3,
    line_width=2,
    alpha=0.8,
    # ❌ REMOVE these lines:
    # keypoint_colors=[[255, 0, 0]] * 9,
    # skeleton_links_color=[[0, 255, 0]] * 8
)


test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='PackPoseInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/home/arumota_pupils/Josue/1PECES/Josue/FIB/Data/',
        data_mode='topdown',
        ann_file='annotations/train.json',  # any dummy file, not used
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline,
        test_mode=True
    )
)
