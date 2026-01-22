default_scope = 'mmpose'
_base_ = ['../../../_base_/default_runtime.py']

# ===== Train schedule =====
train_cfg = dict(max_epochs=300, val_interval=5)

custom_imports = dict(
    imports=[
        'mmpretrain.models',
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
        'custom_transforms',
        'phenolosses.minpheno_distance_loss',
        'phenolosses.vitpose_head_minpheno',
        'mmpose.models.losses.heatmap_loss',
    ],
    allow_failed_imports=False
)

# ===== Optimizer & schedulers =====
optim_wrapper = dict(
    type='AmpOptimWrapper',                 # AMP for VRAM headroom
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=1000, start_factor=0.001, by_epoch=False),  # longer warmup
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        milestones=[210, 260],
        gamma=0.1,
        by_epoch=True
    )
]

# If you change global batch size, update this or remove it and keep LR fixed
auto_scale_lr = dict(base_batch_size=512)

default_hooks = dict(
    checkpoint=dict(save_best='PCK', rule='greater', max_keep_ckpts=1),
    ema=dict(type='EMAHook', momentum=0.0002, priority=49)   # smoother finals
)

# ===== Data/codec & meta =====
# High-precision targets: stride-2 heatmaps -> 256x192
codec = dict(
    type='UDPHeatmap',
    input_size=(512, 384),      # (W, H)
    heatmap_size=(256, 192),    # stride = 2  (x8 upsample from 32x24 feat grid)
    sigma=2.5
)

metainfo = dict(
    dataset_name='fish_20kpt',
    paper_info={},
    keypoint_info={
        0: dict(name='kp1', id=0, color=[255, 0, 0], type='upper', swap=''),
        1: dict(name='kp2', id=1, color=[255, 0, 0], type='upper', swap=''),
        2: dict(name='kp3', id=2, color=[255, 0, 0], type='upper', swap=''),
        3: dict(name='kp4', id=3, color=[255, 0, 0], type='lower', swap=''),
        4: dict(name='kp5', id=4, color=[255, 0, 0], type='lower', swap=''),
        5: dict(name='kp6', id=5, color=[255, 0, 0], type='upper', swap=''),
        6: dict(name='kp7', id=6, color=[255, 0, 0], type='lower', swap=''),
        7: dict(name='kp8', id=7, color=[255, 0, 0], type='upper', swap=''),
        8: dict(name='kp9', id=8, color=[255, 0, 0], type='lower', swap=''),
        9: dict(name='kp10', id=9, color=[255, 0, 0], type='upper', swap=''),
        10: dict(name='kp11', id=10, color=[255, 0, 0], type='lower', swap=''),
        11: dict(name='kp12', id=11, color=[255, 0, 0], type='upper', swap=''),
        12: dict(name='kp13', id=12, color=[255, 0, 0], type='upper', swap=''),
        13: dict(name='kp14', id=13, color=[255, 0, 0], type='upper', swap=''),
        14: dict(name='kp15', id=14, color=[255, 0, 0], type='lower', swap=''),
        15: dict(name='kp16', id=15, color=[255, 0, 0], type='lower', swap=''),
        16: dict(name='kp17', id=16, color=[255, 0, 0], type='lower', swap=''),
        17: dict(name='kp18', id=17, color=[255, 0, 0], type='upper', swap=''),
        18: dict(name='kp19', id=18, color=[255, 0, 0], type='upper', swap=''),
        19: dict(name='kp20', id=19, color=[255, 0, 0], type='lower', swap='')
    },
    skeleton_info={},
    joint_weights=[1.0] * 20,
    sigmas=[0.025] * 20,
    flip_pairs=[],  # keep empty unless you truly have bilateral symmetry
    num_keypoints=20,
    upper_body_ids=[0, 1, 2, 4, 6, 8, 10, 11, 12, 16, 17],
    lower_body_ids=[3, 13, 14, 15, 5, 18, 19, 7, 9]
)

# ===== Model =====
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(512, 384),     # match codec.input_size (W, H)
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mae_pretrain_vit_base_20230913.pth'
        ),
    ),
    head=dict(
        type='MinPhenoHead',     # custom extended head
        in_channels=768,         # ViT-Base channels
        out_channels=20,         # 20 keypoints
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        # three deconvs (k=4,s=2) → x8 upsample → 256x192 heatmaps
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        decoder=codec,           # pass codec so target sizes/decoder align
        loss_pheno=dict(
            type='MinPhenotypeDistanceLoss',
            pairs=[(0,1), (3,2), (4,5), (6,7), (8,9), (10,11),
                   (12,3), (0,3), (0,13), (14,15), (14,3),
                   (0,16), (2,17), (18,19)],
            degree_normalize=True,
            scale_by_SL=False,
            normalization="min_gt",
            percentile=None,
            detach_scale=True,
            clamp_min=1e-4,      # tighter floor for stability/precision
            clamp_max=None,
            beta=15.0            # crisper soft-argmax for PMP
        ),
        alpha_pheno=1e-2
    ),
    test_cfg=dict(
        flip_test=False,         # keep off unless flip_pairs is correct
        flip_mode='heatmap',
        shift_heatmap=False
    )
)

# ===== Data & pipelines =====
data_root = '/data/Datasets/Fish/CavFish'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.15),
    dict(type='RandomBBoxTransform'),  # you can widen scale/rotate if desired
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

test_pipeline = val_pipeline

# Adjust batch_size to fill VRAM; 24 is a safe default for 24GB with AMP+with_cp
train_dataloader = dict(
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='fish20kpt_all_train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
        metainfo=metainfo
    )
)

# If you have a separate val/test file, replace ann_file below.
val_dataloader = dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='fish20kpt_all_train.json',
        data_prefix=dict(img=''),
        pipeline=val_pipeline,
        metainfo=metainfo
    )
)

test_dataloader = dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='fish20kpt_all_train.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
        metainfo=metainfo
    )
)

# ===== Evaluators & cfg =====
val_evaluator = [dict(type='PCKAccuracy', thr=0.05), dict(type='AUC')]
test_evaluator = [dict(type='PCKAccuracy', thr=0.05), dict(type='AUC')]

val_cfg = dict()
test_cfg = dict()

# ===== Visualization =====
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    radius=3,
    line_width=2,
    alpha=0.8
)

# ===== Working dir =====
work_dir = '/data/Pupils/Josue/weights/Fish/ViTPose_20kpt_minphenoloss_res2x'
