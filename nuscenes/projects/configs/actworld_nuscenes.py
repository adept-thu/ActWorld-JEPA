version = 'trainval'
length = dict(trainval=28130, mini=323)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/actworld_nuscenes'
total_batch_size = 12
num_gpus = 4
batch_size = 3
num_iters_per_epoch = 2344
num_epochs = 5
checkpoint_epoch_interval = 1
checkpoint_config = dict(interval=2000, max_keep_ckpts=3, save_last=True)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
load_from = None
resume_from = None
workflow = [('train', 1)]
input_shape = (704, 256)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_class_names = ['ped_crossing', 'divider', 'boundary']
num_classes = 10
num_map_classes = 3
roi_size = (30, 60)
num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 12
ego_fut_mode = 6
queue_length = 4
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = 4
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True
task_config = dict(with_det=True, with_map=True, with_motion_plan=True)
model = dict(
    type='ActWorldMomADv2SixSecond',
    use_grid_mask=True,
    use_deformable_func=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style='pytorch',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        pretrained='ckpt/resnet50-19c8e357.pth'),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=0,
        out_channels=256,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    depth_branch=dict(
        type='DenseDepthNet',
        embed_dims=256,
        num_depth_layers=3,
        loss_weight=0.2),
    head=dict(
        type='SparseDriveHead',
        task_config=dict(with_det=True, with_map=True, with_motion_plan=True),
        det_head=dict(
            type='Sparse4DHead_roboAD',
            cls_threshold_to_reg=0.05,
            decouple_attn=True,
            instance_bank=dict(
                type='InstanceBank',
                num_anchor=900,
                embed_dims=256,
                anchor='data/kmeans/kmeans_det_900.npy',
                anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
                num_temp_instances=600,
                confidence_decay=0.6,
                feat_grad=False),
            anchor_encoder=dict(
                type='SparseBox3DEncoder',
                vel_dims=3,
                embed_dims=[128, 32, 32, 64],
                mode='cat',
                output_fc=False,
                in_loops=1,
                out_loops=4),
            num_single_frame_decoder=1,
            operation_order=[
                'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn',
                'norm', 'deformable', 'ffn', 'norm', 'refine', 'temp_gnn',
                'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine',
                'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm',
                'refine', 'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn',
                'norm', 'refine', 'temp_gnn', 'gnn', 'norm', 'deformable',
                'ffn', 'norm', 'refine'
            ],
            temp_graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=512,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            deformable_model=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparseBox3DKeyPointsGenerator',
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                               [0, 0, -0.45]])),
            refine_layer=dict(
                type='SparseBox3DRefinementModule',
                embed_dims=256,
                num_cls=10,
                refine_yaw=True,
                with_quality_estimation=True),
            sampler=dict(
                type='SparseBox3DTarget',
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[
                    2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                cls_wise_reg_weights=dict(
                    {9: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]})),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_reg=dict(
                type='SparseBox3DLoss',
                loss_box=dict(type='L1Loss', loss_weight=0.25),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True),
                loss_yawness=dict(type='GaussianFocalLoss'),
                cls_allow_reverse=[5]),
            decoder=dict(type='SparseBox3DDecoder'),
            reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        map_head=dict(
            type='Sparse4DHead',
            cls_threshold_to_reg=0.05,
            decouple_attn=False,
            instance_bank=dict(
                type='InstanceBank',
                num_anchor=100,
                embed_dims=256,
                anchor='data/kmeans/kmeans_map_100.npy',
                anchor_handler=dict(type='SparsePoint3DKeyPointsGenerator'),
                num_temp_instances=33,
                confidence_decay=0.6,
                feat_grad=True),
            anchor_encoder=dict(
                type='SparsePoint3DEncoder', embed_dims=256, num_sample=20),
            num_single_frame_decoder=1,
            operation_order=[
                'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine',
                'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm',
                'refine', 'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn',
                'norm', 'refine', 'temp_gnn', 'gnn', 'norm', 'deformable',
                'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm',
                'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn',
                'norm', 'deformable', 'ffn', 'norm', 'refine'
            ],
            temp_graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=512,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            deformable_model=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparsePoint3DKeyPointsGenerator',
                    embed_dims=256,
                    num_sample=20,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023)),
            refine_layer=dict(
                type='SparsePoint3DRefinementModule',
                embed_dims=256,
                num_sample=20,
                num_cls=3),
            sampler=dict(
                type='SparsePoint3DTarget',
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(
                            type='LinesL1Cost',
                            weight=10.0,
                            beta=0.01,
                            permute=True))),
                num_cls=3,
                num_sample=20,
                roi_size=(30, 60)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_reg=dict(
                type='SparseLineLoss',
                loss_line=dict(
                    type='LinesL1Loss', loss_weight=10.0, beta=0.01),
                num_sample=20,
                roi_size=(30, 60)),
            decoder=dict(type='SparsePoint3DDecoder'),
            reg_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0
            ],
            gt_cls_key='gt_map_labels',
            gt_reg_key='gt_map_pts',
            gt_id_key='map_instance_id',
            with_instance_id=False,
            task_prefix='map'),
        motion_plan_head=dict(
            type='MotionPlanningHeadmambaADNoFlow_6s',
            fut_ts=12,
            fut_mode=6,
            ego_fut_ts=12,
            ego_fut_mode=6,
            motion_anchor='data/kmeans/kmeans_motion_6.npy',
            plan_anchor='data/kmeans/kmeans_plan_6.npy',
            embed_dims=256,
            decouple_attn=True,
            use_rescore=True,
            instance_queue=dict(
                type='InstanceQueue',
                embed_dims=256,
                queue_length=4,
                tracking_threshold=0.2,
                feature_map_scale=(8.0, 22.0)),
            mamba_model=dict(
                type='PureTorchSelectiveSSMQueryEncoder',
                d_model=256,
                d_state=16,
                expand=2,
                d_conv=4,
                num_layers=2,
                dropout=0.1),
            use_ssm=True,
            operation_order=[
                'temp_gnn', 'gnn', 'norm', 'cross_gnn', 'norm', 'ffn', 'norm',
                'temp_gnn', 'gnn', 'norm', 'cross_gnn', 'norm', 'ffn', 'norm',
                'temp_gnn', 'gnn', 'norm', 'cross_gnn', 'norm', 'ffn', 'norm',
                'refine'
            ],
            temp_graph_model=dict(
                type='MultiheadAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            cross_graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=256,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            refine_layer=dict(
                type='MotionPlanningRefinementModule',
                embed_dims=256,
                fut_ts=12,
                fut_mode=6,
                ego_fut_ts=12,
                ego_fut_mode=6),
            motion_sampler=dict(type='MotionTarget'),
            motion_loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),
            planning_sampler=dict(
                type='PlanningTarget', ego_fut_ts=12, ego_fut_mode=6),
            plan_loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5),
            plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),
            plan_loss_status=dict(type='L1Loss', loss_weight=1.0),
            motion_decoder=dict(type='SparseBox3DMotionDecoder'),
            planning_decoder=dict(
                type='HierarchicalPlanningDecoder',
                ego_fut_ts=12,
                ego_fut_mode=6,
                use_rescore=True),
            num_det=50,
            num_map=10)),
    actworld=dict(width=128, jepa_weight=1.0, score_scale=0.1))
dataset_type = 'NuScenes3DDataset_roboAD_6s'
data_root = 'data/nuscenes/'
anno_root = 'data/infos/'
file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='ResizeCropFlipImage'),
    dict(type='MultiScaleDepthMapGenerator', downsample=[4, 8, 16]),
    dict(type='BBoxRotation'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='InstanceNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='VectorizeMap',
        roi_size=(30, 60),
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'gt_depth',
            'focal', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_map_labels',
            'gt_map_pts', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
            'ego_status'
        ],
        meta_keys=[
            'T_global', 'T_global_inv', 'timestamp', 'instance_id', 'token'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status',
            'gt_ego_fut_cmd'
        ],
        meta_keys=['T_global', 'T_global_inv', 'timestamp', 'token'])
]
eval_pipeline = [
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='InstanceNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='VectorizeMap', roi_size=(30, 60), simplify=True,
        normalize=False),
    dict(
        type='Collect',
        keys=[
            'vectors', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_agent_fut_trajs',
            'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks',
            'gt_ego_fut_cmd', 'fut_boxes'
        ],
        meta_keys=['token', 'timestamp'])
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
data_basic_config = dict(
    type='NuScenes3DDataset_roboAD_6s',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    map_classes=['ped_crossing', 'divider', 'boundary'],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    version='v1.0-trainval')
eval_config = dict(
    type='NuScenes3DDataset_roboAD_6s',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    map_classes=['ped_crossing', 'divider', 'boundary'],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    version='v1.0-trainval',
    ann_file='data/infos/nuscenes_infos_val_6s.pkl',
    pipeline=[
        dict(
            type='CircleObjectRangeFilter',
            class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
        dict(
            type='InstanceNameFilter',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]),
        dict(
            type='VectorizeMap',
            roi_size=(30, 60),
            simplify=True,
            normalize=False),
        dict(
            type='Collect',
            keys=[
                'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs',
                'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'fut_boxes'
            ],
            meta_keys=['token', 'timestamp'])
    ],
    test_mode=True)
data_aug_conf = dict(
    resize_lim=(0.4, 0.47),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[0, 0])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NuScenes3DDataset_roboAD_6s',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/infos/nuscenes_infos_train_6s.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='ResizeCropFlipImage'),
            dict(type='MultiScaleDepthMapGenerator', downsample=[4, 8, 16]),
            dict(type='BBoxRotation'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='CircleObjectRangeFilter',
                class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
            dict(
                type='InstanceNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='VectorizeMap',
                roi_size=(30, 60),
                simplify=False,
                normalize=False,
                sample_num=20,
                permute=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'gt_depth', 'focal', 'gt_bboxes_3d', 'gt_labels_3d',
                    'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs',
                    'gt_agent_fut_masks', 'gt_ego_fut_trajs',
                    'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status'
                ],
                meta_keys=[
                    'T_global', 'T_global_inv', 'timestamp', 'instance_id',
                    'token'
                ])
        ],
        test_mode=False,
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True),
    val=dict(
        type='NuScenes3DDataset_roboAD_6s',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/infos/nuscenes_infos_val_6s.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'ego_status', 'gt_ego_fut_cmd'
                ],
                meta_keys=['T_global', 'T_global_inv', 'timestamp', 'token'])
        ],
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        test_mode=True,
        eval_config=dict(
            type='NuScenes3DDataset_roboAD_6s',
            data_root='data/nuscenes/',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            map_classes=['ped_crossing', 'divider', 'boundary'],
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            version='v1.0-trainval',
            ann_file='data/infos/nuscenes_infos_val_6s.pkl',
            pipeline=[
                dict(
                    type='CircleObjectRangeFilter',
                    class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
                dict(
                    type='InstanceNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='VectorizeMap',
                    roi_size=(30, 60),
                    simplify=True,
                    normalize=False),
                dict(
                    type='Collect',
                    keys=[
                        'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                        'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                        'gt_ego_fut_cmd', 'fut_boxes'
                    ],
                    meta_keys=['token', 'timestamp'])
            ],
            test_mode=True)),
    test=dict(
        type='NuScenes3DDataset_roboAD_6s',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        ann_file='data/infos/nuscenes_infos_val_6s.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'ego_status', 'gt_ego_fut_cmd'
                ],
                meta_keys=['T_global', 'T_global_inv', 'timestamp', 'token'])
        ],
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        test_mode=True,
        eval_config=dict(
            type='NuScenes3DDataset_roboAD_6s',
            data_root='data/nuscenes/',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            map_classes=['ped_crossing', 'divider', 'boundary'],
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            version='v1.0-trainval',
            ann_file='data/infos/nuscenes_infos_val_6s.pkl',
            pipeline=[
                dict(
                    type='CircleObjectRangeFilter',
                    class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
                dict(
                    type='InstanceNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='VectorizeMap',
                    roi_size=(30, 60),
                    simplify=True,
                    normalize=False),
                dict(
                    type='Collect',
                    keys=[
                        'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                        'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                        'gt_ego_fut_cmd', 'fut_boxes'
                    ],
                    meta_keys=['token', 'timestamp'])
            ],
            test_mode=True)))
optimizer = dict(type='AdamW', lr=3e-05, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.1,
    min_lr_ratio=0.1)
runner = dict(type='IterBasedRunner', max_iters=6000)
eval_mode = dict(
    with_det=False,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2)
evaluation = dict(
    interval=2344,
    eval_mode=dict(
        with_det=False,
        with_tracking=False,
        with_map=False,
        with_motion=False,
        with_planning=True,
        tracking_threshold=0.2,
        motion_threshhold=0.2))
gpu_ids = range(0, 4)
custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin.models.motion.motion_planning_head_mambaAD_no_flow_6s',
        'projects.mmdet3d_plugin.models.actworld_six_second'
    ],
    allow_failed_imports=False)
find_unused_parameters = False
key = 'use_flow_decode'
split = 'test'
