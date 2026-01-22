# Copyright (c) OpenMMLab. All rights reserved.
import logging
from argparse import ArgumentParser

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--draw-heatmap', action='store_true', help='Visualize the predicted heatmap')
    parser.add_argument('--show-kpt-idx', action='store_true', default=False, help='Show the index of keypoints')
    parser.add_argument('--skeleton-style', default='mmpose', type=str,
                        choices=['mmpose', 'openpose'], help='Skeleton style')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness')
    parser.add_argument('--alpha', type=float, default=0.8, help='Transparency of bboxes')
    parser.add_argument('--show', action='store_true', default=False, help='Whether to show image')
    return parser.parse_args()


def main():
    args = parse_args()

    # Enable heatmap output if requested
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True))) if args.draw_heatmap else None

    # Initialize model
    model = init_model(args.config, args.checkpoint, device=args.device, cfg_options=cfg_options)

    # Overwrite dataset_meta for 8 keypoints (not 9!)
    keypoint_names = [
        '0','1', '2', '3','4','5', '6','7','8','9','10','11','12','13','14', '15','16','17','18','19'
    ]

    model.dataset_meta = {
        'dataset_name': 'FishPose20Kpt',
        'num_keypoints': 19,
        'keypoint_names': keypoint_names,
        'keypoint_id2name': {i: name for i, name in enumerate(keypoint_names)},
        'keypoint_name2id': {name: i for i, name in enumerate(keypoint_names)},
        'skeleton': [],
        'keypoint_colors': [[255, 0, 0]] * 19,
        'skeleton_links_color': [[0, 255, 0]] * 7,
        'flip_pairs': [],
        'flip_indices': list(range(19)),
        'upper_body_ids': [0, 2, 3, 4, 5],
        'lower_body_ids': [1, 6, 7],
        'dataset_keypoint_weights': [1.0] * 19,
        'sigmas': [0.05] * 19,
        'CLASSES': ['fish']
    }


    print('\n\n✅ DEBUG: Patched model.dataset_meta keys =', model.dataset_meta.keys(), '\n')

    # Init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style=args.skeleton_style)

    # Run inference
    model.cfg.test_dataloader.dataset.pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='PackPoseInputs')
]
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)
    # Obtener coordenadas predichas
    keypoints = results.pred_instances.keypoints
    scores = results.pred_instances.keypoint_scores


    # Imprimir coordenadas y puntajes
    print("\n📍 Coordenadas predichas:")
    for idx, (xy, score) in enumerate(zip(keypoints[0], scores[0])):
        name = model.dataset_meta['keypoint_names'][idx]
        print(f" - {name}: x={xy[0]:.1f}, y={xy[1]:.1f}, score={score:.3f}")
        import json
        import os

        # Construir el diccionario con los datos
        predicted_data = {
            'image': os.path.basename(args.img),
            'keypoints': [
                {
                    'name': model.dataset_meta['keypoint_names'][idx],
                    'x': float(xy[0]),
                    'y': float(xy[1]),
                    'score': float(score)
                }
                for idx, (xy, score) in enumerate(zip(keypoints[0], scores[0]))
            ]
        }

        # Determinar nombre de archivo de salida
        import os
        output_json = os.path.splitext(args.out_file)[0] + '_keypoints.json'


        # Guardar como JSON
        with open(output_json, 'w') as f:
            json.dump(predicted_data, f, indent=4)

        print(f"\n📝 Coordenadas guardadas en JSON: {output_json}")



    # Show or save results
    img = imread(args.img, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        draw_pred=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file
    )

    if args.out_file is not None:
        print_log(f'✅ Output image saved at {args.out_file}', logger='current', level=logging.INFO)


if __name__ == '__main__':
    main()
