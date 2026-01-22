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

    # Log dataset_meta keys for debugging
    print('\n\nDEBUG: Loaded model.dataset_meta keys =', model.dataset_meta.keys(), '\n\n')

    # Init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style=args.skeleton_style)

    # Run inference
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

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
        print_log(f'The output image has been saved at {args.out_file}', logger='current', level=logging.INFO)


if __name__ == '__main__':
    main()
