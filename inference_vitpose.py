import os
import glob
import subprocess

def main():
    # Define paths
    input_folder = '/data/Datasets/Fish/FIB/FIB/inference/'
    output_folder = 'vis_results/vitpose_600epoch/fish_infer_all/'
    config_path = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py'
    checkpoint_path = 'work_dirs/td-hm_ViTPose-fish9_8xb32-100etrain/epoch_600.pth'

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    image_files.sort()

    # Loop through images and run inference
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        out_file = os.path.join(output_folder, img_name)

        print(f'Running inference on {img_name}...')

        cmd = [
            'python', 'demo/image_demo.py',
            img_path,
            config_path,
            checkpoint_path,
            '--out-file', out_file,
            '--draw-heatmap',
            '--show-kpt-idx'
        ]

        subprocess.run(cmd, check=True)

    print('All inferences completed.')

if __name__ == '__main__':
    main()
