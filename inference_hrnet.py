import os
import glob
import subprocess

def main():
    # Define paths
    input_folder = '/data/Datasets/Fish/FIB/FIB/inference/'
    output_folder = 'vis_results/hrnet/'
    config_path = 'configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_fish-256x256.py'
    checkpoint_path = 'checkpoints/td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_fish-256x256-76c3999f_20230519.pth'

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

        if os.path.exists(out_file):
            print(f'Skipping {img_name} (already processed).')
            continue

        print(f'Running HRNet inference on {img_name}...')

        cmd = [
            'python', 'demo/image_demo_hrnet.py',
            img_path,
            config_path,
            checkpoint_path,
            '--out-file', out_file,
            '--draw-heatmap',
            '--show-kpt-idx',
            '--device', 'cuda'
        ]

        subprocess.run(cmd, check=True)

    print('All HRNet inferences completed.')

if __name__ == '__main__':
    main()
