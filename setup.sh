#!/bin/bash

# Function to print an error message in red
print_error() {
  echo -e "\e[31m[ERROR] $1\e[0m" 
}

# Execute commands and check for success
python -m src.scripts.download_bop || print_error "download_bop failed"
python -m src.scripts.render_template_with_pyrender || print_error "render_template_with_pyrender failed"
python -m src.scripts.download_sam || print_error "download_sam failed"
python -m src.scripts.download_fastsam || print_error "download_fastsam failed"
#python -m src.scripts.download_train_pbr || print_error "download_train_pbr failed"

#python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

#python -m src.scripts.visualize_detectron2 dataset_name=ycbv input_file=/code/datasets/bop23_challenge/results/cnos_exps/FastSAM_template_pbr0_aggavg_5_ycbv.json output_dir=$/code/datasets/bop23_challenge/results/cnos_exps/


#python cnos_ros_wrapper.py dataset_name=ycbv model=cnos_fast model.onboarding_config.rendering_type=pyrender