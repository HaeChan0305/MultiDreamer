source ~/anaconda3/etc/profile.d/conda.sh

# Main pipline
INPUT_IMAGE="/root/MultiDreamer/data/assets/giraffe_and_flower/0_input_giraffe_and_flower.png"
OUTPUT_DIR="/root/MultiDreamer/data/output/giraffe_and_flower/"

# ---------- [1] Object Detachment ----------
conda activate sam
cd /root/MultiDreamer/models/SemanticSAM
BBOX=$(python inference_auto_generation.py --input $INPUT_IMAGE --output_dir $OUTPUT_DIR --level 2) 
conda deactivate

conda activate inpainting
cd /root/MultiDreamer/models/StableDiffusionInpaint
python inpainting.py --input $INPUT_IMAGE --output_dir $OUTPUT_DIR --bbox "${BBOX}"
conda deactivate


# ---------- [2] Mesh Reconstruction ----------
conda activate syncdreamer
cd /root/MultiDreamer/models/SyncDreamer
python generate.py --input $INPUT_IMAGE --output_dir $OUTPUT_DIR --index 0 --mesh
python generate.py --input $INPUT_IMAGE --output_dir $OUTPUT_DIR --index 1 --mesh

# if you want to see the result of baseline together
python generate.py --input $INPUT_IMAGE --output_dir $OUTPUT_DIR --baseline --mesh
conda deactivate


# # ---------- [3] Mesh Alignment ----------
conda activate zoedepth
cd /root/MultiDreamer/models/ZoeDepth
python demo.py --input $INPUT_IMAGE --output_dir $OUTPUT_DIR

cd /root/MultiDreamer/models/Alignment
python main.py --output_dir $OUTPUT_DIR

conda deactivate
