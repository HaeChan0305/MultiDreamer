source ~/anaconda3/etc/profile.d/conda.sh

conda env create -n sam -f ./env/sam.yaml
conda env create -n inpainting -f ./env/inpainting.yaml
conda env create -n syncdreamer -f ./env/syncdreamer.yaml
conda env create -n zoedepth -f ./env/zoedepth.yaml
conda env create -n eval -f ./env/eval.yaml
