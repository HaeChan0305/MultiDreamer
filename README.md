# MultiDreamer: Generating 3D mesh from a Single-view Multi-objects Image
Code release of our paper [MultiDreamer](https://google.com). Check out our [paper](https://google.com), and [website](https://multidreamer-demo.web.app)!

## Abstract
![](images/pipeline.png)

Single-view 3D reconstruction is an actively researched field in computer vision. Many existing models utilize pretrained 2D diffusion models to generate novel views from a single-view image. However, these models face challenges with multi-object input images, struggling to extract accurate 3D positional information. In this work, we introduce MultiDreamer, a method for reconstructing a 3D scene comprising multiple objects from a single-view multi-object image. We enhance 3D mesh accuracy by independently generating meshes for individual objects, utilizing inpainting for obscured regions, and optimizing mesh alignment using the depth map. Our pipeline produces robust 3D scenes for arbitrary objects without training process. Experimental analysis results demonstrate that our pipeline consistently captures the shapes and relative positions of multiple objects compared to the baseline model.


## Development Environment
We use the following development environment for this project:
- Nvidia RTX 3090 GPU
- Intel Xeon Processor
- Ubuntu 22.04
- CUDA Version 11.7
- cudatoolkit 10.2.89, 11.7.1
- torch 1.10.1, 1.13.1
- torchvision 0.11.2
- Detectron2 0.6

## Installation
This code is developed using anaconda3 with Python 3.8 and 3.9 ([download here](https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh)), therefore we recommend a similar setup.

You can simply run the following code in the command line to create the development environment:
```
$ source setup.sh
```


## Running the Demo
![](images/example.png)

We provide two sample input images in the `data/assets` folder. If you want to test with your own example, the images should consist of exactly two objects. To run the demo, you first need to download the pre-trained model checkpoint files for two models, namely SemanticSAM and SyncDreamer, from this [Google Drive folder](https://drive.google.com/drive/folders/1k5-QoS6CmK71r9jkc62LKpnxE6HhIGUU). The path of the downloaded files MUST follow the structure below : 
```plaintext
MultiDreamer/models/
│
├─ SemanticSAM/
│  └─ models/
│     └─ swinl_only_sam_many2many.pth
│
└─ SyncDreamer/
   └─ ckpt/
      ├─ syncdreamer-pretrain.ckpt
      └─ ViT-L-14.pt
```

Before you run `demo.sh`, you shoud ckeck and modify the path of input image and output directory in `demo.sh`. If you need, make the `data/output/` directory :
```
INPUT_IMAGE="/MultiDreamer/data/assets/giraffe_and_flower/0_input_giraffe_and_flower.png"
OUTPUT_DIR="/MultiDreamer/data/output/giraffe_and_flower/"
```
and then, you can run :
```
$ bash demo.sh
```

## Preparing Data
### Downloading Processed Data (Recommended)
We provide 32 examples in this [Google Drive folder](https://drive.google.com/drive/folders/1k5-QoS6CmK71r9jkc62LKpnxE6HhIGUU). In the link, each example folder contains `input png file` and `ground truth glb file`. We recommand setting the downloaded folder as `data/eval/`.

## Evaluation
This is the qualitative result presented in our paper.
![Qualitative](images/qualitative_result.png)

In the evaluation part, we compared results of MultiDreamer(Ours) and SyncDreamer(Baseline). Then, we measured Chamfer Distance, Volume IoU, and F-Score for quantitative evaluation. The code to obtain results for both models and compute the metrics is in `eval/eval.sh`. You can run : 
```
$ cd eval
$ bash eval.sh
```

Finally, you can obtain the result like the table below. The values of each metric may differ from the table, as they are computed from randomly sampled vertices in the inferred mesh and the ground truth mesh.

![Quantitative](images/quantitative_result.png)


### Libraries
- https://pytorch.org/

- https://github.com/facebookresearch/detectron2

- https://github.com/facebookresearch/pytorch3d

- http://www.open3d.org/

### Projects
- https://github.com/liuyuan-pal/SyncDreamer

- https://github.com/UX-Decoder/Semantic-SAM

- https://huggingface.co/runwayml/stable-diffusion-inpainting

- https://github.com/isl-org/ZoeDepth
