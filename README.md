# MultiDreamer: Generating 3D mesh from a Single-view Multi-objects Image
Code release of our paper [MultiDreamer](https://google.com). Check out our [paper](https://google.com), and [website](https://multidreamer-demo.web.app)!

![](images/asdfa.jpg)

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
![](network/assets/demo_out.jpg)

We provide four sample input images in `data/assets` folder. The images should consist of exactly two objects. To run the demo, you first need to download data and config from this [Google Drive folder](https://drive.google.com/drive/folders/1uHwu3YmJnQm5I3HqxDYCuv_NbvvDndRX). `models` folder contains the several pre-trained models, while `data` folder contains images and dataset.

Before you run 'demo.sh', you need to ckeck and set the pathes of input image and output directory in 'demo.sh':
```
INPUT_IMAGE="/MultiDreamer/data/input/1.png"
OUTPUT_DIR="/MultiDreamer/data/output/1/"
```
and then, you can run :
```
$ bash demo.sh
```

## Preparing Data
### Downloading Processed Data (Recommended)
We provide preprocessed images and labels in this [Google Drive folder](https://drive.google.com/drive/folders/1JbPidWsfcLyUswYQsulZN8HDFBTdoQog).  Download and extract all folders to a desired location before running the training and evaluation code.

### Rendering Data
Alternatively, you can render data yourself. Our data preparation code lives in the `renderer` folder.

Our project depends on [ShapeNet](https://shapenet.org/) (Chang et al., '15), [ScanNet](https://github.com/ScanNet/ScanNet) (Dai et al. '16), and [Scan2CAD](https://github.com/skanti/Scan2CAD) (Avetisyan et al. '18) datasets. For ScanNet, we use ScanNet25k images which are provided as a zip file via the ScanNet download script.

Once you get the data, check `renderer/env.sh` file for the locations of different datasets. The meanings of environment variables are described as inline comments in `env.sh`. 

After editing `renderer/env.sh`, run the data generation script:
```
$ cd renderer
$ sh run.sh
```

Please check `run.sh` to see how individual scripts are running for data preprocessing and feel free to customize the data pipeline!

## Training and Evaluating Models
Our training code lives in the `network` directory. Navigate to the `network/env.sh` and edit the environment variables. Make sure data directories are consistent with the ones locations downloaded and extracted folders. If you manually prepared data, make sure locations in `/network/env.sh` are consistent with the variables set in `renderer/env.sh`.

After you are done with `network/env.sh`, run the `run.sh` script to train a new model or evaluate an existing model based on the environment variables you set in `env.sh`:
```
$ cd network
$ sh run.sh
```

### Replicating Experiments from the Main Paper
Based on the configurations in `network/env.sh`, you can run different ablations from the paper. The default config will run the (final) experiment. You can do the following edits <b>cumulatively</b> for different experiments:

1. For P+E+W+R, set `RETRIEVAL_MODE=resnet_resnet+image`
2. For P+E+W, set `RETRIEVAL_MODE=nearest`
3. For P+E, set `NOC_WEIGHTS=0`
4. For P, set `E2E=0`

## Resources
To get the datasets and gain further insight regarding our implementation, we refer to the following datasets and open-source codebases:

### Datasets and Metadata
- https://shapenet.org/

- http://www.scan-net.org/

- https://github.com/skanti/Scan2CAD

### Libraries
- https://pytorch.org/

- https://github.com/facebookresearch/detectron2

- https://github.com/facebookresearch/pytorch3d

- http://www.open3d.org/

### Projects
- https://github.com/facebookresearch/meshrcnn

- https://github.com/xheon/JointEmbedding
