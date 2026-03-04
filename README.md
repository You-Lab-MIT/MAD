# MAD: Microenvironment Aware Distillation 

**[MIT YouLab](https://yougroup.mit.edu/)**

[[`Paper`]()] 

This repository provides the implementation and resources for MAD, a deep learning framework for extracting single-cell embeddings in microscopy. MAD employs a self distillation technique that captures the single cell morphology as well as microenvironmental context to give a more expressive latent.






## Installation

The training and evaluation code requires PyTorch 2.0 and [xFormers](https://github.com/facebookresearch/xformers) 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below (can also install via conda/mamba):

*[.venv](https://docs.python.org/3/library/venv.html)* - Clone the repository and then create and activate a `dinov2` venv environment using the provided environment definition:

```shell
python3.12 -m venv .mad/
source ./.mad/bin/activate
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```shell
pip install -r requirements.txt
```

## Demo 
Refer to 
```
./demo.ipynb
```
for a brief demo 
## Data preparation

### Dataset format

The root directory of the dataset should hold the following contents:

- `<ROOT>/morphology/image00001.JPEG`
- `<ROOT>/morphology/[..]`
- `<ROOT>/morphology/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/microenvironment/image00001.JPEG`
- `<ROOT>/microenvironment/[...]`
- `<ROOT>/microenvironment/image00001.JPEG`

The provided dataset implementation expects a few additional metadata files to be present under the extra directory:

- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

These metadata files can be generated (once) with the following lines of Python code:

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

Note that the root and extra directories do not have to be distinct directories.

<br />

:warning: To execute the commands provided in the next sections for training and evaluation, the `dinov2` package should be included in the Python module search path, i.e. simply prefix the command to run with `PYTHONPATH=.`.

## Training

### Fast setup: training MAD ViT-L/16 on your own dataset 

Run MAD training on a single A100 nodes (4 GPUs) in a SLURM cluster environment with submitit:

```shell
torchrun --nproc_per_node=4   -m dinov2.train.train   --config-file dinov2/configs/train/hest_vitl16.yaml   --output-dir /data/experiments/hest_neig_only_h5   train.dataset_path="MorphNeighborhoodH5:root=<path to dataset>/HEST_h5"   train.single_view_source=neighborhood train.pretrained_weights= <path to pretrained weights> dinov2/checkpoints/dinov2_vitl14_pretrain.pth
```

<!-- # python dinov2/run/train/train.py \
#     --nodes 4 \
#     --config-file dinov2/configs/train/vitl16_short.yaml \
#     --output-dir <PATH/TO/OUTPUT/DIR> \
#     train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> -->
