# ADN: Artifact Disentanglement Network for Unsupervised Medical Image Enhancement

By [Haofu Liao](http://www.liaohaofu.com) (liaohaofu@gmail.com), Spring, 2019

## Citation

If you use this code for your research, please cite our paper.

```latex
@inproceedings{ADN2019,
  title={Artifact Disentanglement Network for Unsupervised Metal Artifact Reduction},
  author={Haofu Liao, Wei-An Lin, Jianbo Yuan, S. Kevin Zhou, Jiebo Luo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2019}
}
```

## Requirements

This repository is tested under the following system settings:

- Ubuntu 16.04
- Python 3.7 (Anaconda/Miniconda reconmmended)
- Pytorch 1.0.0 or above
- CUDA 9.0 or above
- Matlab R2018b

## TODO

- [x] Rename all enc3 to adn
- [x] Remove unnecessary functions/classes/options
- [x] Add DeepLesion and Spineweb dataset
- [x] Fix the tqdm display issue (more display info and first display bug)
- [x] Change default options for ADN
- [x] Add a demo that generate samples
- [x] Add license
- [x] Add some comments to the source code
- [x] Add a list of dependencies
- [x] Add a docker file
- [ ] Retrain the model with 3 sides
- [ ] Prepare google drive for trained models
- [ ] Add sample images
- [ ] Test this repo from scratch
- [ ] Finish Readme file
- [ ] Push repo
- [ ] Review the repo and publish
- [ ] Add repo into paper

## Getting Started

### Install

#### Local

- Clone this repository from Github

```cmd
git clone https://github.com/liaohaofu/adn.git
```

- Install [Pytorch](https://pytorch.org/get-started/locally/) and [Anaconda](https://www.anaconda.com/distribution/#download-section)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Anaconda/Miniconda installation is optional. If not installed, you may install some dependent python packages manually.
- Install Python dependencies.

```cmd
pip install -r requirements.txt
```

#### Docker

- Install [docker-ce](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
- [Optional] If you want a customized version of ADN docker, you may modify the docker file at `docker/Dockerfile` and then build a docker image.

```cmd
cd docker/
docker build -t adn .
```

### Datasets

Two publicly available datasets (DeepLesion and Spineweb) are supported.

#### DeepLesion

- Download the [DeepLesion dataset](https://nihcc.app.box.com/v/DeepLesion). We use the first 9 *.zip* files (Images_png_01.zip to Images_png_09.zip) in our experiments. You may use the `batch_download_zips.py` provided by DeepLesion to batch download the *.zip* files at once.
- Extract the downloaded *.zip* files. The extracted images should be located under `path_to_DeepLesion/Images_png`.
- Create a softlink to DeepLesion.

```cmd
ln -s path_to_DeepLesion/Images_png data/deep_lesion/raw
```

- Prepare DeepLesion dataset for ADN (**MATLAB required**). The configuration file for preparing DeepLesion dataset is can be found at `config/dataset.yaml`.

```matlab
>> prepare_deep_lesion
```

#### Spineweb

- Download the [Spineweb dataset](https://imperialcollegelondon.app.box.com/s/erhcm28aablpy1725lt93xh6pk31ply1).
- Extract the *spine-\*.zip* files. The extracted files should be located under `path_to_Spineweb/spine-*`.
- Create a softlink to Spineweb.

```cmd
mkdir data/spineweb
ln -s path_to_Spineweb/ data/spineweb/raw
```

- Prepare Spineweb dataset for ADN. The configuration file for preparing Spineweb dataset is can be found at `config/dataset.yaml`.

```cmd
python prepare_spineweb.py
```

### Demo

- We provide a demo code to demonstrate the effectiveness of ADN. The input samples images are located at `samples/` and the outputs of the demo can be found at `results/`. To run the demo,

```cmd
python demo.py deep_lesion
python demo.py spineweb
```

- [Optional] By default, the demo code will download pretrained models from goole drive automatically. If the downloading fails, you may download them from google drive manually.
  - Download pretrained models for [DeepLesion](https://drive.google.com/open?id=1NqZtEDGMNemy5mWyzTU-6vIAVIk_Ht-N) and [Spineweb]()
  - Move the downloaded models to `runs/`

  ```cmd
  mv path_to_DeepLesion_model runs/deep_lesion/deep_lesion_49.pt
  mv path_to_Spineweb_model runs/spineweb/spineweb_29.pt
  ```

### Train and Test

- Configure the training and testing. We use a two-stage configuration for ADN, one for the default settings and the other for the run settings.
  - The default settings of ADN can be found at `config/adn.yaml` which is not subject to be changed. When users do not provide the values for a specific setting, the default setting in this file will be used.
  - The run settings can be found at `runs/adn.yaml`. This is where the users provide specific settings for ADN's training and testing. Any provided settings in this file will override the default settings during the experiments. **By default, the settings for training and testing ADN with DeepLesion and Spineweb datasets are provided in** `runs/adn.yaml`.

- Train ADN with DeepLesion or Spineweb datasets. The training results (model checkpoints, configs, losses, training visualizations, etc.) can be found under `runs/run_name/` where `run_name` can be either `deep_lesion` or `spineweb`.

```cmd
python train.py deep_lesion
python train.py spineweb
```

- Test ADN with DeepLesion or Spineweb datasets. The testing results (evaluation metrics and testing visualizations, etc.) can be found under `runs/run_name/` where `run_name` can be either `deep_lesion` or `spineweb`.

```cmd
python test.py deep_lesion
python test.py spineweb
```

## Acknowledgements

The authors would like to thank Dr. Yanbo Zhang (yanbozhang007@gmail.com) and Dr. Hengyong Yu (hengyong_yu@uml.edu) for providing the artifact synthesis code used in this repository.
