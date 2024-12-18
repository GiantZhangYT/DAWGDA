# Diffusion-Augmented Wasserstein Graph Domain Adaption for Cross-network Node Classification
This repository contains the author's implementation in Pytorch for the paper "Diffusion-Augmented Wasserstein Graph Domain Adaption for Cross-network Node Classification".

## Environment Requirement

* Python: 3.7.12

* PyTorch: 1.11.0 + cuda 11.3

> conda create -n DAWGDA python=3.7

> conda activate DAWGDA

> pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

> pip install -r requirements.txt

> conda install pyg=2.0.4=py37_torch_1.11.0_cu113 -c pyg

## Datasets:

The data folder includes different domain data. 

The preprocessed data can be found in our repository.

* `/data/acmv9.mat`

* `/data/dblpv7.mat`

* `/data/citationv1.mat`

* `/data/Blog1.mat`

* `/data/Blog2.mat`

##  Training:

> python DAMGDA-Citation.py --data_src dblpv7 --data_trg acmv9 --lambda_inter 0.1 --n_epoch 100

> python DAMGDA-Blog.py --data_src Blog2 --data_trg Blog1 --lambda_inter 0.15 --n_epoch 500

> cd arixiv_code

> python DAWGDA-arixiv.py --lambda_sso 1e-3 --lambda_inter 1e-5

