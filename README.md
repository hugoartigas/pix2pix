# Image-to-Image Translation with cGANs

*Authors :* Antonin Vidon, Hugo Artigas and Waël Boukhobza

This repo contains our attempt to replicate the results of pix2pix on the 'Facades' dataset. We conduct ablation experiments on data augmentation, choice of loss, batch size, ... We then constitute our own training dataset from [Country211 data](https://openaipublic.azureedge.net/clip/data/country211.tgz). # ajouter les bails faits par wael (pretraining, etc)


## Data

### Download pix2pix dataset

```
python download_data.py --dataset_name = 'facades' # download facades dataset
```

## Top-level directory layout

```./
├── history # training history saved into dictionaries and plots
├── images # data folder
│   └── facades # dataset name
│       |── test
│       |── train
│       └── val
├── plots
├── weights
├── ablation_batchsize_facades.ipynb # ablation experiments with batch size on Facades
├── ablation_da_facades.ipynb # ablation experiments with data augmentation on Facades
├── ablation_loss_facades.ipynb # ablation experiments with losses on Facades
├── download_data.py # script to download pix2pix datasets
├── model.py # our Pix2Pix model implementation
├── train_da_facades.ipynb # training notebook with data augmentation on Facades
├── train_noda_facades.ipynb # training notebook without data augmentation on Facades
├── utils.py # all functions used : create dataset, train model, plot, render outputs, ...
├── README.md
└── requirements.txt
```


## Architecture

### Generator

![generator](./figures/generator.png)

### Discriminator

![discriminator](./figures/discriminator.png)

## Model weights

[Link to model weights](https://drive.google.com/drive/folders/1x1r_KKVbPvnI8zm7YMAIR6RPV_L4ASt4?usp=sharing) (lionmail only).