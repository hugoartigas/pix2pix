# Image-to-Image Translation with Conditional Adversarial Networks

## Download Pix2Pix data

```
python download_data.py --dataset_name = 'facades' #
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
├── download_data.py # script to download any data set from http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
├── model.py # our Pix2Pix model implementation
├── train_da_facades.ipynb # training notebook with data augmentation on Facades
├── train_noda_facades.ipynb # training notebook without data augmentation on Facades
├── utils.py # all functions used : create dataset, train model, plot, render outputs, ...
├── README.md
└── requirements.txt
```

## Generator architecture

![generator](./figures/generator.png)

## Discriminator architecture

![discriminator](./figures/discriminator.png)