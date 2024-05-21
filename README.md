# VistaFormer

This repository is the official code repository for the paper "VistaFormer: Simple Vision Transformers for Satellite Image Time Series Segmentation"

## Installation

To run code from this repository you will first need to install the project dependencies which can be done either using the `requirements.txt` file or the [Poetry](https://python-poetry.org/) configuration. To install this project using the `requirements.txt` file, you can execute the following commands:

```bash
pip install -r requirements.txt
```

## Train Models

To use this repository to run inference on pre-trained weights or train a model on one of the datasets, please see the documentation in the `datasets` directory for instructions on how to download and create these datasets.

Once you have created a dataset you can train a model using the following commands:

```bash
export MODEL_CONFIG="very-real-model-config-path"
python -m vistaformer.train_and_evaluate.train
```

To evaluate the performance of a pre-trained model on a given dataset, please refer to the `notebooks/inference.ipynb` file to compute complete metrics.

*Please note that pre-trained model weights and training logs for each trial that was reported in the results section of the accompanying paper will be released once an unanonymized name can accompany this repository.*


## Results on [PASTIS](https://github.com/VSainteuf/pastis-benchmark) (Optical only) Semantic Segmentation Benchmark

| Model Name         | mIoU | oA | #Params (M) | GFLOPs |
| ------------------ |----- |----- | ------------ | ------|
| U-TAE              | 63.1 | 63.1 | 1.1          | 23.06 |
| TSViT †            | 65.4 | 83.4 | 1.6          | 91.88 |
| **VistaFormer**    | 65.5 | 84.0 | 1.3          | 7.58  |

(†) TSViT operates on PASTIS24, where each sample is split into 24x24px sub-patches.

## Results on [MTLCC](https://github.com/TUM-LMF/MTLCC) Semantic Segmentation Benchmark

| Model Name         | mIoU | oA   | #Params (M) | GFLOPs |
| ------------------ |----- |----- | ------------ | ------|
| U-TAE              | 77.1 | 93.1 | 1.1          | 23.06 |
| TSViT              | 84.8 | 95.0 | 1.6          | 91.88 |
| **VistaFormer**    | 87.8 | 95.9 | 1.3          | 7.58  |

Note that the GFLOPS and parameter measurements are based on inputs with input dimensions (B, C, T, H, W) = (4, 10, 60, 32, 32).
