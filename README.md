# VistaFormer

The official code repository for the paper "VistaFormer: Vision Transformers for Satellite Image Time Series Segmentation"


## Results on [PASTIS Benchmark](https://github.com/VSainteuf/pastis-benchmark) Dataset
### PASTIS - Semantic Segmentation - Optical only (PASTIS)


| Model Name         | mIoU | #Params (M) | GFLOPs |
| ------------------ |----- |------------ | ------|

| U-TAE | 63.1 | 1.1 | 23.06 |
| TSViT † | 65.4 | 1.6 | 91.88 |
| **VistaFormer** | 1.3 | 8.08 | 7.58 |

(†) TSViT operates on PASTIS24, where each sample is split into 24x24px sub-patches.
