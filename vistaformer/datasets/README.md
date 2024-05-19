## Datasets

For both datasets, we recommend downloading data from the provided sources and generating datasets as needed.

### MTLCC dataset (Germany)

The data for Germany can be downloaded from: [github.com/TUM-LMF/MTLCC](https://github.com/TUM-LMF/MTLCC). This dataset was transformed from `.tfrecords` into `.pickle` files for this experiment.

After downloading the data, you can create these files, by running the `vistaformer/datasets/mtlcc/create_samples.py` script using a command with the following format:

```bash
python -m vistaformer.datasets.mtlcc.create_samples --rootdir <very-real-path> --outdir <very-real-path>
```

### PASTIS dataset

The PASTIS dataset can be downloaded from [github.com/VSainteuf/pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark).

After downloading the data, you can create these files by running the `vistaformer/datasets/pastis/create_samples.py` script using a command like the following:

```bash
python -m vistaformer.datasets.pastis.create_samples --rootdir <very-real-path> --outdir <very-real-path>
```
