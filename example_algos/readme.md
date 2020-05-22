# Example Algorithms

This folder contains a few simple example OoD algorithms for the _Medical Out-of-Distribution Analysis Challenge_.

### Quickstart

Install python requirements:

```
pip install -r requirements_algos.txt
```

And use the data/preprocess.py script to preprocess the data (you may want to use more sophisticated preprocessing for your own submission):

```
python data/preprocess.py -i input_folder -o output_folder
```

### Run the example algorithms

Basically all the algorithms in the algorithm folder are ready-to-run, or can be used as a starting point for your own algorithms. All algorithms take the same basic command_line arguments:

- -r [--run], Which part of the algorithm pipeline you want to run ("train", "predict", "test", "all").
- -d [--data-dir], The directory containing the preprocessed training data.
- -o [--log-dir], The directory to which you want the logs to be stored in.
- -t [--test-dir], The directory containing the test data (requires a folder with the same name plus the suffix '\_label').
- -m [--mode], pixel-level or sample-level algorithm ('pixel', 'sample').
- --logger, if you want to use a logger, can either use a visdom server (running on port 8080) or tensorboard ("visdom", "tensorboard").

For more arguments checkout the python files.

The example Algorithms includes:

#### 2D Autoencoder (ae_2d.py)

A simple 2d autoencoder which uses the reconstruction error as OoD score:

```
python algorithms/ae_2d.py -r all -o output_dir -t /data/mood/brain/toy  --mode pixel --logger visdom -d /data/mood/brain/train_preprocessed
```

#### 3D Autoencoder (ae_3d.py)

A simple 3d autoencoder which uses the reconstruction error as OoD score:

```
python algorithms/ae_3d.py -r all -o output_dir -t /data/mood/brain/toy  --mode pixel --logger visdom -d /data/mood/brain/train_preprocessed
```

#### ceVAE (ce_vae.py)

A simple context-encoding Variational Autoencoder. Can be used as only a VAE or CE as well.

```
python algorithms/ce_vae.py -r all -o output_dir -t /data/mood/brain/toy  --mode pixel --logger visdom -d /data/mood/brain/train_preprocessed --ce-factor 0.5 --score-mode combi
```

With additional arguments:

- --ce-factor, determines the 'mixing' between VAE and CE (0.0=VAE only, 1.0=CE only).
- --score-mode, How to determine the OoD score ("rec", "grad", "combi").

#### fAnoGAN (f_ano_gan.py)

A fAnoGAN algorithm built on top of improved Wasserstein GANs. Can use be used as AnoGAN only (without the encoding).

```
python algorithms/f_ano_gan.py -r all -o output_dir -t /data/mood/brain/toy  --mode pixel --logger visdom -d /data/mood/brain/train_preprocessed --use-encoder
```

With argument:

- --use-encoder/--no-encoder, to determine whether to train an additional encoder (fAnoGAN) to reconstruct the image or use no encoder and use backpropagation for reconstruction.

### More Code

While the _example_algos_ code is ready to run there are a lot of excellent code repositories which can also be checked out. Some of the Include:

- <https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI> , if you use tensorflow.
- <https://github.com/AntixK/PyTorch-VAE>, for more VAE variants.
- <https://github.com/hoya012/awesome-anomaly-detection>, for more pointers/ great algorimths.
- <https://github.com/shubhomoydas/ad_examples> , <https://github.com/yzhao062/pyod> , collections of basic, easy-to-use algorithms.
- <https://github.com/tSchlegl/f-AnoGAN>, the original f-AnoGAN implementation.
- <https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection>
- <https://github.com/izikgo/AnomalyDetectionTransformations>
- <https://github.com/ashafaei/OD-test>

And as always, _Papers With Code_:

- [Anomaly Detection](https://paperswithcode.com/task/anomaly-detection/)
- [Out-of-Distribution Detection](https://paperswithcode.com/task/out-of-distribution-detection/)
- [Outlier Detection](https://paperswithcode.com/task/outlier-detection/)
- [Density Estimation](https://paperswithcode.com/task/density-estimation/)

Good Luck and Have Fun! =)
