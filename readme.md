_Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Please make sure that your usage of this code is in compliance with the code license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/MIC-DKFZ/basic_unet_example/blob/master/LICENSE)

---

# MOOD 2020 - Repository

This repo has the supplementary code for the _Medical Out-of-Distribution Analysis Challenge_ at MICCAI 2020.

Also checkout our [Website](http://medicalood.dkfz.de/web/) and [Submission Platform](https://www.synapse.org/mood).

### Requirements

Please install and use docker for submission: <https://www.docker.com/get-started>

For GPU support you may need to install the NVIDIA Container Toolkit: <https://github.com/NVIDIA/nvidia-docker>

Install python requirements:

```
pip install -r requirements.txt
```

We suggest the following folder structure (to work with our examples):

```
data/
--- brain/
------ brain_train/
------ toy/
------ toy_label/
--- colon/
------ colon_train/
------ toy/
------ toy_label/
```

### Run Simple Example

Have a lot at the simple_example in how to build a simple docker, load and write files, and run a simple evaluation.
After installing the requirements you can also try the simple_example:

```
python docker_example/run_example.py -i /data/brain/ --no_gpu False
```

With `-i` you can pass an input folder (which has to contain a _toy_ and _toy_label_ directory) and with `--no_gpu` you can turn on/off GPU support for the docker (you may need to install the NVIDIA Container Toolkit for docker GPU support).

### Test Your Docker

After you built your docker you can test you docker locally using the toy cases. After submitting your docker, we will also report the toy-test scores on the toy examples back to you, so you can check if your submission was successful and the scores match:

```
python scripts/test_docker.py -d mood_docker -i /data/ -t sample
```

With `-i` you can pass the name of your docker image, with `-i` pass the path to your base*data dir (see \_Requirements*), with `-t` you can define the Challenge Task (either _sample_ or _pixel_), and with `--no_gpu` you can turn on/off GPU support for the docker (you may need to install the NVIDIA Container Toolkit for docker GPU support).

### Scripts

In the scripts folder you can find:

- `test_docker.py` : The script to test your docker.
- `evalresults.py` : The script with our evaluation code.

### Example Algorithms

For _'ready to run'_ simple example algorithms checkout the [example_algos](https://github.com/MIC-DKFZ/mood/tree/master/example_algos) folder.
