# detr-deepstream-sdk

## Description

This repository is a demo showing how to export a custom object-detection model to use with NVIDIA DeepStream SDK for
inference using [Meta DETR](https://huggingface.co/facebook/detr-resnet-101) as an example.

## Usage

### Pre-requisites

1. Set up an environment that supports inference using NVIDIA DeepStream SDK following the
[official documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html).
> ***NOTE:*** The contents of this repository were developed and tested with NVIDIA GeForce RTX 4090 dGPU on a
standalone installation of Ubuntu 20.04 LTS. NVIDIA doesn't guarantee proper functioning of DeepStream on either newer
versions of Ubuntu, or WSL installation.

2. Install the repository sources and dependencies:
```bash
$ make .
```
