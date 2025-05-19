# Getting Started with Forge Demos

This document walks you through how to set up to run demo models using tt-forge. The following topics are covered:

* [Configuring Hardware](#configuring-hardware)
* [Setting up the Docker Container](#setting-up-the-docker-container)
* [Installing Dependencies](#installing-depencencies)
* [Creating a Virtual Environment](#creating-a-virtual-environment)
* [Installing a Wheel](#installing-a-wheel)
* [Running a Demo](#running-a-demo)


> **NOTE:** If you encounter issues with anything, please request assistance on the
>[tt-forge Issues](https://github.com/tenstorrent/tt-forge/issues) page.

> **NOTE:** If you plan to do development work in the tt-forge repo, please see the
> [build instructions for tt-forge-fe](https://github.com/tenstorrent/tt-forge-fe/
> blob/main/docs/src/build.md).

## Configuring Hardware

Configure your hardware with tt-installer:

```bash
TT_SKIP_INSTALL_PODMAN=0 TT_SKIP_INSTALL_METALIUM_CONTAINER=0 /bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

## Setting up the Docker Container

The simplest way to run models is to use the Docker image. You should have 50G free for the container.

**Docker Image**: This image includes all the necessary dependencies.
    * ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-base-ird-ubuntu-22-04

To install, do the following:

1. Install Docker if you do not already have it:

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

2. Test that docker is installed:

```bash
docker --version
```

3. Add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

4. Run the container using the docker image:

```bash
sudo docker run \
  --rm \
  -it \
  --privileged \
  --device /dev/tenstorrent/0 \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --mount type=bind,source=/sys/devices/system/node,target=/sys/devices/system/node \
  ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-ird-ubuntu-22-04
```

## Creating a Virtual Environment
It is recommended that you install a virtual environment for the wheel you want to work with. Wheels from different repos may have conflicting dependencies.

Create a virtual environment (the environment name in the command is an example for the command, it's not required to use the same name listed):

```bash
python3 -m venv forge-venv
source forge-venv/bin/activate
```

## Installing a Wheel
This section walks you through downloading and installing a wheel. You can install the wheel wherever you would like if it's for running a model. If you want to do development work, you must clone the repo you want, navigate into it, and then set up the wheel.

1. Make sure you are in an active virtual environment.

> **NOTE**: If you plan to do development work, before continuing with these instructions, clone the repo you plan to use, then navigate into the repo. If you are just running models, this step is not necessary.

2. Download the wheel(s) you want to use from the [Tenstorrent Nightly Releases](https://github.com/tenstorrent/tt-forge/releases) page.

For this walkthrough, tt-forge-fe is used. You need to install two wheels for set up:

```bash
pip install https://github.com/tenstorrent/tt-forge/releases/download/nightly-0.1.0.dev20250514060212/forge-0.1.0.dev20250514060212-cp310-cp310-linux_x86_64.whl
```

```bash
pip install https://github.com/tenstorrent/tt-forge/releases/download/nightly-0.1.0.dev20250514060212/tvm-0.1.0.dev20250514060212-cp310-cp310-linux_x86_64.whl
```

> **NOTE:** The commands are examples, for the latest install link, go to the
> [Tenstorrent Nightly Releases](https://github.com/tenstorrent/tt-forge/releases)
> page. The generic download will be:
> `https://github.com/tenstorrent/tt-forge/releases/download/nightly-0.1.0.devDATE/
> NAMEOFWHEEL`
>
> If you plan to work with wheels from different repositories, make a separate
> environment for each one. Some wheels have conflicting dependencies.

## Running a Demo

To run a demo, do the following:

1. Clone the tt-forge repo (alternatively, you can download the script for the model you want to try):

```bash
git clone https://github.com/tenstorrent/tt-forge.git
```

2. Navigate to **tt-forge/demos/tt-forge-fe**.

3. Choose one of the available demos. At this time, you can try:

| Model | Model Type | Description | Demo Code |
|-------|------------|-------------|------------|
| MobileNetV2 | CNN | Lightweight convolutional neural network for efficient image classification | [`cnn/mobile_netv2_demo.py`](cnn/mobile_netv2_demo.py) |
| ResNet-50 | CNN | Deep residual network for image classification | [`cnn/resnet_50_demo.py`](cnn/resnet_50_demo.py) |
| ResNet-50 (ONNX) | CNN | Deep residual network for image classification using ONNX format | [`cnn/resnet_onnx_demo.py`](cnn/resnet_onnx_demo.py) |
| BERT | NLP | Bidirectional Encoder Representations from Transformers for natural language understanding tasks | [`nlp/bert_demo.py`](nlp/bert_demo.py) |

In this walkthrough, **resnet_50_demo.py** is used.

4. Run the selected script. As an example, this walkthrough uses the [ResNet 50 Demo](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-forge-fe/cnn/resnet_50_demo.py) script. Navigate into the **/cnn folder** and run the following command:

```bash
python3 resnet_50_demo.py
```

If all goes well, you should see an image of a cat, and terminal output where the model predicts what the image is and presents a score indicating how confident it is in its prediction.
