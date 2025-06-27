# Getting Started with Forge Demos

This document walks you through how to set up to run demo models using tt-forge. The following topics are covered:

* [Configuring Hardware](#configuring-hardware)
* [Setting up the Docker Container](#setting-up-the-docker-container)
* [Creating a Virtual Environment](#creating-a-virtual-environment)
* [Installing a Wheel](#installing-a-wheel)
* [Running a Demo](#running-a-demo)
* [Running Performance Benchmark Tests](#running-performance-benchmark-tests)

> **NOTE:** If you encounter issues, please request assistance on the
>[tt-forge Issues](https://github.com/tenstorrent/tt-forge/issues) page.

> **NOTE:** If you plan to do development work, please see the
> build instructions for the repo you want to work with.

## Configuring Hardware

Configure your hardware with tt-installer:

```bash
TT_SKIP_INSTALL_PODMAN=0 TT_SKIP_INSTALL_METALIUM_CONTAINER=0 /bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

>**NOTE:** This walkthrough assumes that you use the [Quick Installation](https://docs.tenstorrent.com/getting-started/README.html#quick-installation) instructions. If you want to use the tools installed by this script, you must activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

## Setting up the Docker Container

The simplest way to run models is to use the Docker image. You should have 50G free for the container.

**Docker Image**: ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-base-ird-ubuntu-22-04

To install, do the following:

1. Install Docker if you do not already have it:

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

2. Test that Docker is installed:

```bash
docker --version
```

3. Add your user to the Docker group:

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

Create a virtual environment:

```bash
python3 -m venv forge-venv
source forge-venv/bin/activate
```

## Installing a Wheel
This section walks you through downloading and installing a wheel. You can install the wheel wherever you would like if it's for running a model. If you want to do development work, you must clone the repo you want, navigate into it, and then set up the wheel.

1. Make sure you activate the virtual environment you created in the last step.

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

4. Run the selected script. As an example, this walkthrough uses the [ResNet 50 Demo](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-forge-fe/cnn/resnet_50_demo.py) script. Navigate to the root of the tt-forge repository and run the following commands:

```bash
export PYTHONPATH=.
python3 demos/tt-forge-fe/cnn/resnet_50_demo.py
```

If all goes well, you should see an image of a cat, and terminal output where the model predicts what the image is and presents a score indicating how confident it is in its prediction.

## Running Performance Benchmark Tests

To run performance benchmarks for all models, you need to install additional libraries that are not included in the Docker container or the wheel package.

### Prerequisites

1. **Install Python Requirements**

   Install the required Python packages from the [requirements](../../benchmark/requirements.txt) file:

   ```bash
   pip install -r benchmark/requirements.txt
   ```

2. **Install System Dependencies**

   Install the required system libraries for OpenGL rendering and core application support:

   ```bash
   sudo apt update
   sudo apt install libgl1-mesa-glx libgl1-mesa-dev mesa-utils
   ```

3. **Set up Hugging Face Authentication**

   To run models on real datasets, you need to register and authenticate with Hugging Face:

   a. Login or register at [Hugging Face](https://huggingface.co/)

   b. Set up an access token following the [User Access Tokens guide](https://huggingface.co/docs/hub/en/security-tokens#user-access-tokens)

   c. Configure your environment with the token:

   ```bash
   export HUGGINGFACE_TOKEN=[YOUR_TOKEN]
   huggingface-cli login --token $HUGGINGFACE_TOKEN
   ```

   d. Access the Imagenet dataset [here](https://huggingface.co/datasets/mlx-vision/imagenet-1k)

### Running Benchmarks

Once you have completed the prerequisites, you can run the performance benchmarks:

1. Navigate to the benchmark directory:

   ```bash
   cd benchmark
   ```

2. Run the benchmark script with your desired options:

   ```bash
   python benchmark.py [options]
   ```

   **Available Options:**

   | Option | Short | Type | Default | Description |
   |--------|-------|------|---------|-------------|
   | `--project` | `-p` | string | *required* | The project directory containing the model file |
   | `--model` | `-m` | string | *required* | Model to benchmark (e.g. bert, mnist_linear). The test file name without .py extension |
   | `--config` | `-c` | string | None | Model configuration to benchmark (e.g. tiny, base, large) |
   | `--training` | `-t` | flag | False | Benchmark training mode |
   | `--batch_size` | `-bs` | integer | 1 | Batch size, number of samples to process at once |
   | `--loop_count` | `-lp` | integer | 1 | Number of times to run the benchmark |
   | `--input_size` | `-isz` | integer | None | Input size of the input sample (if model supports variable input size) |
   | `--hidden_size` | `-hs` | integer | None | Hidden layer size (if model supports variable hidden size) |
   | `--output` | `-o` | string | None | Output JSON file to write results to. Results will be appended if file exists |
   | `--task` | `-ts` | string | "na" | Task to benchmark (e.g. classification, segmentation) |
   | `--data_format` | `-df` | string | "float32" | Data format (e.g. float32, bfloat16) |

   **Example:**

   ```bash
   python benchmark/benchmark.py -p tt-forge-fe -m mobilenetv2_basic -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-mobilenetv2_basic.json
   ```

3. Alternatively, you can run specific model tests using pytest:

   ```bash
   python -m pytest project/model_name.py
   ```

   **Example:**

   ```bash
   python -m pytest -svv tt-forge-fe/yolo_v8.py
   ```
