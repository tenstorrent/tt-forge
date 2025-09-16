# Getting Started with Forge Demos

This document walks you through how to set up to run demo models using TT-Forge. The following topics are covered:

* [Setting up a Front End to Run a Demo](#setting-up-a-front-end-to-run-a-demo)
* [Running a Demo](#running-a-demo)
* [Running Performance Benchmark Tests](#running-performance-benchmark-tests)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-Forge Issues](https://github.com/tenstorrent/tt-forge/issues) page.

> **NOTE:** If you plan to do development work, please see the
> build instructions for the repo you want to work with.

## Setting up a Front End to Run a Demo
This section provides instructions for how to set up your frontend so you can run models from the TT-Forge repo.

Before running one of the demos in TT-Forge, you must:
1. Determine which frontend you want to use:
   * [TT-XLA](https://github.com/tenstorrent/tt-xla) - For use with JAX, TensorFlow, PyTorch
   * [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe) - For use with ONNX and PaddlePaddle
   * [TT-Torch](https://github.com/tenstorrent/tt-torch) - (Deprecated, if you want to use PyTorch use TT-XLA)

2. Decide what setup you want to use for the frontend:
   * Wheel
   * Docker

3. Follow the installation instructions from the repo for your selected setup method:
   * [TT-XLA Wheel](https://docs.tenstorrent.com/tt-xla/getting_started.html)
   * [TT-XLA Docker](https://docs.tenstorrent.com/tt-xla/getting_started_docker.html)
   * [TT-Forge-FE Wheel](https://docs.tenstorrent.com/tt-forge-fe/getting_started.html)
   * [TT-Forge-FE Docker](https://docs.tenstorrent.com/tt-forge-fe/getting_started_docker.html)
   * [TT-Torch Wheel](https://docs.tenstorrent.com/tt-torch/getting_started.html) - (deprecated)
   * [TT-Torch Docker](https://docs.tenstorrent.com/tt-torch/getting_started_docker.html) - (deprecated)

4. Return to this repo and follow the instructions in the [Running a Demo](#running-a-demo) section.

## Running a Demo

To run a demo, do the following:

1. Clone the TT-Forge repo (alternatively, you can download the script for the model you want to try):

```bash
git clone https://github.com/tenstorrent/tt-forge.git
```

2. Navigate to the folder for the frontend you want:
   * [TT-XLA Models](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla) 
   * [TT-Forge-FE Models](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-fe)
   * [TT-Torch Models](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-torch) - (deprecated)

3. In this walkthrough, [**resnet_50_demo.py**](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-forge-fe/cnn/resnet_50_demo.py) from the TT-Forge-FE folder is used.

4. From the TT-Forge-FE folder for models, run the **resnet_50_demo.py** script. Navigate to the [main folder in the TT-Forge repository](https://github.com/tenstorrent/tt-forge/tree/main) and run the following commands:

```bash
export PYTHONPATH=.
python3 demos/tt-forge-fe/cnn/resnet_50_demo.py
```

If all goes well, you should see an image of a cat, and terminal output where the model predicts what the image is and presents a score indicating how confident it is in its prediction.

## Running Performance Benchmark Tests

To run performance benchmarks for all models, you need to install additional libraries that are not included in the Docker container or the wheel package.

### Prerequisites

1. **Install Python Requirements**

   Install the required Python packages from the `requirements.txt` file of the project you wish to run:

   ```bash
   pip install -r benchmark/[project]/requirements.txt
   ```

   **Example:**

   If you want to test a model from the TT-Torch project, you would run:

   ```bash
   pip install -r benchmark/tt-torch/requirements.txt
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

3. Alternatively, you can run specific model tests using `pytest`:

   ```bash
   python -m pytest [project]/[model_name].py
   ```

   **Example:**

   ```bash
   python -m pytest -svv tt-forge-fe/yolo_v8.py
   ```
