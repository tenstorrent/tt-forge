# Getting Started with Forge Demos

This document walks you through how to set up to run demo models using TT-Forge. The following topics are covered:

* [Setting up a Front End to Run a Demo](#setting-up-a-front-end-to-run-a-demo)
* [Running a Demo](#running-a-demo)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-Forge Issues](https://github.com/tenstorrent/tt-forge/issues) page.

> **NOTE:** If you plan to do development work, please see the
> build instructions for the repo you want to work with.

## Setting up a Front End to Run a Demo

**Validated environment for this quickstart:** Ubuntu 24.04, Python 3.12.

> **NOTE:** These demo steps were validated on Ubuntu 24.04 with Python 3.12.
> The broader Tenstorrent software installation docs currently recommend Ubuntu
> 22.04 LTS for other workflows. Follow the frontend-specific install guide if
> you need to match that wider environment recommendation.

This section provides instructions for how to set up your frontend so you can run models from the TT-Forge repo.

Before running one of the demos in TT-Forge, you must:
1. Determine which frontend you want to use:
   * [TT-XLA](https://github.com/tenstorrent/tt-xla) - For use with JAX, TensorFlow, PyTorch
   * [TT-Forge-ONNX](https://github.com/tenstorrent/tt-forge-onnx) - For use with ONNX and PaddlePaddle

2. Decide what setup you want to use for the frontend:
   * Wheel
   * Docker
   * Build From Source

>**NOTE:** At this time, if you want to use TT-Forge-ONNX, you must use Docker or the build from source option.

3. Follow the installation instructions from the repo for your selected setup method:
   * [TT-XLA Wheel](https://docs.tenstorrent.com/tt-xla/getting_started.html)
   * [TT-XLA Docker](https://docs.tenstorrent.com/tt-xla/getting_started_docker.html)
   * [TT-Forge-ONNX Docker](https://docs.tenstorrent.com/tt-forge-onnx/getting_started_docker.html)
   * [TT-Forge-ONNX Build From Source](https://docs.tenstorrent.com/tt-forge-onnx/getting_started_build_from_source.html)

4. Return to this repo and follow the instructions in the [Running a Demo](#running-a-demo) section.

## Running a Demo

To run a demo, do the following:

1. Clone the TT-Forge repo (alternatively, you can download the script for the model you want to try):

```bash
git clone https://github.com/tenstorrent/tt-forge.git
```

2. Navigate into TT-Forge and run the following command:

```bash
git submodule update --init --recursive
```

3. Navigate to the folder for the frontend you want:
   * [TT-XLA Models](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla)
   * [TT-Forge-ONNX Models](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-onnx)

In this walkthrough, the [**resnet_demo.py**](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-xla/cnn/resnet_demo.py) from the TT-XLA folder is used.

4. From the main folder in the TT-Forge repository, run the **resnet_demo.py** script:

```bash
export PYTHONPATH=.
python demos/tt-xla/cnn/resnet_demo.py
```

If all goes well, you should see an image of a cat, and terminal output where the model predicts what the image is and presents a score indicating how confident it is in its prediction.
