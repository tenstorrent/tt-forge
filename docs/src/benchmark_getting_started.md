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
   python -m pytest -svv tt-forge-fe/resnet_hf.py
   ```
