# Qwen2.5-7B Model with Tensor Parallelism in JAX

This directory contains the implementation of the Qwen2.5-7B model using JAX with tensor parallelism. The model is optimized for multi-device environments using TT-xla.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo script:
   ```bash
   python qwen2_5_7b_demo.py
   ```

## Usage

The demo script demonstrates how to load the Qwen2.5-7B model, configure it for tensor parallelism, and run inference.

## Sample Input/Output

- **Input:** A sample text prompt.
- **Output:** The next predicted token and its probability.

## Dependencies

- JAX
- Flax
- NumPy

Ensure that you have the necessary hardware and software setup to run the model on multiple devices.
