# TT-XLA Model Demos

This directory contains demonstration examples for running PyTorch and JAX models on Tenstorrent hardware via the [TT-XLA](https://github.com/tenstorrent/tt-xla) frontend.

TT-XLA is the primary frontend for PyTorch and JAX. It leverages a PJRT interface to integrate with TT-MLIR and Tenstorrent hardware, supporting both single and multi-chip configurations.

## Directory Structure

- **`cnn/`** - Computer vision models (PyTorch)
- **`nlp/`** - Natural language processing models
  - `pytorch/` - NLP models using PyTorch
  - `jax/` - NLP models using JAX

## Available Demos

### CNN (PyTorch)

| Model    | Description                                              | Demo Code                                        |
|----------|----------------------------------------------------------|--------------------------------------------------|
| ResNet   | Deep residual network for image classification (ResNet-18/34/50/101/152) | [`cnn/resnet_demo.py`](cnn/resnet_demo.py)       |
| Arnold   | Reinforcement learning model for game environments       | [`cnn/arnold_demo.py`](cnn/arnold_demo.py)       |

### NLP — PyTorch

| Model       | Description                                                        | Demo Code                                                      |
|-------------|---------------------------------------------------------------------|----------------------------------------------------------------|
| ALBERT      | Lightweight BERT variant for masked LM, QA, token & sequence classification | [`nlp/pytorch/albert_demo.py`](nlp/pytorch/albert_demo.py)     |
| BGE-M3      | Multi-lingual embedding model for dense, sparse, and ColBERT retrieval      | [`nlp/pytorch/bge3_demo.py`](nlp/pytorch/bge3_demo.py)         |
| BGE-M3 Encode | BGE-M3 encode function for sentence similarity and lexical matching       | [`nlp/pytorch/bge3_encode_demo.py`](nlp/pytorch/bge3_encode_demo.py) |
| OPT         | Open Pre-trained Transformer for causal LM, QA, and sequence classification | [`nlp/pytorch/opt_demo.py`](nlp/pytorch/opt_demo.py)           |

### NLP — JAX

| Model    | Description                                                  | Demo Code                                              |
|----------|--------------------------------------------------------------|--------------------------------------------------------|
| ALBERT   | Lightweight BERT variant for masked language modeling        | [`nlp/jax/albert_demo.py`](nlp/jax/albert_demo.py)     |
| GPT-2    | Autoregressive language model for text generation (Base/Medium/Large/XL) | [`nlp/jax/gpt_demo.py`](nlp/jax/gpt_demo.py)           |
| OPT      | Open Pre-trained Transformer for causal language modeling    | [`nlp/jax/opt_demo.py`](nlp/jax/opt_demo.py)           |

## Running the Demos

For details about how to set up an environment and run a demo, please see the [TT-Forge Getting Started](../../docs/src/getting_started.md) page.

If you encounter any issues or have questions, please file them at [github.com/tenstorrent/tt-forge/issues](https://github.com/tenstorrent/tt-forge/issues).

## Additional Resources

- [TT-XLA Documentation](https://docs.tenstorrent.com/tt-xla)
- [Getting Started Guide](https://docs.tenstorrent.com/tt-xla/getting_started.html)
- [TT-XLA GitHub Repository](https://github.com/tenstorrent/tt-xla)
