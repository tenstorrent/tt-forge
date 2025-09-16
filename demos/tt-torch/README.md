# How to use
Build/install and activate the environment as you would for development on tt-torch: [Build Instructions](https://docs.tenstorrent.com/tt-torch/getting_started.html)

Additional python dependencies may be required for some demos. The following set of requirements covers all demos.
```
pip install -r benchmark/tt-torch/requirements.txt
pip install accelerate tabulate FlagEmbedding
```

From the project root, run
```
python demos/tt-torch/resnet50_demo.py
```
for the interactive demo that uses a single-device to classify an image using ResNet.


# Results

## resnet50_demo.py
When running this on the default image (http://images.cocodataset.org/val2017/000000039769.jpg):

![Image of two cats](http://images.cocodataset.org/val2017/000000039769.jpg)

The output is ResNet's top 5 predictions:
```
Top 5 Predictions
----------------------------------
tabby: 0.56640625
tiger cat: 0.236328125
Egyptian cat: 0.02734375
pillow: 0.005218505859375
remote control: 0.0037689208984375
```
Notice that in the code there was no explicit device management:
```Python
options = BackendOptions()
options.compiler_config = cc
# We didn't provide any explicit device while
# compiling the model.
tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)
```
This causes the model to be compiled onto the default device present in the board. The device acquisition and release get handled automatically.

## Benchmarking on 100 Images from ImageNet (resnet50_benchmark.py)

The `resnet50_benchmark.py` script can also be used to stream 100 images from the ImageNet dataset and compute the classification accuracy.
```
python demos/tt-torch/resnet50_benchmark.py
```

The script will process the first 100 images, output the top-1 accuracy, and display a summary at the end. This demonstrates the model's performance on a larger, more diverse set of images.

## Pipeline Parallel for Models Too Large for 1 Device (llama_7b_pipeline_parallel.py)

The `llama_7b_pipeline_parallel.py` script will split a large language model (Llama-7B) across two devices since it's too large to fit on a single device. The script automatically maps model layers to different devices and processes a text prompt to predict the next tokens.

```
python demos/tt-torch/llama_7b_pipeline_parallel.py
```

## Generative Models (llama3.2_generate_demo.py)

The `llama3.2_generate_demo.py` script demonstrates a text generation loop using the llama 3.2 3B parameter model. It shows usage of an on-device static kv-cache shared between prefill and decode stages, and produces a stream of output tokens.

```
python demos/tt-torch/llama3.2_generate_demo.py
```

## bge_m3_demo.py

The `bge_m3_demo.py` script demonstrates the BGE-M3 embedding model by processing two sets of sentences and computing similarity scores. The approches used follow the Sparse Embedding (Lexical Weigth) and Mult-Vector (ColBERT) demos from the [BGE-M3 Hugging Face documentation](https://huggingface.co/BAAI/bge-m3#generate-embedding-for-text).

```
python demos/tt-torch/bge_m3_demo.py
```
