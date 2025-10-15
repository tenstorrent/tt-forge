# TT-Forge-FE Model Demos

This directory contains example implementations of popular deep learning models using TT-Forge-FE. These demos showcase how to use TT-Forge-FE to run inference on various computer vision and natural language processing models.

>**NOTE:** TT-Forge-FE does not support multi-chip configurations; it is for single-chip projects only.

## Available Demos

| Model                    | Model Type | Description                                                             | Demo Code                                              |
|--------------------------|------------|-------------------------------------------------------------------------|--------------------------------------------------------|
| AlexNet                  | CNN        | Classic deep CNN architecture for image classification                  | [`cnn/alexnet_demo.py`](cnn/alexnet_demo.py)           |
| AutoEncoder (Linear)     | CNN        | Simple linear autoencoder for image reconstruction                      | [`cnn/autoencoder_linear.py`](cnn/autoencoder_linear.py) |
| DeiT                     | CNN        | Data-efficient Image Transformer for image classification               | [`cnn/deit_demo.py`](cnn/deit_demo.py)                 |
| DenseNet                 | CNN        | Densely connected convolutional network for image classification        | [`cnn/densenet_demo.py`](cnn/densenet_demo.py)         |
| DLA                      | CNN        | Deep Layer Aggregation for image classification                         | [`cnn/dla_demo.py`](cnn/dla_demo.py)                   |
| EfficientNet             | CNN        | Scalable and efficient CNN model for image classification               | [`cnn/efficientnet_demo.py`](cnn/efficientnet_demo.py) |
| GhostNet                 | CNN        | Lightweight CNN using ghost modules for image classification            | [`cnn/ghostnet_demo.py`](cnn/ghostnet_demo.py)         |
| HRNet                    | CNN        | High-Resolution Network for image classification                        | [`cnn/hrnet_demo.py`](cnn/hrnet_demo.py)               |
| MNIST                    | CNN        | CNN for handwritten digit classification                                | [`cnn/mnist_demo.py`](cnn/mnist_demo.py)               |
| MobileNetV1              | CNN        | Efficient CNN for mobile vision applications                            | [`cnn/mobile_netv1_demo.py`](cnn/mobile_netv1_demo.py) |
| MobileNetV2              | CNN        | Lightweight CNN for MNIST digit classification                          | [`cnn/mobile_netv2_demo.py`](cnn/mobile_netv2_demo.py) |
| MonoDepth2               | CNN        | Monocular depth estimation model                                        | [`cnn/monodepth2_demo.py`](cnn/monodepth2_demo.py)     |
| RegNet                   | CNN        | Self-Regulated Network for image classification                         | [`cnn/regnet_demo.py`](cnn/regnet_demo.py)             |
| ResNet-50                | CNN        | Deep residual network for image classification                          | [`cnn/resnet_50_demo.py`](cnn/resnet_50_demo.py)       |
| ResNet-50 (ONNX)         | CNN        | ResNet-50 model using ONNX format                                       | [`cnn/resnet_onnx_demo.py`](cnn/resnet_onnx_demo.py)   |
| ResNeXt                  | CNN        | Aggregated residual transformation model for image classification       | [`cnn/resnext_demo.py`](cnn/resnext_demo.py)           |
| SegFormer                | CNN        | Transformer-based efficient image classifier                            | [`cnn/segformer_demo.py`](cnn/segformer_demo.py)       |
| Swin Transformer         | CNN        | Hierarchical vision transformer for image classification                | [`cnn/swin_demo.py`](cnn/swin_demo.py)                 |
| VGG                      | CNN        | Classic CNN with deep convolutional layers                              | [`cnn/vgg_demo.py`](cnn/vgg_demo.py)                   |
| ViT                      | CNN        | Vision Transformer for image classification                             | [`cnn/vit_demo.py`](cnn/vit_demo.py)                   |
| WideResNet               | CNN        | Wider variant of ResNet for image classification                        | [`cnn/wideresnet_demo.py`](cnn/wideresnet_demo.py)     |
| Xception                 | CNN        | Depthwise separable CNN for image classification                        | [`cnn/xception_demo.py`](cnn/xception_demo.py)         |
| BERT                     | NLP        | Transformer-based model for language understanding                      | [`nlp/bert_demo.py`](nlp/bert_demo.py)                 |
| Bloom                    | NLP        | Causal language model for text generation                               | [`nlp/bloom_demo.py`](nlp/bloom_demo.py)               |
| DistilBERT               | NLP        | Lightweight version of BERT for masked language modeling                | [`nlp/distilbert_demo.py`](nlp/distilbert_demo.py)     |
| Falcon                   | NLP        | Causal language model for code/text generation                          | [`nlp/falcon_demo.py`](nlp/falcon_demo.py)             |
| GPT-Neo                  | NLP        | GPT-style autoregressive transformer                                    | [`nlp/gptneo_demo.py`](nlp/gptneo_demo.py)             |
| RoBERTa                  | NLP        | Robust BERT variant for sentiment analysis                              | [`nlp/roberta_demo.py`](nlp/roberta_demo.py)           |
| SqueezeBERT              | NLP        | Lightweight transformer for text classification                         | [`nlp/squeezebert_demo.py`](nlp/squeezebert_demo.py)   |

## Running the Demos

For details about how to set up an environment and run a demo, please see the [tt-forge Getting Started](../../docs/src/getting-started.md) page.

If you encounter any issues or have questions, please file them at [github.com/tenstorrent/tt-forge/issues](https://github.com/tenstorrent/tt-forge/issues).

## Additional Resources

- [TT-Forge-FE Documentation](https://docs.tenstorrent.com/tt-forge-fe/)
- [Getting Started Guide](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html)
- [TT-Forge-FE GitHub Repository](https://github.com/tenstorrent/tt-forge)
