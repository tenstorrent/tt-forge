# NLP Demos for TT-Forge

This directory contains demos showcasing NLP models with TT-Forge.

## Available Demos

### 1. BERT Demo (`bert_demo.py`)
Basic BERT model demonstration.

### 2. Turkish BERT Demo (`turkish_bert_demo.py`)
Demonstrates using the `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` model with Forge compiler for Turkish language processing.

#### Features
- Loads a specialized Turkish BERT model
- Generates sentence embeddings for Turkish text
- Calculates semantic similarity between sentences
- Shows how to use mean pooling with BERT outputs

#### Usage
```bash
# Make sure you have the required dependencies installed
# Activate your virtual environment if needed
python turkish_bert_demo.py
```

The demo processes three example Turkish sentences and shows:
1. The generated embeddings for each sentence
2. The similarity scores between sentence pairs

This demonstrates how specialized language models can be compiled and used with Forge for non-English language processing tasks.
