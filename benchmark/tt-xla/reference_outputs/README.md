# Reference Outputs for LLM Accuracy Testing

This directory contains reference output files (`.refpt`) used for token accuracy testing in tt-forge LLM benchmarks.

## What are .refpt Files?

`.refpt` (reference point) files contain precomputed ground truth data for accuracy testing. Each file stores:

- **reference_tokens**: Full token sequence from "Tale of Two Cities" text corpus
- **top5_tokens**: Top 5 predicted tokens at each position (from HuggingFace model on CPU/GPU)

## File Format

Each `.refpt` file is a PyTorch serialized dictionary:

```python
{
    'reference_tokens': torch.Tensor,  # Shape: [1, total_length]
    'top1_tokens': torch.Tensor,       # Shape: [total_length-1]
    'top5_tokens': torch.Tensor,       # Shape: [total_length-1, 5]
    'library_versions': {              # For reproducibility validation
        'torch': str,                  # e.g., "2.5.1"
        'transformers': str,           # e.g., "4.46.3"
    }
}
```

## How They're Used

During accuracy testing:

1. The first half of `reference_tokens` is used as the input prompt (prefill)
2. The second half serves as ground truth for validation
3. **Teacher forcing** is applied: model predictions are stored, but ground truth tokens are fed as input for the next iteration
4. TOP1 and TOP5 accuracy are computed by comparing predictions against reference data
5. Library versions (torch, transformers) are validated to ensure reproducibility

**TOP1 Accuracy**: Percentage where predicted token matches `top1_tokens` (argmax from reference model)
**TOP5 Accuracy**: Percentage where predicted token is in any of the top 5 from `top5_tokens`

## Generating New Reference Files

To generate a `.refpt` file for a new model:

```bash
python3 benchmark/tt-xla/scripts/generate_reference_outputs.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --output_file "benchmark/tt-xla/reference_outputs/Llama-3.2-1B-Instruct.refpt" \
    --total_length 1024
```

**Requirements:**
- Runs on CPU (forced to match accuracy testing environment for reproducibility)
- HuggingFace model must be accessible
- `tale-of-two-cities.txt.bz2` must exist in this directory

## Filename Convention

The filename must match what `TokenAccuracy.get_model_name_from_variant()` returns:

- Extract the model name from HuggingFace path (everything after last `/`)
- Example: `meta-llama/Llama-3.2-1B-Instruct` → `Llama-3.2-1B-Instruct.refpt`

## Available Models

### Models with Accuracy Tests

- Llama-3.2-1B-Instruct.refpt
- Llama-3.2-3B-Instruct.refpt
- Llama-3.1-8B-Instruct.refpt
- Mistral-7B-Instruct-v0.3.refpt
- Qwen2.5-7B-Instruct.refpt
- gemma-1.1-2b-it.refpt (pending generation)
- gemma-2-2b-it.refpt (pending generation)
- phi-1.refpt (pending generation)
- phi-1_5.refpt (pending generation)
- phi-2.refpt (pending generation)
- Falcon3-1B-Base.refpt (pending generation)
- Falcon3-3B-Base.refpt (pending generation)
- Falcon3-7B-Base.refpt (pending generation)
- Qwen2.5-0.5B-Instruct.refpt (pending generation)
- Qwen2.5-1.5B-Instruct.refpt (pending generation)
- Qwen2.5-3B-Instruct.refpt (pending generation)
- Qwen3-0.6B.refpt (pending generation)
- Qwen3-1.7B.refpt (pending generation)
- Qwen3-4B.refpt (pending generation)
- Qwen3-8B.refpt (pending generation)
- Ministral-8B-Instruct-2410.refpt (pending generation)

## Running Accuracy Tests

Accuracy tests are run by passing `--accuracy-testing true` to any LLM test:

```bash
# Single model in accuracy mode
pytest -svv benchmark/tt-xla/test_llms.py::test_llama_3_2_1b --accuracy-testing true --output-file results.json
```

## Source Text

The reference data is generated from "A Tale of Two Cities" by Charles Dickens, stored in `tale-of-two-cities.txt.bz2`.

## File Size

Each `.refpt` file is approximately 50KB for 1024 tokens.
