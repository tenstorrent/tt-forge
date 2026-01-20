# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_model(model_path="openlm-research/open_llama_3b", **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)

    # Use defaults or values from kwargs
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)
    if "num_hidden_layers" in kwargs and kwargs["num_hidden_layers"] is not None:
        config.num_hidden_layers = kwargs["num_hidden_layers"]

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(model_path, config=config)
    framework_model.eval()

    use_lora = kwargs.get("use_lora", False)
    if use_lora:
        lora_r = kwargs.get("lora_r", 4)
        lora_alpha = kwargs.get("lora_alpha", 8)
        # Applying LoRA to the last half of all layers due to memory constraints, this should be configurable
        ltt = range(framework_model.config.num_hidden_layers // 2, framework_model.config.num_hidden_layers)
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, task_type="CAUSAL_LM", layers_to_transform=ltt)
        framework_model = get_peft_model(framework_model, lora_config)

    # Using AutoTokenizer for default tokenizers for both openllama and llama 3.2
    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    return framework_model, tokenizer
