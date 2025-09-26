# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import pytest
import time
import socket
import subprocess
import json
import random
import os
from datetime import datetime

# Third-party modules
import torch
from transformers import LlamaTokenizer
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# Forge modules
import forge
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge.config import CompilerConfig
from forge.verify.compare import compare_with_golden
from forge._C import DataFormat

from benchmark.utils import get_ffe_device_arch

# Utils
from utils import load_model


class LlamaModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_ids, attention_mask=None, position_ids=None, *past_key_values):

        inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values:
            past_key_values = [past_key_values[i : i + 2] for i in range(0, len(past_key_values), 2)]
        else:
            past_key_values = None

        if past_key_values is not None:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length()
        else:
            past_key_values_length = 0

        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        print("type of outputs: ", type(outputs))
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # Flatten past_key_values for output
        flattened_past_key_values = []
        for layer_past in past_key_values:
            flattened_past_key_values.extend(layer_past)

        return [logits] + flattened_past_key_values


def calculate_attention_mask_and_postion_ids(
    padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
):

    # Calculate attention mask
    attention_mask = torch.zeros(padded_past_key_values_seq_length + input_seq_length, dtype=torch.long)
    attention_mask[:non_padding_past_key_values_seq_length] = 1
    attention_mask[-1] = 1
    attention_mask = attention_mask.unsqueeze(0)

    # Calculate position ids
    position_ids = torch.arange(
        non_padding_past_key_values_seq_length,
        input_seq_length + non_padding_past_key_values_seq_length,
        dtype=torch.long,
    )
    position_ids = position_ids.unsqueeze(0)

    return attention_mask, position_ids


# Common constants

# Model path
MODEL_PATH = ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"]

# Data format configurations
DATA_FORMAT = [
    "bfloat16",
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
@pytest.mark.parametrize("model_path", MODEL_PATH, ids=["_".join(item.split("/")) for item in MODEL_PATH])
def test_llama_decode(
    training,
    batch_size,
    model_path,
    loop_count,
    data_format,
    model_name,
    measure_cpu,
):

    if data_format == "bfloat16":
        torch_dtype = torch.bfloat16
        data_format_override = DataFormat.Float16_b
        compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    else:
        torch_dtype = torch.float32
        compiler_cfg = CompilerConfig()

    if training:
        pytest.skip("Training is not supported")

    if batch_size > 1:
        pytest.skip("Batch size greater than 1 is not supported")

    # Load model with cache enabled
    use_fast = False if model_path == "openlm-research/open_llama_3b" else True
    model, tokenizer = load_model(
        model_path,
        use_cache=True,
        use_fast=use_fast,
    )

    model.config.max_position_embeddings = 2048
    max_sequence_length = model.config.max_position_embeddings
    framework_model = LlamaModelWrapper(model)
    framework_model = framework_model.to(torch_dtype)
    framework_model.eval()

    # Skip CPU measurement for LLaMA models
    cpu_fps = -1.0

    if model_path == "openlm-research/open_llama_3b":
        tokenizer.pad_token_id = model.config.pad_token_id
    elif model_path == "meta-llama/Llama-3.2-1B":
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run Prefill on CPU with cache to get the initial logits and past key-values
    prefill_output = framework_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    next_token_logits = prefill_output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    generated_tokens = inputs.input_ids
    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    model_inputs = [next_token.unsqueeze(0)]
    model_inputs.extend(prefill_output[1:])

    # Zero Pad past key values in key_value_seq_len(i.e -2) dimension
    max_new_tokens = loop_count  # Use loop_count as max_new_tokens for benchmarking
    non_padding_seq_length = prefill_output[1].shape[-2]
    for idx, past_key_or_values_states in enumerate(model_inputs[1:]):
        model_inputs[idx + 1] = torch.cat(
            [
                past_key_or_values_states,
                torch.zeros(
                    past_key_or_values_states.shape[-4],
                    past_key_or_values_states.shape[-3],
                    max_sequence_length - non_padding_seq_length,
                    past_key_or_values_states.shape[-1],
                ).to(past_key_or_values_states.dtype),
            ],
            dim=-2,
        )

    # Calculate attention mask and postion_ids
    padded_past_key_values_seq_length = model_inputs[1].shape[-2]
    input_seq_length = model_inputs[0].shape[-1]
    attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
        padded_past_key_values_seq_length, non_padding_seq_length, input_seq_length
    )

    # Compile the model
    module_name = "LlamaModelDecode"
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[model_inputs[0], attention_mask, position_ids, *model_inputs[1:]],
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )
    compiled_model.save(f"{model_name}.ttnn")

    # Enable program cache on all devices
    settings = DeviceSettings()
    settings.enable_program_cache = True
    configure_devices(device_settings=settings)

    # Run decode stage on TT device and generate tokens
    start = time.time()
    tokens_generated = 0

    for max_new_tokens_idx in range(max_new_tokens):
        non_padding_past_key_values_seq_length = non_padding_seq_length + max_new_tokens_idx
        padded_past_key_values_seq_length = model_inputs[1].shape[-2]
        input_seq_length = model_inputs[0].shape[-1]
        attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
            padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
        )

        tt_inputs = [model_inputs[0], attention_mask, position_ids, *model_inputs[1:]]
        tt_output = compiled_model(*tt_inputs)

        logits = tt_output[0]
        past_key_values = tt_output[1:]

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        model_inputs = [next_token.unsqueeze(0)]
        model_inputs.extend(past_key_values)

        # Move generated token states to padding position
        for idx in range(len(model_inputs[1:])):
            model_inputs[idx + 1][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[idx + 1][:, :, -1, :]
            model_inputs[idx + 1] = model_inputs[idx + 1][:, :, :-1, :].contiguous()

        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)
        tokens_generated += 1

    end = time.time()

    # Calculate metrics
    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    device_arch = get_ffe_device_arch()
    input_size = len(inputs.input_ids[0])
    total_time = end - start
    total_tokens = tokens_generated

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    full_model_name = "Llama Decode"
    model_type = "Text Generation, Random Text Data"
    dataset_name = "Llama, Random Data"
    num_layers = -1
    batch_size = 1

    input_sequence_length = len(inputs.input_ids[0])
    output_sequence_length = tokens_generated

    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print("====================================================================")
    print("| Llama Decode Benchmark Results:                                  |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {full_model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: {total_time}")
    print(f"| Total tokens generated: {total_tokens}")
    print(f"| Token per second: {tokens_per_sec}")
    print(f"| Input size: {input_size}")
    print(f"| Generated text: {generated_text}")
    print("====================================================================")

    result = {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(full_model_name.split())}_{input_size}_{tokens_generated}",
        "config": {"model_size": "small"},
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": data_format,
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": input_sequence_length,
        "output_sequence_length": output_sequence_length,
        "perf_analysis": False,
        "training": training,
        "measurements": [
            {
                "iteration": 1,
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_tokens",
                "value": total_tokens,
                "target": -1,
                "device_power": -1.0,
                "device_temperature": -1.0,
            },
            {
                "iteration": 1,
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,
                "device_power": -1.0,
                "device_temperature": -1.0,
            },
            {
                "iteration": 1,
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "cpu_fps",
                "value": cpu_fps,
                "target": -1,
                "device_power": -1.0,
                "device_temperature": -1.0,
            },
        ],
        "device_info": {
            "device_name": machine_name,
            "galaxy": False,
            "arch": device_arch,
            "chips": 1,
        },
        "device_ip": None,
    }

    return result


def benchmark(config: dict):

    training = config["training"]
    batch_size = config["batch_size"]
    model_path = MODEL_PATH[0]
    loop_count = config["loop_count"]
    model_name = config["model"]
    data_format = config["data_format"]
    measure_cpu = config["measure_cpu"]

    return test_llama_decode(
        training=training,
        batch_size=batch_size,
        model_path=model_path,
        loop_count=loop_count,
        model_name=model_name,
        data_format=data_format,
        measure_cpu=measure_cpu,
    )
