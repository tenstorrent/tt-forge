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

# Forge modules
import forge
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge.config import CompilerConfig
from forge.verify.compare import compare_with_golden


# llama.utils.utils
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_model(model_path="openlm-research/open_llama_3b", **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)

    # Use defaults or values from kwargs
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)
    if "num_hidden_layers" in kwargs:
        config.num_hidden_layers = kwargs["num_hidden_layers"]

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", config=config)
    framework_model.eval()

    use_lora = kwargs.get("use_lora", False)
    if use_lora:
        lora_r = kwargs.get("lora_r", 4)
        lora_alpha = kwargs.get("lora_alpha", 8)
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, task_type="CAUSAL_LM")
        framework_model = get_peft_model(framework_model, lora_config)

    # Using AutoTokenizer for default tokenizers for both openllama and llama 3.2
    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    return framework_model, tokenizer


# Common constants

# Model path
MODEL_PATH = ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]


def prefil_on_cpu(model, input_ids):
    with torch.no_grad():
        transformer_outputs = model.model(
            input_ids=input_ids,  # Pass the entire updated sequence
        )
        hidden_states = transformer_outputs.last_hidden_state
    return hidden_states


@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("model_path", MODEL_PATH, ids=["_".join(item.split("/")) for item in MODEL_PATH])
def test_llama_prefill(
    training,
    batch_size,
    model_path,
    loop_count,
):

    if training:
        pytest.skip("Training is not supported")

    if batch_size > 1:
        pytest.skip("Batch size greater than 1 is not supported")

    if model_path == "openlm-research/open_llama_3b":
        pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 32 GB during compile time)")

    # Load Llama model and tokenizer
    model, tokenizer = load_model(model_path, return_dict=True)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Compiler configuration
    compiler_config = CompilerConfig()
    # @TODO - For now, we are skipping enabling MLIR optimizations, because it is not working with the current version of the model.
    # Turn on MLIR optimizations.
    # compiler_config.mlir_config = MLIRConfig().set_enable_optimizer(True)

    # This is the part of the model needed for prefill; model without the last Linear layer (lm_head)
    model_decoder = model.get_decoder()
    compiled_decoder = forge.compile(model_decoder, sample_inputs=input_ids, compiler_cfg=compiler_config)

    # Enable program cache on all devices
    settings = DeviceSettings()
    settings.enable_program_cache = True
    configure_devices(device_settings=settings)

    # Prefill Phase - Process the initial prompt on device
    # This what we actually want to benchmark, and measure the time taken.
    transformer_outputs = compiled_decoder(input_ids)
    start = time.time()
    for _ in range(loop_count):
        transformer_outputs = compiled_decoder(input_ids)
    end = time.time()

    # Get hidden states for all tokens from the last "transformer layer".
    hidden_states_compiled = transformer_outputs[0]
    hidden_states_compiled = hidden_states_compiled.to("cpu")

    # Get hidden states for all tokens from the last "transformer layer" calculated on CPU.
    hidden_states_framework = prefil_on_cpu(model, input_ids)

    # Compare result of prefilling on device with the result of prefilling on CPU.
    # Calculate the pcc for only the last vector in the hidden states tensor.
    assert compare_with_golden(hidden_states_framework[:, -1, :], hidden_states_compiled[:, -1, :])

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    input_size = len(input_ids[0])
    total_time = end - start
    total_tokens = input_size * loop_count

    tokens_per_sec = total_tokens / total_time
    model_name = "Llama Prefill"
    model_type = "Text Generation, Random Text Data"
    dataset_name = "Llama, Random Data"
    num_layers = -1  # Number of layers in the model is not relevant here.
    batch_size = 1  # Batch size is always 1 for text generation.

    input_sequence_length = len(input_ids[0])
    output_sequence_length = -1  # We are not generating any output here.
    # This will be changed when we add the decoding part of the model.

    print("====================================================================")
    print("| Llama Benchmark Results:                                         |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: : {total_time}")
    print(f"| Total tokens: {total_tokens}")
    print(f"| Token per second: {tokens_per_sec}")
    print(f"| Input size: {input_size}")
    print("====================================================================")

    result = {
        "model": model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(model_name.split())}_{input_size}_{loop_count}",
        "config": {"model_size": "small"},
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": "f32",  # This is we call dataformat, it should be generic, too, but for this test we don't experiment with it
        # "math_fidelity": math_fidelity, @TODO - For now, we are skipping these parameters, because we are not supporting them
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": input_sequence_length,
        "output_sequence_length": output_sequence_length,
        # This parameter can't have a generic value, so we are leaving it empty.
        "perf_analysis": False,
        "training": training,
        "measurements": [
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_tokens",
                "value": total_tokens,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
        ],
        "device_info": {
            "device_name": "",
            "galaxy": False,
            "arch": "",
            "chips": 1,
        },
        "device_ip": None,
    }

    return result


def benchmark(config: dict):

    training = config["training"]
    batch_size = config["batch_size"]
    model_path = MODEL_PATH[1]
    loop_count = config["loop_count"]

    return test_llama_prefill(training=training, batch_size=batch_size, model_path=model_path, loop_count=loop_count)
