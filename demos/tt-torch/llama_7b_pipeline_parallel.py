# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend, BackendOptions
import torch
from tt_torch.tools.device_manager import DeviceManager
from transformers import AutoModelForCausalLM, AutoTokenizer # need transformers <= 4.52.4
from accelerate import infer_auto_device_map
from tt_torch.tools.verify import verify_against_golden
import tabulate


def main():
    model_name = "huggyllama/llama-7b"
    prompt = "I enjoy walking in the"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    parent_device = DeviceManager.create_parent_mesh_device([1, 2])

    # Create submeshes that target different devices
    device1 = DeviceManager.create_sub_mesh_device(parent_device, (0, 0))
    device2 = DeviceManager.create_sub_mesh_device(parent_device, (0, 1))

    dont_split = (
        model._no_split_modules if hasattr(model, "_no_split_modules") else None
    )
    device_map = infer_auto_device_map(
        model, max_memory={0: "11GiB", 1: "11GiB"}, no_split_module_classes=dont_split
    )

    options = BackendOptions()
    cc = CompilerConfig()
    options.compiler_config = cc
    cc.device_map = device_map
    cc.enable_consteval = True
    cc.consteval_parameters = True
    options.devices = [device1, device2]
    compiled_model = torch.compile(model, backend=backend, dynamic=False, options=options)
    out = compiled_model(**inputs)
    golden = model(**inputs)

    # Display the device map used by the compiled model
    print(
        f"\nDevice map:\n"
        f"{tabulate.tabulate(cc.device_map.items(), headers=['Module', 'Device'])}\n"
    )
    
    # Display input text
    print(f"Input text: '{prompt}'")
    # Show top 5 predicted tokens
    next_token_logits = out.logits[0, -1, :]
    top_k = 5
    top_logits, top_indices = torch.topk(next_token_logits, top_k)
    print(f"\nTop {top_k} predicted tokens:")
    for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
        token = tokenizer.decode([idx])
        print(f"{i+1}. '{token}'")
    print()
    
    verify_against_golden(
        tuple([golden.logits]), tuple([out.logits]), True, False, required_atol=0.1
    )

    DeviceManager.release_sub_mesh_device(device1)
    DeviceManager.release_sub_mesh_device(device2)
    DeviceManager.release_parent_device(parent_device)



if __name__ == "__main__":
    main()
