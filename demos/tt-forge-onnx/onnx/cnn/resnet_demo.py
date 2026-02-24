# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet ONNX Demo Script

import onnx
import onnxruntime as ort
import torch

import forge
from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant


def run_resnet_onnx(variant: ModelVariant):
    """
    Run a ResNet ONNX model
    """

    # Load Model and inputs using the ModelLoader
    model_loader = ModelLoader(variant=variant)
    pytorch_resnet_model = model_loader.load_model()
    sample_input_tensor = model_loader.load_inputs()

    # Export the PyTorch model to ONNX
    onnx_model_path = f"resnet_{variant.value}.onnx"
    torch.onnx.export(
        pytorch_resnet_model,
        sample_input_tensor,
        onnx_model_path,
        opset_version=17,
    )

    # Load and validate the exported ONNX model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # Run ONNX reference inference (CPU) for comparison
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    onnx_output = ort_session.run(
        None,
        {input_name: sample_input_tensor.detach().numpy()},
    )
    onnx_predicted_index = onnx_output[0].argmax(-1).item()
    onnx_predicted_class = pytorch_resnet_model.config.id2label[onnx_predicted_index]

    # Compile the model
    compiled_resnet_model = forge.compile(
        onnx_model,
        sample_inputs=[sample_input_tensor],
    )

    # Run inference
    forge_output = compiled_resnet_model(sample_input_tensor)

    # Post-process
    forge_predicted_index = forge_output[0].argmax(-1).item()
    forge_predicted_class = pytorch_resnet_model.config.id2label[forge_predicted_index]

    print("[Result] Predicted classes:")
    print(f"  ONNX (CPU reference) : {onnx_predicted_class}")
    print(f"  Forge (Tenstorrent device)   : {forge_predicted_class}")


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.RESNET_50_HF,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_resnet_onnx(variant)
