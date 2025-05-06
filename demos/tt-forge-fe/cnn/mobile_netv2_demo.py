# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import timm
import requests
from PIL import Image
from torchvision import transforms
import forge

# Common ImageNet classes (subset)
IMAGENET_CLASSES = {
    281: "tabby cat",
    282: "tiger cat",
    283: "Persian cat",
    284: "Siamese cat",
    285: "Egyptian cat",
    258: "Samoyed, Samoyede",  # White fluffy dog breed
    259: "Siberian husky",
    260: "German shepherd",
    261: "golden retriever",
    262: "Labrador retriever",
}


def load_image_from_url(url):
    """Load and preprocess an image from a URL."""
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")


def get_imagenet_transforms():
    """Get standard ImageNet preprocessing transforms."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def run_mobilenet_inference(image_url=None, model_name="mobilenetv2_100"):
    """Run inference with MobileNetV2 model.

    Args:
        image_url (str): URL of the input image
        model_name (str): Name of the MobileNetV2 model variant from timm

    Returns:
        dict: Prediction result with class label and confidence
    """
    # Load model and preprocessing
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    transform = get_imagenet_transforms()

    # Load and preprocess image
    image = load_image_from_url(image_url)
    input_tensor = transform(image).unsqueeze(0)

    # Create input sample for compilation
    input_sample = [torch.rand(1, 3, 224, 224)]

    # Compile model with forge
    compiled_model = forge.compile(model, input_sample)

    # Run inference
    with torch.no_grad():
        output = compiled_model(input_tensor)
        # Ensure output is the right shape [1, num_classes]
        if len(output[0].shape) == 1:
            logits = output[0].unsqueeze(0)
        else:
            logits = output[0]

        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Get top prediction
        confidence, class_idx = torch.max(probabilities[0], dim=0)
        class_idx = class_idx.item()
        confidence = confidence.item()

    # Get human-readable label
    label = IMAGENET_CLASSES.get(class_idx, f"Class {class_idx}")

    return {"label": label, "confidence": confidence, "class_idx": class_idx}


if __name__ == "__main__":
    # Example usage with different images
    test_images = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # Cat image
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",  # Dog image
    ]

    for image_url in test_images:
        print(f"\nProcessing image: {image_url}")
        result = run_mobilenet_inference(image_url=image_url)
        print(f"Prediction: {result['label']} (class {result['class_idx']})")
        print(f"Confidence: {result['confidence']:.3f}")
