# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import requests
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

def load_image_from_url(url):
    """Load an image from a URL."""
    return Image.open(requests.get(url, stream=True).raw).convert('RGB')

def run_vilt_inference(image_url, question, model_name="dandelin/vilt-b32-finetuned-vqa"):
    """Run VQA inference using ViLT model.
    
    Args:
        image_url (str): URL of the input image
        question (str): Question about the image
        model_name (str): Name of the ViLT model variant
        
    Returns:
        dict: Prediction result with answer and confidence
    """
    # Load model and processor
    processor = ViltProcessor.from_pretrained(model_name)
    model = ViltForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    
    # Load and process image
    image = load_image_from_url(image_url)
    
    # Prepare inputs
    inputs = processor(
        image,
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get top prediction
        confidence, idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        
        # Get predicted answer
        predicted_answer = model.config.id2label[idx.item()]
    
    return {
        'answer': predicted_answer,
        'confidence': confidence,
        'question': question
    }

if __name__ == "__main__":
    # Example usage
    test_cases = [
        {
            'image_url': "http://images.cocodataset.org/val2017/000000039769.jpg",
            'question': "What animal is in this image?"
        },
        {
            'image_url': "http://images.cocodataset.org/val2017/000000039769.jpg",
            'question': "What color is the cat?"
        },
        {
            'image_url': "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            'question': "Is the dog sitting or standing?"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nQuestion: {test_case['question']}")
        print(f"Image: {test_case['image_url']}")
        
        result = run_vilt_inference(
            image_url=test_case['image_url'],
            question=test_case['question']
        )
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
