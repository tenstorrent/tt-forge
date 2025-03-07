# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import requests
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import forge

class ViltEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vilt_model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        image_token_type_idx=None,
    ):
        embeddings, masks = self.vilt_model.vilt.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            image_token_type_idx=image_token_type_idx,
        )
        return embeddings, masks

class ViltModelWrapper(torch.nn.Module):
    def __init__(self, model, task=None, text_seq_len=None):
        super().__init__()
        self.vilt_model = model
        self.task = task
        self.text_seq_len = text_seq_len

    def forward(self, embedding_output, attention_mask, head_mask=None):
        head_mask = self.vilt_model.vilt.get_head_mask(head_mask, self.vilt_model.vilt.config.num_hidden_layers)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min

        encoder_outputs = self.vilt_model.vilt.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.vilt_model.vilt.layernorm(sequence_output)
        pooled_output = (
            self.vilt_model.vilt.pooler(sequence_output) if self.vilt_model.vilt.pooler is not None else None
        )

        viltmodel_output = (sequence_output, pooled_output) + encoder_outputs[1:]
        sequence_output, pooled_output = viltmodel_output[:2]

        if self.task == "qa":
            logits = self.vilt_model.classifier(pooled_output)
            viltmodel_output = (logits,) + viltmodel_output[2:]

        return viltmodel_output

def load_image_from_url(url):
    """Load an image from a URL with caching."""
    import os
    import hashlib
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), 'image_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename from URL
    cache_file = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest() + '.jpg')
    
    # If image is cached, load it
    if os.path.exists(cache_file):
        return Image.open(cache_file).convert('RGB')
    
    # Download and cache the image
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(response.raw).convert('RGB')
        img.save(cache_file)
        return img
    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Error loading image from {url}: {e}")
        raise

def run_vilt_inference(image_url, question, model_name="dandelin/vilt-b32-finetuned-vqa", use_forge=True):
    """Run VQA inference using ViLT model with optional forge compilation.
    
    This function supports two modes of operation:
    1. Forge Compilation (use_forge=True): Uses the Forge compiler to optimize the model
       by splitting it into embedding and classifier components. This allows for better
       performance through compilation optimizations.
    2. Original Model (use_forge=False): Uses the HuggingFace model directly without
       any compilation. This is useful for comparing results and validating that the
       Forge compilation maintains model accuracy.
    
    Args:
        image_url (str): URL of the input image
        question (str): Question about the image
        model_name (str): Name of the ViLT model variant
        use_forge (bool): Whether to use forge compilation (True) or original model (False)
        
    Returns:
        dict: Prediction result containing:
            - answer (str): The model's predicted answer
            - confidence (float): Confidence score for the prediction
            - question (str): The original question asked
    """
    # Set model configurations
    config = ViltConfig.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)
    
    # Load model and processor
    processor = ViltProcessor.from_pretrained(model_name)
    model = ViltForQuestionAnswering.from_pretrained(model_name, config=config)
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
    
    if use_forge:
        # Create wrapper models for compilation
        embedding_wrapper = ViltEmbeddingWrapper(model)
        model_wrapper = ViltModelWrapper(model, task="qa")
        
        # Get embeddings
        with torch.no_grad():
            embeddings, attention_mask = embedding_wrapper(**inputs)
        
        # Create input samples for compilation
        model_input_sample = [embeddings.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)]
        
        # Compile model
        print("Compiling model...")
        module_name = "vilt_qa"
        compiled_model = forge.compile(model_wrapper, sample_inputs=model_input_sample, module_name=module_name)
        
        # Run inference
        with torch.no_grad():
            # Get output from compiled model
            output = compiled_model(*model_input_sample)
            logits = output[0]
    else:
        # Run inference with original model
        print("Running original model...")
        with torch.no_grad():
            output = model(**inputs)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output.logits
    
    # Get top prediction
    idx = logits.argmax(-1).item()
    predicted_answer = model.config.id2label[idx]
    
    # Get confidence
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0, idx].item()
    
    return {
        'answer': predicted_answer,
        'confidence': confidence,
        'question': question
    }

if __name__ == "__main__":
    """Example usage of the ViLT model with both Forge compilation and original model.
    
    This demo runs each test case twice:
    1. First with Forge compilation enabled - demonstrates the optimized path
    2. Then with the original HuggingFace model - allows for comparison of:
       - Prediction accuracy
       - Confidence scores
       - Model behavior consistency
    
    This helps validate that the Forge compilation maintains the model's accuracy
    while providing potential performance benefits.
    """
    
    test_cases = [
        {
            'image_url': "http://images.cocodataset.org/val2017/000000039769.jpg",
            'question': "What animal is in this image?"
        },
        {
            'image_url': "http://images.cocodataset.org/val2017/000000039769.jpg",
            'question': "What color is the dog?"
        },
        {
            'image_url': "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            'question': "What breed of dog is this?"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nQuestion: {test_case['question']}")
        print(f"Image: {test_case['image_url']}")
        
        # Try with forge compilation
        try:
            print("\nUsing Forge compilation:")
            result = run_vilt_inference(
                image_url=test_case['image_url'],
                question=test_case['question'],
                use_forge=True
            )
            print(f"Answer: {result['answer']} (Confidence: {result['confidence']:.4f})")
        except Exception as e:
            print(f"Error with Forge: {str(e)}")
            
        # Try without forge compilation
        try:
            print("\nUsing original model:")
            result = run_vilt_inference(
                image_url=test_case['image_url'],
                question=test_case['question'],
                use_forge=False
            )
            print(f"Answer: {result['answer']} (Confidence: {result['confidence']:.4f})")
        except Exception as e:
            print(f"Error with original model: {str(e)}")