# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import forge

def run_bert_classification(text, model_name="textattack/bert-base-uncased-SST-2"):
    """Run sentiment classification with BERT on SST-2 dataset.
    
    Args:
        text (str): Input text to classify
        model_name (str): Name of the BERT model variant
        
    Returns:
        dict: Classification result with sentiment label and confidence
    """
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Extract tensors from tokenizer output
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs.get('token_type_ids', None)
    
    # Create input sample for compilation
    if token_type_ids is not None:
        input_sample = [input_ids, attention_mask, token_type_ids]
    else:
        input_sample = [input_ids, attention_mask]
    
    # Compile model with forge
    compiled_model = forge.compile(model, input_sample)
    
    # Run inference
    with torch.no_grad():
        if token_type_ids is not None:
            outputs = compiled_model(input_ids, attention_mask, token_type_ids)
        else:
            outputs = compiled_model(input_ids, attention_mask)
        logits = outputs[0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get prediction
    predicted_value = logits[0].argmax(-1).item()
    confidence = probabilities[0][predicted_value].item()
    predicted_sentiment = model.config.id2label[predicted_value]
    
    return {
        'text': text,
        'sentiment': predicted_sentiment,
        'confidence': confidence
    }

if __name__ == "__main__":
    # Example usage - Sentiment Classification
    reviews = [
        "This movie was fantastic! I really enjoyed every moment.",
        "The plot was confusing and the acting was terrible.",
        "An absolute masterpiece of modern cinema!"
    ]
    
    for review in reviews:
        result = run_bert_classification(review)
        print(f"Review: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})\n")
