# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoTokenizer, AutoModel
import forge

def calculate_similarity(text1, text2, model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"):
    """Calculate semantic similarity between two Turkish texts using BERT.
    
    Args:
        text1 (str): First Turkish text
        text2 (str): Second Turkish text
        model_name (str): Name of the Turkish BERT model variant
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize both inputs
    inputs = tokenizer(
        [text1, text2],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Extract tensors from tokenizer output
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Create input sample for compilation
    input_sample = [input_ids, attention_mask]
    
    # Compile model with forge
    compiled_model = forge.compile(model, input_sample)
    
    # Run inference
    with torch.no_grad():
        outputs = compiled_model(input_ids, attention_mask)
        # Get embeddings (use mean pooling)
        embeddings = torch.mean(outputs[0] * attention_mask.unsqueeze(-1), dim=1)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        similarity_score = similarity.item()
    
    return {
        'text1': text1,
        'text2': text2,
        'similarity_score': similarity_score
    }

if __name__ == "__main__":
    # Example usage - Turkish Text Similarity
    text_pairs = [
        (
            "Bu film gerçekten çok etkileyiciydi.",
            "Film beni derinden etkiledi."
        ),
        (
            "Hava bugün çok güzel.",
            "Yarın yağmur yağacak."
        ),
        (
            "Ankara Türkiye'nin başkentidir.",
            "Türkiye'nin başkenti Ankara'dır."
        )
    ]
    
    for text1, text2 in text_pairs:
        result = calculate_similarity(text1, text2)
        print(f"Text 1: {result['text1']}")
        print(f"Text 2: {result['text2']}")
        print(f"Similarity Score: {result['similarity_score']:.3f}\n")

