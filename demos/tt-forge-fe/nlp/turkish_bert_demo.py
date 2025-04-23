#!/usr/bin/env python3
# Demo using emrecan/bert-base-turkish-cased-mean-nli-stsb-tr with Forge compiler

import torch
from transformers import BertModel, BertTokenizer
import forge

# Utility function for mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    # Model variant
    variant = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    
    print(f"Loading model: {variant}")
    
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(variant)
    model = BertModel.from_pretrained(variant, return_dict=False, use_cache=False)
    model.eval()
    
    # Prepare input - example Turkish sentences
    sentences = [
        "Bu örnek bir cümle.",
        "Türkçe doğal dil işleme çok önemlidir.",
        "Yapay zeka teknolojileri hızla gelişiyor."
    ]
    
    print(f"Processing sentences:")
    for i, sentence in enumerate(sentences):
        print(f"  {i+1}. {sentence}")
    
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    inputs = [
        encoded_input["input_ids"], 
        encoded_input["attention_mask"], 
        encoded_input["token_type_ids"]
    ]
    
    # Compile the model with Forge
    print("\nCompiling model with Forge...")
    compiled_model = forge.compile(
        model, 
        sample_inputs=inputs,
        module_name="bert_turkish_sentence_embeddings"
    )
    
    # Run inference with the compiled model
    print("Running inference...")
    outputs = compiled_model(*inputs)
    
    # Generate sentence embeddings via mean pooling
    sentence_embeddings = mean_pooling(outputs, encoded_input["attention_mask"])
    
    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    # Print results
    print("\nGenerated sentence embeddings:")
    for i, sentence in enumerate(sentences):
        print(f"\nEmbedding for: \"{sentence}\"")
        embedding = sentence_embeddings[i]
        print(f"Shape: {embedding.shape}")
        print(f"First 5 values: {embedding[:5]}")
    
    # Calculate similarity between sentences
    print("\nSimilarity between sentences:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarity = torch.dot(sentence_embeddings[i], sentence_embeddings[j])
            print(f"Similarity between sentence {i+1} and {j+1}: {similarity.item():.4f}")

if __name__ == "__main__":
    main()
