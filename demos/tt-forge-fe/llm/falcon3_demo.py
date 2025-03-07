import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import forge

class SimpleModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Return only the logits for simpler compilation
        return outputs.logits

def demo_falcon3():
    # Initialize tokenizer and model
    model_name = "tiiuae/falcon3-1b-base"
    print(f"Loading {model_name}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    # Create wrapper model for compilation
    wrapped_model = SimpleModelWrapper(model)
    
    # Example prompts to showcase different capabilities
    prompts = [
        "What is machine learning?",
        "Write a short poem about artificial intelligence:",
        "Explain how a neural network works in simple terms:"
    ]

    # Prepare sample input for compilation
    sample_inputs = tokenizer(prompts[0], return_tensors="pt")
    input_ids = sample_inputs['input_ids']
    attention_mask = sample_inputs['attention_mask']
    
    # Compile model with forge
    print("Compiling model with forge...")
    input_sample = [input_ids, attention_mask]
    compiled_model = forge.compile(wrapped_model, input_sample)

    print("\nGenerating responses for example prompts...")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Generate response using original model since generation requires the full model
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    demo_falcon3()
