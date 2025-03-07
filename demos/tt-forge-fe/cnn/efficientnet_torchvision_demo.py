import torch
import torchvision.models as models
from PIL import Image
import requests
from torchvision import transforms
import forge

class EfficientNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, x):
        return self.model(x)

def run_efficientnet_inference(image_url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"):
    # Create model
    model = models.efficientnet_b0()
    # Download weights directly from torchvision
    state_dict = torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
        progress=True,
        check_hash=False
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Create wrapper
    wrapper = EfficientNetWrapper(model)

    # Download and load sample image
    img = Image.open(requests.get(image_url, stream=True).raw)
    tensor = wrapper.transform(img).unsqueeze(0)

    # Create input sample for compilation
    input_sample = (tensor,)

    print("Compiling model...")
    compiled_model = forge.compile(wrapper, input_sample)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = compiled_model(*input_sample)
    
    # Get predictions
    logits = output[0]  # Get first output tensor
    print("Logits shape:", logits.shape)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    print("Probabilities shape:", probabilities.shape)
    
    # Get top 5 predictions
    topk_scores = torch.topk(probabilities[0], k=5)
    scores = topk_scores.values.detach().cpu().numpy()
    indices = topk_scores.indices.detach().cpu().numpy()
    
    # Print top 5 predictions
    for idx in range(5):
        score = scores[idx]
        label_id = indices[idx]
        print(f'Top {idx+1}: Score {score:.4f}, Label ID {label_id}')

if __name__ == '__main__':
    run_efficientnet_inference()
