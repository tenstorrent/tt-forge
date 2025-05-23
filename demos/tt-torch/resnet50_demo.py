# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend, BackendOptions
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import tabulate
import requests
from tt_torch.tools.device_manager import DeviceManager


def main():
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    model = models.resnet152(weights=weights).to(torch.bfloat16).eval()
    classes = weights.meta["categories"]
    preprocess = weights.transforms()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc
    tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)

    headers = ["Top 5 Predictions"]
    topk = 5
    prompt = 'Enter the path of the image (type "stop" to exit or hit enter to use a default image): '
    img_path = input(prompt)
    while img_path != "stop":
        if img_path == "":
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            img = Image.open(requests.get(url, stream=True).raw)
        elif img_path.startswith("http"):
            img = Image.open(requests.get(img_path, stream=True).raw)
        else:
            img = Image.open(img_path)
        img = preprocess(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(torch.bfloat16)

        top5, top5_indices = torch.topk(tt_model(img).squeeze().softmax(-1), topk)
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")

        rows = []
        for i in range(topk):
            rows.append([tt_classes[i]])

        print(tabulate.tabulate(rows, headers=headers))
        print()

        img_path = input(prompt)


if __name__ == "__main__":
    main()
