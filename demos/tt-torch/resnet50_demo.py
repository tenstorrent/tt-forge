# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch
import tt_torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import tabulate
import requests
from tt_torch.tools.device_manager import DeviceManager


def main(run_interactive):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).to(torch.bfloat16).eval()
    classes = weights.meta["categories"]
    preprocess = weights.transforms()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc
    tt_model = torch.compile(model, backend="tt", dynamic=False, options=options)

    headers = ["Top 5 Predictions"]
    topk = 5

    DEFAULT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

    def process_image(img_path):
        if img_path == "":
            url = DEFAULT_URL
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

    if not run_interactive:
        print("Running with default image URL: ", DEFAULT_URL)
        process_image("")
    else:
        prompt = 'Enter the path of the image (type "stop" to exit or hit enter to use a default image): '
        img_path = input(prompt)
        while img_path != "stop":
            process_image(img_path)
            img_path = input(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_interactive",
        action="store_true",
        help="Run the demo interactively as opposed to once with the default image.",
    )
    args = parser.parse_args()
    main(args.run_interactive)
