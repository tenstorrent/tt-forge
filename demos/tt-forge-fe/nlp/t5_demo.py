# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# FLAN-T5 Demo Script

import sys
import forge
# Add repository root to path to locate third_party modules
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.t5.pytorch import ModelLoader, ModelVariant


def run_flant5_demo_case(variant):

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    input_dict = loader.load_inputs()

    # T5 requires decoder_input_ids as a third input
    inputs = [input_dict["input_ids"], input_dict["attention_mask"], input_dict["decoder_input_ids"]]

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Run inference on Tenstorrent device
    output = compiled_model(*inputs)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # T5 variants
        ModelVariant.SMALL,
        ModelVariant.BASE,
        ModelVariant.FLAN_T5_SMALL,
        ModelVariant.FLAN_T5_BASE,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_flant5_demo_case(variant)
