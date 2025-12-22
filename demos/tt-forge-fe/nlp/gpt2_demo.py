# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from forge.verify.verify import verify
# Add repository root to path to locate third_party modules
import sys
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.gpt2.pytorch import ModelLoader, ModelVariant

# Load model and inputs via ModelLoader
loader = ModelLoader(variant=ModelVariant.GPT2_SEQUENCE_CLASSIFICATION)
model = loader.load_model()
inputs_dict = loader.load_inputs()
inputs = [inputs_dict["input_ids"]]

# Compile with Forge
compiled_model = forge.compile(model, sample_inputs=inputs)

# Run verification
_, co_out = verify(inputs, model, compiled_model)

# For classification variant, decode and print sentiment
predicted_value = loader.decode_output(co_out)
print(f"Predicted Sentiment: {predicted_value}")
