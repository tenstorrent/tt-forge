# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
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


from third_party.tt_forge_models.deepseek.deepseek_math.pytorch import (
    ModelLoader as MathLoader,
)
from third_party.tt_forge_models.deepseek.deepseek_math.pytorch import (
    ModelVariant as MathVariant,
)

DEEPSEEK_MATH_VARIANTS = [
    MathVariant.DEEPSEEK_7B_INSTRUCT,
]


class DeepSeekWrapper(torch.nn.Module):
    def __init__(self, model, max_new_tokens=200):
        super().__init__()
        self.model = model
        self.max_new_tokens = max_new_tokens

    def forward(self, input_tensor):
        return self.model(input_tensor, max_new_tokens=self.max_new_tokens).logits


# Load Model and Tokenizer
loader = MathLoader(variant=MathVariant.DEEPSEEK_7B_INSTRUCT)
model = loader.load_model()
tokenizer = loader._load_tokenizer()
framework_model = DeepSeekWrapper(model)
framework_model.eval()

# Prepare inputs
padded_inputs = loader.load_inputs()

# Compile model
compiled_model = forge.compile(
    framework_model,
    sample_inputs=[padded_inputs],
)

# Verify correctness
verify([padded_inputs], framework_model, compiled_model)

# Generate output
generated_text = loader.decode_output(
    max_new_tokens=512,
    model=compiled_model,
    inputs=padded_inputs,
    tokenizer=tokenizer,
)
print(generated_text)
