# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
from transformers import AutoTokenizer, FlaxAlbertForSequenceClassification

MODEL = "bhadresh-savani/albert-base-v2-emotion"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
with jax.default_device(jax.devices("cpu")[0]):
    model = FlaxAlbertForSequenceClassification.from_pretrained(MODEL)


@jax.jit
def call_model(params, input_ids, token_type_ids, attention_mask):
    return model(params=params, input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


sentence = "The possibilities of AI make me so excited!"

inputs = tokenizer(sentence, return_tensors="jax")
print(inputs)
outputs = call_model(model.params, **inputs)
probabilities = jax.nn.softmax(outputs.logits, axis=-1).tolist()

id2label = model.config.id2label
print(f"Input sentence: {sentence}")
print("Probabilities for each emotion class:")
for idx, prob in enumerate(probabilities[0]):
    print(f"{id2label[idx]}: {float(prob):.4f}")
