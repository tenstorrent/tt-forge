# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Densenet Demo Script

import forge
from third_party.tt_forge_models.vilt.question_answering.pytorch import ModelLoader, ModelVariant
import torch


class ViLtEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vilt_model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        image_token_type_idx=None,
    ):

        embeddings, masks = self.vilt_model.vilt.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            image_token_type_idx=image_token_type_idx,
        )
        return embeddings, masks


class ViltModelWrapper(torch.nn.Module):
    def __init__(self, model, task=None, text_seq_len=None):
        super().__init__()
        self.vilt_model = model
        self.task = task
        self.text_seq_len = text_seq_len

    def forward(self, embedding_output, attention_mask, head_mask=None):
        """
        Note:
            Embedding block is run on CPU due to dynamic shapes unsupported on TT.
            This wrapper runs the remaining model (encoder and task-specific layers) on device.

        Args:
            embedding_output (Tensor): Output from embedding layer.
            attention_mask (Tensor): Attention mask.
            head_mask (Tensor, optional): Optional attention head mask.

        Returns:
            Tuple: Task-specific output (e.g., logits) and intermediate outputs.

        More info: https://github.com/tenstorrent/tt-forge-onnx/issues/1119
        """

        head_mask = self.vilt_model.vilt.get_head_mask(head_mask, self.vilt_model.vilt.config.num_hidden_layers)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min

        encoder_outputs = self.vilt_model.vilt.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )

        sequence_output = encoder_outputs[0]

        sequence_output = self.vilt_model.vilt.layernorm(sequence_output)
        pooled_output = (
            self.vilt_model.vilt.pooler(sequence_output) if self.vilt_model.vilt.pooler is not None else None
        )

        viltmodel_output = (sequence_output, pooled_output) + encoder_outputs[1:]

        sequence_output, pooled_output = viltmodel_output[:2]

        if self.task == "maskedlm":

            if self.text_seq_len is None:
                raise ValueError("You cannot must provide text sequence length")

            text_features, _ = (sequence_output[:, : self.text_seq_len], sequence_output[:, self.text_seq_len :])

            mlm_logits = self.vilt_model.mlm_score(text_features)

            viltmodel_output = (mlm_logits,) + viltmodel_output[2:]

        if self.task == "qa":

            logits = self.vilt_model.classifier(pooled_output)

            viltmodel_output = (logits,) + viltmodel_output[2:]

        return viltmodel_output


def run_vilt_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    model = ViltModelWrapper(model, task="qa")
    embedding_output, attention_mask = text_vision_embedding_model(**inputs)

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[embedding_output, attention_mask])

    # Run inference on Tenstorrent device
    output = compiled_model(embedding_output, attention_mask)

    # Post-process the output
    print("Predicted answer:", loader.decode_output(output))

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.VQA,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_vilt_demo_case(variant)
