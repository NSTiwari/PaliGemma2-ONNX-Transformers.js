#!/usr/bin/env python3
# Converts PaliGemma 2 to ONNX format and optionally quantizes + uploads to Hugging Face.
# Developed by Nitin Tiwari (github.com/NSTiwari)
#
# Usage:
#   python convert_paligemma2_to_onnx.py --model_id google/paligemma2-3b-mix-224
#   python convert_paligemma2_to_onnx.py --model_id google/paligemma2-3b-mix-224 --upload --hf_username your_username
#
# Install dependencies first:
#   pip install -r requirements.txt
#
# Make sure HF_TOKEN is set in your environment before running:
#   export HF_TOKEN=your_huggingface_token

import os
import json
import shutil
import argparse
import subprocess

import torch
import torch.nn as nn
import onnx
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    DynamicCache,
)
from optimum.onnx.graph_transformations import check_and_save_model


TEXT_MODEL_NAME = "decoder_model_merged.onnx"
VISION_MODEL_NAME = "vision_encoder.onnx"
EMBED_MODEL_NAME = "embed_tokens.onnx"

SUPPORTED_MODELS = [
    "google/paligemma2-3b-mix-224",
    "google/paligemma2-3b-mix-448",
    "google/paligemma2-3b-pt-224",
    "google/paligemma2-3b-ft-docci-448",
    "google/paligemma2-3b-pt-448",
    "google/paligemma2-3b-pt-896",
]


class VisionEncoder(nn.Module):
    """Wraps SigLIP vision tower + multimodal projector as a standalone exportable module."""

    def __init__(self, paligemma_model):
        super().__init__()
        self.config = paligemma_model.config
        self.vision_tower = paligemma_model.vision_tower
        self.multi_modal_projector = paligemma_model.multi_modal_projector

    def forward(self, pixel_values: torch.FloatTensor):
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        # Scale by 1/sqrt(hidden_size) — matches the original model's normalization
        image_features = image_features / (self.config.text_config.hidden_size ** 0.5)
        return image_features


class PatchedPaliGemmaForConditionalGeneration(PaliGemmaForConditionalGeneration):
    """
    Replaces the default forward() to accept flat positional args instead of a DynamicCache object.
    ONNX can't represent Python objects in its graph, so we flatten the KV cache into individual tensors.
    """

    def forward(self, *args):
        inputs_embeds, position_ids, *past_key_values_args = args
        config = self.config.text_config

        # Reconstruct DynamicCache from the flat key/value list
        if len(past_key_values_args) == 0:
            past_key_values = None
        else:
            past_key_values = DynamicCache(config.num_hidden_layers)
            for i in range(config.num_hidden_layers):
                key = past_key_values_args.pop(0)
                value = past_key_values_args.pop(0)
                past_key_values.update(key_states=key, value_states=value, layer_idx=i)

        batch_size = inputs_embeds.shape[0]

        # All-zeros attention mask tells the model to attend to every token
        output = self.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.zeros(batch_size, 1, 1, 1, dtype=torch.float32),
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        result = {"logits": output.logits}
        for i, (key, value) in enumerate(
            zip(output.past_key_values.key_cache, output.past_key_values.value_cache)
        ):
            result[f"present.{i}.key"] = key
            result[f"present.{i}.value"] = value

        return result


def load_model_and_processor(model_id):
    print(f"Loading {model_id} ...")
    model = PatchedPaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_dummy_inputs(model, processor):
    """Builds random tensors matching the model's expected input shapes for tracing."""
    text_config = model.config.text_config
    num_key_value_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    num_layers = text_config.num_hidden_layers
    hidden_size = text_config.hidden_size

    batch_size = 2
    seq_len = 32
    past_seq_len = 8

    past_kv = {
        f"past_key_values.{i}.{kv}": torch.zeros(
            batch_size, num_key_value_heads, past_seq_len, head_dim
        )
        for i in range(num_layers)
        for kv in ["key", "value"]
    }

    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(1, seq_len + 1).expand(batch_size, seq_len)

    text_inputs = dict(inputs_embeds=inputs_embeds, position_ids=position_ids, **past_kv)

    img_size = processor.image_processor.size
    pixel_values = torch.randn(2, 3, img_size["height"], img_size["width"], requires_grad=True)
    vision_inputs = dict(pixel_values=pixel_values)

    return text_inputs, vision_inputs, num_layers


def export_onnx_models(model, processor, output_folder):
    temp_dir = os.path.join(output_folder, "temp")
    final_dir = os.path.join(output_folder, "onnx")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    text_config = model.config.text_config
    num_layers = text_config.num_hidden_layers
    num_key_value_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    hidden_size = text_config.hidden_size

    vision_model = VisionEncoder(model)
    embed_layer = model.language_model.model.embed_tokens

    text_inputs, vision_inputs, _ = build_dummy_inputs(model, processor)

    # This flag needs to be set to False before exporting — there's a bug in PyTorch's ONNX
    # shape inference that causes the export to fail for certain graph structures.
    # See: https://github.com/pytorch/pytorch/issues/147259
    from torch.onnx._globals import GLOBALS
    GLOBALS.onnx_shape_inference = False

    # --- Language decoder (Gemma 2) ---
    print("Exporting language decoder (Gemma 2) ...")
    text_model_path = os.path.join(temp_dir, TEXT_MODEL_NAME)
    torch.onnx.export(
        model,
        args=tuple(text_inputs.values()),
        f=text_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=list(text_inputs.keys()),
        output_names=["logits"] + [
            f"present.{i}.{kv}" for i in range(num_layers) for kv in ["key", "value"]
        ],
        dynamic_axes={
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            **{
                f"past_key_values.{i}.{kv}": {0: "batch_size", 2: "past_sequence_length"}
                for i in range(num_layers)
                for kv in ["key", "value"]
            },
            "logits": {0: "batch_size", 1: "sequence_length"},
            **{
                f"present.{i}.{kv}": {0: "batch_size", 2: "total_sequence_length"}
                for i in range(num_layers)
                for kv in ["key", "value"]
            },
        },
        external_data_format=True,
    )

    # --- Vision encoder (SigLIP) ---
    print("Exporting vision encoder (SigLIP) ...")
    vision_model_path = os.path.join(temp_dir, VISION_MODEL_NAME)
    torch.onnx.export(
        vision_model,
        args=tuple(vision_inputs.values()),
        f=vision_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
    )

    # --- Token embedding layer ---
    print("Exporting embedding layer ...")
    embed_model_path = os.path.join(temp_dir, EMBED_MODEL_NAME)
    dummy_input_ids = torch.randint(0, embed_layer.num_embeddings, (2, 32))
    torch.onnx.export(
        embed_layer,
        args=(dummy_input_ids,),
        f=embed_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # Run shape inference on each model and copy to the final folder
    for name in (TEXT_MODEL_NAME, VISION_MODEL_NAME, EMBED_MODEL_NAME):
        print(f"Post-processing {name} ...")
        temp_path = os.path.join(temp_dir, name)
        onnx.shape_inference.infer_shapes_path(temp_path, check_type=True, strict_mode=True)
        onnx_model = onnx.load(temp_path)
        check_and_save_model(onnx_model, os.path.join(final_dir, name))

    shutil.rmtree(temp_dir)
    print(f"ONNX models saved to: {final_dir}")
    return final_dir


def save_artifacts(model, processor, output_folder):
    """Saves the model config, generation config, and processor files."""
    model.config.save_pretrained(output_folder)
    model.generation_config.save_pretrained(output_folder)
    processor.save_pretrained(output_folder)

    # Minify tokenizer.json — the default pretty-printed version is unnecessarily large
    tokenizer_path = os.path.join(output_folder, "tokenizer.json")
    with open(tokenizer_path, "r") as f:
        tokenizer = json.load(f)
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer, f)

    # Transformers.js needs head_dim and num_image_tokens at the top level of config.json
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["text_config"]["head_dim"] = model.config.text_config.head_dim
    config["num_image_tokens"] = config["text_config"]["num_image_tokens"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def run_quantization(onnx_folder, output_folder):
    """Calls quantize.py to produce fp16/int8/uint8/q4/q4f16/bnb4 variants."""
    cmd = [
        "python", "quantize.py",
        "--input_folder", onnx_folder,
        "--output_folder", output_folder,
        "--modes", "fp16", "int8", "uint8", "q4", "q4f16", "bnb4",
        "--per_channel",
        "--reduce_range",
        "--block_size", "64",
        "--is_symmetric",
        "--accuracy_level", "2",
        "--quant_type", "1",
    ]
    print("Running quantization (this takes ~40 minutes) ...")
    subprocess.run(cmd, check=True)


def upload_to_huggingface(output_dir, model_id, username):
    from huggingface_hub import upload_folder, create_repo

    repo_id = f"{username}/paligemma2-3b-mix-224-onnx"
    repo_id = create_repo(repo_id, exist_ok=True).repo_id

    upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message=f"{model_id} ONNX",
        ignore_patterns=["step_*", "epoch_*"],
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PaliGemma 2 to ONNX for use with Transformers.js"
    )
    parser.add_argument(
        "--model_id",
        default="google/paligemma2-3b-mix-224",
        choices=SUPPORTED_MODELS,
        help="PaliGemma 2 variant to convert (default: paligemma2-3b-mix-224)",
    )
    parser.add_argument(
        "--output_folder",
        default="output",
        help="Root folder for converted model files (default: ./output)",
    )
    parser.add_argument(
        "--skip_quantize",
        action="store_true",
        help="Skip the quantization step",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the final model to Hugging Face",
    )
    parser.add_argument(
        "--hf_username",
        default=None,
        help="Your Hugging Face username (required when --upload is set)",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set.\n"
            "Set it to your Hugging Face access token:\n"
            "  export HF_TOKEN=hf_..."
        )

    # torch.onnx.export traces the model using __len__, but the default Tensor __len__
    # raises an error during tracing — this one-liner patches it to work correctly.
    torch.Tensor.__len__ = lambda self: self.shape[0]

    model_output_dir = os.path.join(args.output_folder, args.model_id)
    os.makedirs(model_output_dir, exist_ok=True)

    model, processor = load_model_and_processor(args.model_id)
    save_artifacts(model, processor, model_output_dir)
    onnx_dir = export_onnx_models(model, processor, model_output_dir)

    if not args.skip_quantize:
        quant_dir = "onnx_model_quantized"
        run_quantization(onnx_dir, quant_dir)

        # Merge quantized weights back into the ONNX folder
        import glob
        for f in glob.glob(os.path.join(quant_dir, "*")):
            shutil.copy(f, onnx_dir)
        print(f"Quantized weights merged into: {onnx_dir}")

    if args.upload:
        if not args.hf_username:
            raise ValueError("--hf_username is required when using --upload")
        upload_to_huggingface(model_output_dir, args.model_id, args.hf_username)

    print("\nDone. Next step: run inference_paligemma2_with_transformers_js.py")


if __name__ == "__main__":
    main()
