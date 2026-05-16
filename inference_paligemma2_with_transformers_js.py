#!/usr/bin/env python3
# Runs PaliGemma 2 ONNX inference using Transformers.js (Node.js backend).
# Developed by Nitin Tiwari (github.com/NSTiwari)
#
# Supported tasks:
#   Object detection  ->  --prompt "detect person"
#   Image captioning  ->  --prompt "caption en"
#   OCR               ->  --prompt "OCR"
#   Visual Q&A        ->  --prompt "What color is the car?"
#
# Usage:
#   python inference_paligemma2_with_transformers_js.py \
#       --image test_images/image_01.png \
#       --prompt "detect person"
#
# Install Python dependencies first:
#   pip install -r requirements.txt
#
# Node.js 20+ is also required (not a pip package):
#   https://nodejs.org/en/download/

import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path


PACKAGE_JSON = {
    "name": "paligemma2",
    "version": "1.0.0",
    "main": "index.js",
    "type": "module",
    "scripts": {
        "test": "echo \"Error: no test specified\" && exit 1"
    },
    "keywords": [],
    "author": "Nitin Tiwari",
    "license": "MIT",
    "description": "PaliGemma 2 ONNX inference with Transformers.js",
    "dependencies": {
        "@huggingface/transformers": "^3.1.2",
        "canvas": "^2.11.2"
    }
}

# The actual inference logic lives in Node.js because Transformers.js is a JS library.
# This template gets filled in with the image path and prompt, then executed with `node`.
INDEX_JS_TEMPLATE = """\
import {{ AutoProcessor, PaliGemmaForConditionalGeneration, load_image }} from "@huggingface/transformers";
import {{ createCanvas, loadImage }} from "canvas";
import fs from "fs";

function getRandomColor() {{
  const r = Math.floor(Math.random() * 256);
  const g = Math.floor(Math.random() * 256);
  const b = Math.floor(Math.random() * 256);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

const model_id = "{model_id}";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await PaliGemmaForConditionalGeneration.from_pretrained(model_id, {{
  dtype: {{
    embed_tokens: "fp16",
    vision_encoder: "fp16",
    decoder_model_merged: "q4",
  }},
}});

const image = "{image_path}";
const raw_image = await load_image(image);
const prompt = "{prompt}";

// Pull out the object name if this is a detection prompt like "<image>detect person"
const labelMatch = prompt.match(/detect (\\w+)/);
const label = labelMatch ? labelMatch[1] : "Unknown";
const capitalizedLabel = label.charAt(0).toUpperCase() + label.slice(1);

const inputs = await processor(raw_image, prompt);

try {{
  const response = await model.generate({{
    ...inputs,
    max_new_tokens: 100,
  }});

  const generatedIds = response.slice(null, [inputs.input_ids.dims[1], null]);
  const decodedAnswer = processor.batch_decode(generatedIds, {{ skip_special_tokens: true }});

  if (prompt.includes("<image>detect")) {{
    // PaliGemma encodes bounding boxes as <loc0123> tokens in 1024x1024 space
    const boundingBoxes = decodedAnswer[0].match(/<loc(\\d+)>/g);

    if (boundingBoxes && boundingBoxes.length === 4) {{
      const coordinates = boundingBoxes.map(tag =>
        parseInt(tag.replace("<loc", "").replace(">", ""))
      );
      const [y1, x1, y2, x2] = coordinates;

      // Scale coordinates back to the actual image dimensions
      const normX1 = Math.round((x1 / 1024) * raw_image.width);
      const normY1 = Math.round((y1 / 1024) * raw_image.height);
      const normX2 = Math.round((x2 / 1024) * raw_image.width);
      const normY2 = Math.round((y2 / 1024) * raw_image.height);

      console.log("Response:", decodedAnswer[0]);
      console.log("Bounding box (x1,y1,x2,y2):", [normX1, normY1, normX2, normY2]);

      const canvasImage = await loadImage(image);
      const canvas = createCanvas(canvasImage.width, canvasImage.height);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(canvasImage, 0, 0);

      const boxColor = getRandomColor();
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 5;
      ctx.strokeRect(normX1, normY1, normX2 - normX1, normY2 - normY1);

      // Draw label badge above the bounding box
      const labelPadding = 10;
      const textWidth = ctx.measureText(capitalizedLabel).width;
      const labelWidth = textWidth * 2.5;
      const labelHeight = 30;
      const labelY = normY1 - labelHeight;

      ctx.fillStyle = boxColor;
      ctx.fillRect(normX1, labelY, labelWidth, labelHeight);
      ctx.fillStyle = "white";
      ctx.font = "bold 20px Arial";
      ctx.fillText(capitalizedLabel, normX1 + labelPadding, labelY + labelHeight - labelPadding);

      const outputPath = "{output_image}";
      fs.writeFileSync(outputPath, canvas.toBuffer("image/jpeg"));
      console.log("Annotated image saved to:", outputPath);

    }} else {{
      console.log("No bounding box found in the model output.");
    }}

  }} else {{
    console.log("Response:", decodedAnswer[0]);
  }}

}} catch (error) {{
  console.error("Inference failed:", error);
  process.exit(1);
}}
"""


def check_node_installed():
    """Checks that Node.js is available. Tells the user how to install it if not."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"Node.js found: {version}")
    except FileNotFoundError:
        raise EnvironmentError(
            "Node.js is not installed or not in your PATH.\n"
            "Install it from: https://nodejs.org/en/download/"
        )


def setup_node_project(project_dir):
    """Creates the Node.js project and installs Transformers.js + canvas."""
    os.makedirs(project_dir, exist_ok=True)

    pkg_path = os.path.join(project_dir, "package.json")
    with open(pkg_path, "w") as f:
        json.dump(PACKAGE_JSON, f, indent=2)

    print(f"Setting up Node.js project in: {project_dir}")
    subprocess.run(["npm", "install"], cwd=project_dir, check=True)
    print("Dependencies installed.\n")


def write_index_js(project_dir, model_id, image_path, prompt, output_image):
    """Fills in the JS template and writes it to index.js in the project folder."""
    js_code = INDEX_JS_TEMPLATE.format(
        model_id=model_id,
        image_path=image_path.replace("\\", "/"),
        prompt=prompt,
        output_image=output_image.replace("\\", "/"),
    )
    index_path = os.path.join(project_dir, "index.js")
    with open(index_path, "w") as f:
        f.write(js_code)
    return index_path


def run_inference(project_dir, model_id, image_path, prompt, output_image="output.jpg"):
    """Writes index.js with the current inputs and runs it via Node.js."""
    write_index_js(project_dir, model_id, image_path, prompt, output_image)

    print(f"Prompt: {prompt}")
    print(f"Image:  {image_path}\n")
    result = subprocess.run(["node", "index.js"], cwd=project_dir)

    if result.returncode != 0:
        print("Node.js exited with an error. Check the output above.")

    return result.returncode == 0


def run_demo_examples(project_dir, model_id, test_images_dir, output_dir):
    """Runs all four example tasks shown in the notebook."""
    os.makedirs(output_dir, exist_ok=True)

    examples = [
        {
            "name": "Object Detection",
            "image": os.path.join(test_images_dir, "image_01.png"),
            "prompt": "<image>detect person",
            "output": os.path.join(output_dir, "detection_output.jpg"),
        },
        {
            "name": "Image Captioning",
            "image": os.path.join(test_images_dir, "image_02.png"),
            "prompt": "<image>caption en",
            "output": os.path.join(output_dir, "caption_output.jpg"),
        },
        {
            "name": "OCR",
            "image": os.path.join(test_images_dir, "image_03.png"),
            "prompt": "<image>OCR",
            "output": os.path.join(output_dir, "ocr_output.jpg"),
        },
        {
            "name": "Visual Q&A",
            "image": os.path.join(test_images_dir, "image_04.png"),
            "prompt": "<image>What time does the clock show?",
            "output": os.path.join(output_dir, "vqa_output.jpg"),
        },
    ]

    for example in examples:
        print(f"\n--- {example['name']} ---")
        image_path = os.path.abspath(example["image"])
        if not os.path.exists(image_path):
            print(f"  Skipping — image not found: {image_path}")
            continue
        run_inference(
            project_dir=project_dir,
            model_id=model_id,
            image_path=image_path,
            prompt=example["prompt"],
            output_image=os.path.abspath(example["output"]),
        )


def main():
    parser = argparse.ArgumentParser(
        description="PaliGemma 2 ONNX inference via Transformers.js"
    )
    parser.add_argument(
        "--model_id",
        default="NSTiwari/paligemma2-3b-mix-224-onnx",
        help="Hugging Face model ID for the ONNX weights (default: NSTiwari/paligemma2-3b-mix-224-onnx)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image. Omit to run all four demo examples.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Task prompt. The '<image>' prefix is added automatically if missing.\n"
            "  'detect person'         -> object detection\n"
            "  'caption en'            -> image captioning\n"
            "  'OCR'                   -> optical character recognition\n"
            "  'What color is the X?'  -> visual question answering"
        ),
    )
    parser.add_argument(
        "--output",
        default="output.jpg",
        help="Output path for the annotated image (detection only, default: output.jpg)",
    )
    parser.add_argument(
        "--project_dir",
        default="paligemma2_node",
        help="Directory for the Node.js project (default: ./paligemma2_node)",
    )
    parser.add_argument(
        "--skip_setup",
        action="store_true",
        help="Skip Node.js project setup if you've already run it before",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all four demo examples (detection, captioning, OCR, VQA)",
    )
    parser.add_argument(
        "--test_images_dir",
        default="test_images",
        help="Directory containing the test images for --demo mode (default: ./test_images)",
    )
    args = parser.parse_args()

    check_node_installed()

    if not args.skip_setup:
        setup_node_project(args.project_dir)

    if args.demo:
        run_demo_examples(
            project_dir=args.project_dir,
            model_id=args.model_id,
            test_images_dir=args.test_images_dir,
            output_dir="demo_outputs",
        )
        return

    # Single-image inference mode
    if not args.image or not args.prompt:
        parser.error("--image and --prompt are required unless --demo is set")

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Add the <image> prefix if the user forgot it
    prompt = args.prompt
    if not prompt.startswith("<image>"):
        prompt = f"<image>{prompt}"

    run_inference(
        project_dir=args.project_dir,
        model_id=args.model_id,
        image_path=image_path,
        prompt=prompt,
        output_image=os.path.abspath(args.output),
    )


if __name__ == "__main__":
    main()
