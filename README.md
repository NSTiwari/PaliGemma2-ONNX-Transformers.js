# PaliGemma 2 ONNX Transformers.js
This repository is a step-by-step implementation of converting and quantizing the PaliGemma 2 Vision Language Model to ONNX weights, and inferencing it on the browser using Hugging Face Transformers.js.

## PaliGemma 2 to ONNX Conversion:
<img src="https://github.com/NSTiwari/PaliGemma2-ONNX-Transformers.js/blob/main/assets/paligemma2-onnx-pipeline.png"/>

## Run the Web App:

1. Clone the repository on your local machine.
2. Navigate to `cd PaliGemma2-ONNX-Transformers.js/Web App` directory.
3. Run `npm install` to install the packages.
4. Run `node server.js` to start the server.
5. Open `localhost:3000` on your web browser and start inferencing with PaliGemma 2.

## Results:
<img src="https://github.com/NSTiwari/PaliGemma2-ONNX-Transformers.js/blob/main/assets/paligemma2-onnx-output.gif"/>

## Resources & References

1. [Google DeepMind PaliGemma 2](https://developers.googleblog.com/en/introducing-paligemma-2-mix/)
2. Colab Notebooks: 
<table>
  <tr>
    <td><b>Convert and quantize PaliGemma 2 to ONNX</b></td>
    <td><a target="_blank" href="https://colab.research.google.com/github/NSTiwari/PaliGemma2-ONNX-Transformers.js/blob/main/Convert_PaliGemma2_to_ONNX.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td>
  </tr>
  <tr>
    <td><b>Inference PaliGemma 2 with Transformers.js</b></td>
    <td><a target="_blank" href="https://colab.research.google.com/github/NSTiwari/PaliGemma2-ONNX-Transformers.js/blob/main/Inference_PaliGemma2_with_Transformers_js.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td>
  </tr>
</table>

3. [**Medium Blog**](https://tiwarinitin1999.medium.com/) for step-by-step implementation.
4. [ONNX Community](https://huggingface.co/onnx-community)


## Acknowledgment:
<img src="https://github.com/NSTiwari/PaliGemma2-ONNX-Transformers.js/blob/main/assets/google.png">
This project was developed as part of Google's ML Developer Programs Vertex AI sprint. Thanks to the MLDP Team for their generous support in providing GCP credits and Colab units to help facilitate this project.

## Citation
If you find this project useful for your work, please cite it using the following BibTeX entry:

```
@misc{PaliGemma on Android using Hugging Face API,
  authors      = {Nitin Tiwari},
  title        = {Inference PaliGemma 2 with Transformers.js},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/NSTiwari/PaliGemma2-ONNX-Transformers.js}},
}
```
