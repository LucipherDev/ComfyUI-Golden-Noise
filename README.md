# ComfyUI-Golden-Noise
ComfyUI Custom Node for ["Golden Noise for Diffusion Models: A Learning Framework"](https://arxiv.org/abs/2411.09502) and most of the code is adapted from [here]( https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models). This node refines the initial latent noise in the diffusion process, enhancing both image quality and semantic coherence.

## Installation

1. Navigate to your ComfyUI's custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/LucipherDev/ComfyUI-Golden-Noise
```

3. Install requirements:
```bash
cd ComfyUI-Golden-Noise
pip install -r requirements.txt
```

## Usage

Download the pre-trained NPNet weights of Stable Diffusion XL, DreamShaper-xl-v2-turbo, and Hunyuan-DiT from [here](https://drive.google.com/drive/folders/1Z0wg4HADhpgrztyT3eWijPbJJN5Y2jQt) and put them in the **models/npnets** folder.

The node can be found in "sampling/custom_sampling/noise" category as "GoldenNoise".

Look at example workflow for more info.

### Inputs

- **noise**: Noise output from RandomNoise node
- **conditioning**: Prompt conditioning
- **model_id**: "SDXL", "DreamShaper", "DiT"
- **npnet_model**: Pretrained NPNet model
- **device**: "cuda", "cpu"

## Citation
Original Paper

```bibtex
@misc{zhou2024goldennoisediffusionmodels,
      title={Golden Noise for Diffusion Models: A Learning Framework}, 
      author={Zikai Zhou and Shitong Shao and Lichen Bai and Zhiqiang Xu and Bo Han and Zeke Xie},
      year={2024},
      eprint={2411.09502},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```