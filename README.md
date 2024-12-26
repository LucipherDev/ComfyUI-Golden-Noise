# ComfyUI-Golden-Noise
ComfyUI Custom Node for ["Golden Noise for Diffusion Models: A Learning Framework"](https://arxiv.org/abs/2411.09502). Most of the code for this node is adapted from [here]( https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models). This node refines the initial latent noise in the diffusion process, enhancing both image quality and semantic coherence.

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

### Or Install via ComfyUI Manager

## Usage

Download the safetensors of the pre-trained NPNet weights of Stable Diffusion XL, DreamShaper-xl-v2-turbo, and Hunyuan-DiT from Huggingface [LucipherDev/Golden-Noise-NPNets](https://huggingface.co/LucipherDev/Golden-Noise-NPNets) and put them in the **models/npnets** folder.

The node can be found in "sampling/custom_sampling/noise" category as "GoldenNoise".

Take a look at the example workflow for more info.

### Inputs

- **noise**: Noise output from RandomNoise node
- **conditioning**: Prompt conditioning
- **model_id**: "SDXL", "DreamShaper", "DiT"
- **npnet_model**: Pretrained NPNet model
- **device**: "cuda", "cpu"

## Image Comparisons

![comparison](https://github.com/user-attachments/assets/1e285c39-b044-43c3-bc5c-2986e387cacb)
**Comparisons from the node*

This is a small comparison with only 50 images for each prompt, with and without using the golden noise node. The images were generated using the provided workflow (the random noise used to generate the first image is fed into the GoldenNoise node to generate the golden noise), and then each image was scored using [ImageReward](https://github.com/THUDM/ImageReward). The mean scores for with and without the node are shown in the comparison. From this, we can see that when using the node, the overall image quality is increased. However, The sample size is admittedly small, but it's all I can do with the resources and time I have.

Another comparison. This is taken from the original paper.
![x1](https://github.com/user-attachments/assets/a031e0c0-8ccc-4c4f-b861-e7e72f7c5a9d)
**Comparisons from the original paper*

## Citation

```bibtex
@misc{zhou2024goldennoisediffusionmodels,
      title={Golden Noise for Diffusion Models: A Learning Framework}, 
      author={Zikai Zhou and Shitong Shao and Lichen Bai and Zhiqiang Xu and Bo Han and Zeke Xie},
      year={2024},
      eprint={2411.09502},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.09502}, 
}
```
