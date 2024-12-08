import torch
import os
from diffusers.models.normalization import AdaGroupNorm

import folder_paths

from comfy.utils import common_upscale

from .model import NoiseTransformer, SVDNoiseUnet

MODELS_DIR = os.path.join(folder_paths.models_dir, "npnets")
if "npnets" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["npnets"]
folder_paths.folder_names_and_paths["npnets"] = (current_paths, folder_paths.supported_pt_extensions)

class Noise_GoldenNoise:
    def __init__(self, model, noise, cond):
        self.model = model
        self.noise = noise
        self.cond = cond
        self.seed = self.noise.seed

    def generate_noise(self, input_latent):
        og_shape = input_latent["samples"].shape
        rescale = og_shape[-1] != 128 or og_shape[-2] != 128
        
        if rescale:
            input_latent = input_latent.copy()
            input_latent["samples"] = common_upscale(input_latent["samples"], 128, 128, "nearest-exact", "disabled")
            
        initial_noise = self.noise.generate_noise(input_latent).to(self.model["device"])
        cond = self.cond[0].clone().to(self.model["device"])
        cond = cond[:, :77, :]
        
        try:
            cond = cond.float().view(cond.shape[0], -1)
            text_emb = self.model["text_embedding"](initial_noise.float(), cond)

            encoder_hidden_states_svd = initial_noise
            encoder_hidden_states_embedding = initial_noise + text_emb

            golden_embedding = self.model["unet_embedding"](encoder_hidden_states_embedding.float())

            golden_noise = (
                self.model["unet_svd"](encoder_hidden_states_svd.float())
                + (2 * torch.sigmoid(self.model["_alpha"]) - 1) * text_emb
                + self.model["_beta"] * golden_embedding
            )
            
            golden_noise.to("cpu")
            
            if rescale:
                golden_noise = common_upscale(golden_noise, og_shape[-1], og_shape[-2], "nearest-exact", "disabled")
                
        except Exception as e:
            print("Noise could not be turned golden:", e)
            return initial_noise
        
        return golden_noise

class GoldenNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "conditioning": ("CONDITIONING",),
                "model_id": (["SDXL", "DreamShaper", "DiT"],),
                "npnet_model": (folder_paths.get_filename_list("npnets"),),
                "device": (["cuda", "cpu"],),
            },
        }

    RETURN_TYPES = ("NOISE",)
    OUTPUT_TOOLTIPS = ("The golden noise",)
    
    CATEGORY = "sampling/custom_sampling/noise"
    FUNCTION = "get_noise"
    DESCRIPTION = "Turn noise into golden noise."

    def __init__(self):
        self.models = {}

    def load_model(self, model_id, npnet_model_path, device="cuda"):
        if model_id not in self.models:
            unet_embedding = NoiseTransformer(resolution=128).to(device, torch.float32)
            unet_svd = SVDNoiseUnet(resolution=128).to(device, torch.float32)

            if model_id == 'DiT':
                  text_embedding = AdaGroupNorm(1024 * 77, 4, 1, eps=1e-6).to(device, torch.float32)
            else:
                  text_embedding = AdaGroupNorm(2048 * 77, 4, 1, eps=1e-6).to(device, torch.float32)

            gloden_unet = torch.load(npnet_model_path, weights_only=True, map_location=device)
            unet_svd.load_state_dict(gloden_unet["unet_svd"])
            unet_embedding.load_state_dict(gloden_unet["unet_embedding"])
            text_embedding.load_state_dict(gloden_unet["embeeding"])
            _alpha = gloden_unet["alpha"]
            _beta = gloden_unet["beta"]

            self.models[model_id] = {
                "unet_svd": unet_svd,
                "unet_embedding": unet_embedding,
                "text_embedding": text_embedding,
                "_alpha": _alpha,
                "_beta": _beta,
                "device": device
            }

        return self.models[model_id]

    def get_noise(self, noise, conditioning, model_id, npnet_model, device):
        npnet_model_path = folder_paths.get_full_path("npnets", npnet_model)
        
        model = self.load_model(model_id, npnet_model_path, device)
        cond = conditioning[0]
        
        return (Noise_GoldenNoise(model, noise, cond),)

NODE_CLASS_MAPPINGS = {
    "GoldenNoise": GoldenNoise,
}
