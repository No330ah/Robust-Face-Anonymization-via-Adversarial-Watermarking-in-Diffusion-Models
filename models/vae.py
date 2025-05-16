# models/vae.py

from diffusers import AutoencoderKL
import torch

def load_vae_model(model_path: str, device='cuda'):
    """
    load VAE model，用于 latent 
    model_path: HuggingFace Diffusers
    """
    vae = AutoencoderKL.from_pretrained(model_path)
    vae.to(device)
    vae.eval()
    return vae
