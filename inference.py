import os
import torch
from PIL import Image
from torchvision import transforms
from einops import rearrange
from models.vae import load_vae_model
from models.arcface_load import load_arcface_model
from models.mm_dit import MM_DiT
from models.watermark import WatermarkEmbedder

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # shape: (1, 3, 256, 256)

def save_image(tensor, save_path):
    image = tensor.squeeze().detach().cpu()
    image = transforms.ToPILImage()(image)
    image.save(save_path)

@torch.no_grad()
def run_inference(image_path, crop_face_path, mask_path, vae_path, arcface_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae_model(vae_path, device=device)
    arcface = load_arcface_model(arcface_path, device=device)
    mm_dit = MM_DiT().to(device)
    watermark = WatermarkEmbedder().to(device)
    mm_dit.eval()

    image = preprocess_image(image_path).to(device)
    crop = preprocess_image(crop_face_path).to(device)
    mask = preprocess_image(mask_path).to(device)

    x_1 = vae.encode(image).latent_dist.sample()
    x_1 = (x_1 - vae.config.shift_factor) * vae.config.scaling_factor

    masked_img = image * (1 - mask)
    x_masked = vae.encode(masked_img).latent_dist.sample()
    x_masked = (x_masked - vae.config.shift_factor) * vae.config.scaling_factor

    mask_low = transforms.Resize(256 // 8)(mask)
    t = torch.sigmoid(torch.randn((1,), device=device)) / 3
    t_1 = t.view(1, 1, 1, 1)

    x_0 = torch.randn_like(x_1)
    x_t = (1 - t_1) * x_1 + t_1 * x_0
    x_t = x_t * mask_low + x_1 * (1 - mask_low)

    x_t_patch = rearrange(x_t, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    x_masked_patch = rearrange(x_masked, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    mask_patch = rearrange(mask_low, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    input_patch = torch.cat([x_t_patch, x_masked_patch, mask_patch], dim=-1)

    ID_feat, _ = arcface(crop)
    image_embed = torch.randn(1, 1, 512).to(device)
    ID_feat_exp = ID_feat.unsqueeze(1)
    pred = mm_dit(input_patch, ID_feat_exp, image_embed)

    x_hat = x_t - pred * t_1
    x_hat = rearrange(x_hat, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=16, w=16, ph=2, pw=2)
    x_out = vae.decode(x_hat / vae.config.scaling_factor + vae.config.shift_factor)[0]
    save_image(x_out, save_path)
