import torch
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import torch.nn as nn
from tqdm.auto import tqdm
import json
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Your code here


# # Load prompts from JSON file
# with open('prompts.json') as f:
#     prompts = json.load(f)

# prompt = prompts["prompts"]

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 200         # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise

batch_size = 1

#latent_tensors = []

def generate_diffusion_embedding(prompt, target):

    
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    target_input = tokenizer(target, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        target_embedding = text_encoder(target_input.input_ids.to(torch_device))
        target_cls_embedding = target_embedding.last_hidden_state[:,0,:]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    m = nn.Flatten()
    latents = m(latents)
    
    # disc = disc.to(torch_device)
    #import pdb; pdb.set_trace()
    # disc_tensor = disc(latents).view(-1)
    return latents, target_cls_embedding
    #latent_tensors.append(latents)

# Convert the list of latent tensors to a tensor
#latent_tensors = torch.cat(latent_tensors, dim=0)
