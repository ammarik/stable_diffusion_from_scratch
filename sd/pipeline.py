from typing import Dict, Optional
import numpy as np
import torch

from ddpm import DDPMSampler
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512

LATENT_WIDTH = 512//8
LATENT_HEIGHT = 512//8


def generate(
        prompt: str,
        uncond_prompt: str,
        input_image=None,
        strength: int=0.8,
        do_cfg: bool=True,
        cfg_scale: float=7.5,
        sampler_name: str='ddpm',
        n_inference_steps: int=90,
        models: Dict={},
        seed: Optional[float]=None,
        device = None,
        idle_device = None,
        tokenizer = None
        ):
    # Note: uncond_prompt = negative prompt

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError('strength must be 0 < strength <= 1')
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generate.manual_seed(seed)
        
        clip = models['clip']
        clip = clip.to(device)


        if do_cfg:
            # Convert the prompt into list of tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens)

            # Convert the  unconditional prompt into tokens using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            # (batch_size, seq_len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)

            # Concat them together
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            context = clip(cond_tokens)
        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMsampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f'Unknown sampler {sampler_name}')
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((HEIGHT, WIDTH))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # Scale values to be in range (-1, 1)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # Add batch dimension: (height, width, channel) -> (batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unqueeze(0)
            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.rand(latents_shape, generator=generator, device=device)

            # Run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise) # (z in the diagram)

            # set strength - tells how much we want the model to pay attention to the input image
            # More noise we add -> more creative model
            sampler.set_strength(strength=strength) # by setting the strength the scheduler will create timesteps schedule
            # Add noise to the latent representation of the image
            latents = sampler.add_noise(latents, sampler.timesteps[0]) # we will start with the maximal noise level








        

        
