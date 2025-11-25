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
        uncond_prompt: str, # Negative prompt or empty string
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

            to_idle(encoder)
        else:
            # I we're doing text to image - start with a random noise
            latents = torch.rand(latents_shape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timesteps).to(device)

            # (batch_size, 4, latents_height, latents_width)
            model_input = latents
 
            if do_cfg:
                # We need to send the same latent with the prompt and with the negative prompt.
                # (batch_size, 4, height, width) -> (2 * batch_size, 4, height, width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model output is the predicted noise by the unet
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Now we have both uncond and cond output in model_output
                # We will split it into two separate tensors
                output_cond, output_uncond = model_output.chunk(2)
                # Now we combine them together. (Formula: output = w * (out_cond - out_uncond) + out_uncond)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # Remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, -1), (0, 255), clamp=True)
        # (batch_size, channel, height, width) -> (batch_size, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to('cpu', torch.uint8).numpy()

        return images[0]

def rescale(x, old_range, new_range, clamp):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x

def get_time_embedding(timestep):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


