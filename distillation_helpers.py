import torch
from diffusers import StableDiffusionPipeline


def reverse_step(args,t:int,pipeline:StableDiffusionPipeline,
                 latents:torch.Tensor,prompt_embeds:torch.Tensor, added_cond_kwargs:dict):
    latent_model_input = torch.cat([latents] * 2) if args.do_classifier_free_guidance else latents
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
    
    # predict the noise residual
    noise_pred = pipeline.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=pipeline.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    # compute the previous noisy sample x_t -> x_t-1
    if args.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents