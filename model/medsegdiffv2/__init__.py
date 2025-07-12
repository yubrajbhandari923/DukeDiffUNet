# https://github.com/SuperMedIntel/MedSegDiff/blob/master/guided_diffusion/train_util.py

# Actual Train Script Source:
# https://github.com/SuperMedIntel/MedSegDiff/blob/master/scripts/segmentation_train.py

from .guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_model,
    create_gaussian_diffusion
    # args_to_dict,
)
import torch
import logging

logging.basicConfig(level=logging.INFO)

from .guided_diffusion.resample import create_named_schedule_sampler


class MedSegDiffModel:
    def __init__(self, model_config, diffusion_config):
        

        # self.model, self.diffusion = create_model_and_diffusion(**self.args)
        self.model = create_model(
            model_config.image_size,
            model_config.num_channels,
            model_config.num_res_blocks,
            channel_mult=model_config.channel_mult,
            learn_sigma=model_config.learn_sigma,
            class_cond=model_config.class_cond,
            use_checkpoint=model_config.use_checkpoint,
            attention_resolutions=model_config.attention_resolutions,
            in_ch=model_config.in_ch,
            out_ch=model_config.out_ch,
            num_heads=model_config.num_heads,
            num_head_channels=model_config.num_head_channels,
            num_heads_upsample=model_config.num_heads_upsample,
            use_scale_shift_norm=model_config.use_scale_shift_norm,
            dropout=model_config.dropout,
            resblock_updown=model_config.resblock_updown,
            use_fp16=model_config.use_fp16,
            use_new_attention_order=model_config.use_new_attention_order,
            version=model_config.version,
            dims=model_config.dims,  # Yubraj Added
        )
        
        self.diffusion = create_gaussian_diffusion(
            steps=diffusion_config.diffusion_steps,
            learn_sigma=diffusion_config.learn_sigma,
            noise_schedule=diffusion_config.noise_schedule,
            use_kl=diffusion_config.use_kl,
            predict_xstart=diffusion_config.predict_xstart,
            rescale_timesteps=diffusion_config.rescale_timesteps,
            rescale_learned_sigmas=diffusion_config.rescale_learned_sigmas,
            dpm_solver=diffusion_config.dpm_solver,
            timestep_respacing=diffusion_config.timestep_respacing,
        )
        
        
    def get_model(self):
        return self.model

    def get_diffusion(self):
        return self.diffusion

    def get_schedule_sampler(self, name, maxt):
        return create_named_schedule_sampler(name, self.diffusion, maxt=maxt)
