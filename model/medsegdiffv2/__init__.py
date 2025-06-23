# https://github.com/SuperMedIntel/MedSegDiff/blob/master/guided_diffusion/train_util.py

# Actual Train Script Source:
# https://github.com/SuperMedIntel/MedSegDiff/blob/master/scripts/segmentation_train.py

from .guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    # args_to_dict,
)
import torch
import logging

logging.basicConfig(level=logging.INFO)

from .guided_diffusion.resample import create_named_schedule_sampler


class MedSegDiffModel:
    def __init__(self, args=None):
        if args is None:
            args = model_and_diffusion_defaults()
        else:
            default = model_and_diffusion_defaults()

            del_keys = []
            # remove unexpected keys
            for key in args:
                if key not in default:
                    del_keys.append(key)
            
            for key in del_keys:
                del args[key]

            # add missing keys
            for key in default:
                if key not in args:
                    args[key] = default[key]

        # self.args = args_to_dict(args)
        self.args = args
        self.model, self.diffusion = create_model_and_diffusion(**self.args)

    def get_model(self):
        return self.model

    def get_diffusion(self):
        return self.diffusion

    def get_schedule_sampler(self, name, maxt):
        return create_named_schedule_sampler(name, self.diffusion, maxt=maxt)
