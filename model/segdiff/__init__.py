## https://github.com/tomeramit/SegDiff/blob/main/image_train_diff_medical.py

### Descarded because of the 2D nature of the model

from .script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    # args_to_dict,
)

from .resample import create_named_schedule_sampler


class SegDiffModel:
    def __init__(self, args=None):
        if args is None:
            args = model_and_diffusion_defaults()
        else:
            default = model_and_diffusion_defaults()

            # remove unexpected keys
            for key in args:
                if key not in default:
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

    def get_schedule_sampler(self, name):
        return create_named_schedule_sampler(name, self.diffusion)
