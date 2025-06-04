import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample, ResidualUnit
from monai.networks.layers.factories import Act


class ConditionEncoder(nn.Module):
    """
    Image encoder to extract multi-scale features for conditioning.
    Returns a list of feature maps at each resolution.
    """

    def __init__(self, in_channels=1, base_channels=32, num_levels=4, act="RELU"):
        super().__init__()
        self.num_levels = num_levels
        self.enc_blocks = nn.ModuleList()
        ch = in_channels
        for i in range(num_levels):
            out_ch = base_channels * (2**i)
            self.enc_blocks.append(
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=ch,
                    out_channels=out_ch,
                    act=act,
                    norm="batch",
                )
            )
            ch = out_ch
            if i < num_levels - 1:
                self.enc_blocks.append(nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        feats = []
        for block in self.enc_blocks:
            x = block(x)
            # collect features after each residual block (before pooling)
            if isinstance(block, ResidualUnit):
                feats.append(x)
        return feats


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding as in diffusion models.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=device)
            * -(torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DenoiserUNet(nn.Module):
    """
    Conditional UNet for denoising. Integrates time and image condition features.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        cond_channels=None,
        num_levels=4,
        time_emb_dim=128,
        act="RELU",
    ):
        super().__init__()
        # time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            Act[act](),
        )
        # down blocks
        self.down_blocks = nn.ModuleList()
        ch = in_channels
        for i in range(num_levels):
            out_ch = base_channels * (2**i)
            # convolution block takes concat of x and cond
            in_ch = ch + (cond_channels[i] if cond_channels is not None else 0)
            self.down_blocks.append(
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    act=act,
                    norm="batch",
                )
            )
            ch = out_ch
            if i < num_levels - 1:
                self.down_blocks.append(nn.MaxPool2d(2))
        # up blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_levels - 1)):
            in_ch = (
                ch
                + (base_channels * (2**i))
                + (cond_channels[i] if cond_channels is not None else 0)
            )
            out_ch = base_channels * (2**i)
            self.up_blocks.append(
                nn.Sequential(
                    UpSample(spatial_dims=2, scale_factor=2, mode="nearest"),
                    ResidualUnit(
                        spatial_dims=2,
                        in_channels=in_ch + time_emb_dim,
                        out_channels=out_ch,
                        act=act,
                        norm="batch",
                    ),
                )
            )
            ch = out_ch
        # final conv to segmentation mask logits
        self.final_conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x, t, cond_feats=None):
        # time embed
        t_emb = self.time_mlp(t)
        # down path
        skip_feats = []
        for block in self.down_blocks:
            if isinstance(block, ResidualUnit):
                # optionally concat condition feature
                if cond_feats:
                    feat = cond_feats[len(skip_feats)]
                    x = torch.cat([x, feat], dim=1)
                x = block(x)
                skip_feats.append(x)
            else:
                x = block(x)
        # up path
        for idx, block in enumerate(self.up_blocks):
            # upsample + residual
            x = block[0](x)  # UpSample
            # gather skip
            skip = skip_feats[-(idx + 2)]
            x = torch.cat([x, skip], dim=1)
            # concat condition
            if cond_feats:
                cond = cond_feats[-(idx + 2)]
                x = torch.cat([x, cond], dim=1)
            # add time embedding
            batch, _, h, w = x.shape
            t_emb_img = t_emb[:, :, None, None].expand(batch, -1, h, w)
            x = torch.cat([x, t_emb_img], dim=1)
            x = block[1](x)
        # final
        return self.final_conv(x)


class DiffusionSegModel(nn.Module):
    """
    Wrapper model: uses ConditionEncoder and DenoiserUNet.
    """

    def __init__(self, image_encoder: nn.Module, denoiser: nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.denoiser = denoiser

    def forward(self, x_noisy, t, image):
        # image: conditioning input
        cond_feats = self.image_encoder(image)
        seg_logits = self.denoiser(x_noisy, t, cond_feats)
        return seg_logits


# Example instantiation:
# image_encoder = ConditionEncoder(in_channels=1, base_channels=32)
# cond_channels = [32 * (2**i) for i in range(4)]
# denoiser = DenoiserUNet(in_channels=1, base_channels=64, cond_channels=cond_channels)
# model = DiffusionSegModel(image_encoder, denoiser)

# Training can be done with MONAI SupervisedTrainer or a custom Ignite engine:
# from monai.engines import SupervisedEvaluator, SupervisedTrainer
# trainer = SupervisedTrainer(
#     device=device,
#     max_epochs=100,
#     train_data_loader=train_loader,
#     network=model,
#     optimizer=optimizer,
#     loss_function=nn.BCEWithLogitsLoss(),
#     key_train_metric=None,
#     train_handlers=[...],
# )
# trainer.run()
