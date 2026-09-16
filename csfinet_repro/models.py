"""Paper-guided reconstruction; deviations and unspecified choices are versioned in docs.

This is an independent implementation, not a recovered version of the author's code.
Tensor convention: segmentation NCHW; Swin uses NHWC internally; survival NCDHW.
"""

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock
from torch.utils.checkpoint import checkpoint


@contextmanager
def preserve_batchnorm_buffers(module):
    """Recomputation must not count a second training observation in BatchNorm."""
    saved = [(child, child.running_mean.clone(), child.running_var.clone(), child.num_batches_tracked.clone())
             for child in module.modules() if isinstance(child, nn.modules.batchnorm._BatchNorm)
             and child.track_running_stats]
    try:
        yield
    finally:
        with torch.no_grad():
            for child, mean, variance, count in saved:
                child.running_mean.copy_(mean)
                child.running_var.copy_(variance)
                child.num_batches_tracked.copy_(count)


def checkpoint_call(module, *args):
    if getattr(module, "activation_checkpoint", False) and module.training and torch.is_grad_enabled():
        return checkpoint(module, *args, use_reentrant=False, preserve_rng_state=True,
                          context_fn=lambda: (nullcontext(), preserve_batchnorm_buffers(module)))
    return module(*args)


def enable_activation_checkpoint(model, enabled=True):
    for module in model.modules():
        module.activation_checkpoint = enabled
    return model


def conv_block(in_channels, out_channels, double=True):
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1),
              nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False)]
    if double:
        layers += [nn.Conv2d(out_channels, out_channels, 3, padding=1),
                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False)]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([conv_block(a, b) for a, b in zip((4, *channels[:-1]), channels)])
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, x):
        features = []
        for index, block in enumerate(self.blocks):
            x = checkpoint_call(block, self.pool(x) if index else x)
            features.append(x)
        return features


def warp(feature, flow):
    """Backward sample at (x + dx, y + dy); flow is measured in feature pixels."""
    n, _, h, w = feature.shape
    if flow.shape != (n, 2, h, w) or min(h, w) < 2:
        raise ValueError("Warp requires N2HW flow and spatial dimensions >= 2")
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=feature.device, dtype=torch.float32),
                           torch.linspace(-1, 1, w, device=feature.device, dtype=torch.float32), indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    scale = flow.new_tensor([2 / (w - 1), 2 / (h - 1)], dtype=torch.float32)
    grid = grid + flow.permute(0, 2, 3, 1).float() * scale
    return F.grid_sample(feature.float(), grid, mode="bilinear", padding_mode="zeros",
                         align_corners=True).to(feature.dtype)


class FIU(nn.Module):
    """R Eq.8-10: shared flow, warp both inputs, add, then CBR."""
    def __init__(self, previous_channels, channels):
        super().__init__()
        self.project = nn.Conv2d(previous_channels, channels, 1)
        self.flow = nn.Conv2d(channels * 2, 2, 3, padding=1)
        self.fuse = conv_block(channels, channels, double=False)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, previous, current):
        previous = F.interpolate(self.project(previous), size=current.shape[-2:], mode="bilinear", align_corners=True)
        flow = self.flow(torch.cat((previous, current), dim=1))
        return self.fuse(warp(previous, flow) + warp(current, flow))


def partition(feature, window):
    n, h, w, c = feature.shape
    if h % window or w % window:
        raise ValueError("PIU spatial dimensions must be divisible by their window size")
    return feature.reshape(n, h // window, window, w // window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(
        n, h // window, w // window, window * window, c)


def restore(tokens, window):
    n, h, w, count, c = tokens.shape
    if count != window * window:
        raise ValueError("PIU token count does not match window size")
    return tokens.reshape(n, h, w, window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(n, h * window, w * window, c)


class PIU(nn.Module):
    def __init__(self, channels, dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.project = nn.ModuleList([nn.Linear(c, dim) for c in channels])
        self.reverse = nn.ModuleList([nn.Linear(dim, c) for c in channels])
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, features):
        grouped = [partition(project(x.permute(0, 2, 3, 1)), window)
                   for x, project, window in zip(features, self.project, (4, 2, 1))]
        combined = torch.cat(grouped, dim=-2)
        shape = combined.shape
        tokens = combined.reshape(-1, 21, shape[-1])
        norm = self.norm1(tokens)
        tokens = tokens + self.attention(norm, norm, norm, need_weights=False)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        parts = tokens.reshape(shape).split((16, 4, 1), dim=-2)
        return [project(restore(part, window)).permute(0, 3, 1, 2).contiguous()
                for part, project, window in zip(parts, self.reverse, (4, 2, 1))]


class CSFINet(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 1024), depths=(2, 2, 6), heads=(4, 8, 16),
                 window_size=7, piu_dim=512, piu_heads=8, piu_dropout=0.1, drop_path=0.1):
        super().__init__()
        self.encoder = Encoder(channels)
        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        stages, offset = [], 0
        for c, depth, head in zip(channels[2:], depths, heads):
            stages.append(nn.Sequential(*[
                SwinTransformerBlock(c, head, [window_size, window_size],
                                     [0, 0] if i % 2 == 0 else [window_size // 2] * 2,
                                     mlp_ratio=4, dropout=0.0, attention_dropout=0.0,
                                     stochastic_depth_prob=rates[offset + i]) for i in range(depth)]))
            offset += depth
        self.swin = nn.ModuleList(stages)
        self.merge = nn.ModuleList([PatchMerging(channels[2]), PatchMerging(channels[3])])
        self.piu = PIU(channels[2:], piu_dim, piu_heads, piu_dropout)
        self.fusion = nn.ModuleList([nn.Conv2d(c * 2, c, 1) for c in channels[2:]])
        reverse = tuple(reversed(channels))
        self.fiu = nn.ModuleList([FIU(a, b) for a, b in zip((reverse[0], *reverse[:-1]), reverse)])
        self.head = nn.Conv2d(channels[0], 4, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 4 or x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("CSFINet expects N4HW with H,W divisible by 16 (including 240)")
        cnn = self.encoder(x)
        features = []
        feature = cnn[2].permute(0, 2, 3, 1)
        for i, stage in enumerate(self.swin):
            if i:
                feature = self.merge[i - 1](feature) + cnn[i + 2].permute(0, 2, 3, 1)
            feature = checkpoint_call(stage, feature)
            features.append(feature.permute(0, 3, 1, 2).contiguous())
        interacted = checkpoint_call(self.piu, features)
        fused = cnn[:2] + [project(torch.cat((local, global_), dim=1))
                           for local, global_, project in zip(cnn[2:], interacted, self.fusion)]
        previous = torch.zeros_like(fused[-1])
        for unit, current in zip(self.fiu, reversed(fused)):
            previous = checkpoint_call(unit, previous, current)
        return self.head(previous)


class UNet(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoder = Encoder(channels)
        reverse = tuple(reversed(channels))
        self.up = nn.ModuleList([nn.ConvTranspose2d(a, b, 2, stride=2) for a, b in zip(reverse, reverse[1:])])
        self.decoder = nn.ModuleList([conv_block(c * 2, c) for c in reverse[1:]])
        self.head = nn.Conv2d(channels[0], 4, 1)

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        for up, block, skip in zip(self.up, self.decoder, reversed(features[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = checkpoint_call(block, torch.cat((x, skip), dim=1))
        return self.head(x)


class SurvivalNet(nn.Module):
    """Explicit reconstruction choices for manuscript's incompletely specified 3D CNN."""
    def __init__(self, clinical_features=0, channels=(16, 32, 64), dropout=0.3, in_channels=1):
        super().__init__()
        if clinical_features not in (0, 1, 3, 4):
            raise ValueError("Use image-only, age, resection one-hot, or both")
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("Survival input channels must be a positive integer")
        self.in_channels = in_channels
        blocks = []
        for a, b in zip((in_channels, *channels[:-1]), channels):
            blocks += [nn.Conv3d(a, b, 3, padding=1), nn.BatchNorm3d(b), nn.ReLU(), nn.MaxPool3d(2)]
        self.image = nn.Sequential(*blocks, nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(channels[-1], 128), nn.ReLU())
        self.clinical_features = clinical_features
        if clinical_features:
            self.clinical = nn.Sequential(nn.Linear(clinical_features, 32), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(32, 16), nn.ReLU())
        widths = (144, 64, 32, 1) if clinical_features else (128, 128, 64, 32, 1)
        layers = []
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(a, b))
            if b != 1:
                layers.append(nn.ReLU())
                if i == 0:
                    layers.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*layers, nn.Softplus())

    def forward(self, image, clinical=None):
        image = self.image(image)
        if self.clinical_features:
            if clinical is None or clinical.shape != (image.shape[0], self.clinical_features):
                raise ValueError("Clinical feature shape mismatch")
            image = torch.cat((image, self.clinical(clinical)), dim=1)
        elif clinical is not None:
            raise ValueError("Image-only model must not receive clinical inputs")
        return self.head(image).squeeze(-1)


def build_segmentation(name, config):
    settings = config["segmentation"]
    if name == "unet":
        return enable_activation_checkpoint(UNet(tuple(settings["channels"])), settings.get("activation_checkpoint", False))
    if name != "csfinet":
        raise ValueError(f"Unknown segmentation model: {name}")
    model = CSFINet(channels=tuple(settings["channels"]), depths=tuple(settings["swin_depths"]),
                   heads=tuple(settings["swin_heads"]), window_size=settings["window_size"],
                   piu_dim=settings["piu_dim"], piu_heads=settings["piu_heads"],
                   piu_dropout=settings["piu_dropout"], drop_path=settings["drop_path"])
    return enable_activation_checkpoint(model, settings.get("activation_checkpoint", False))
