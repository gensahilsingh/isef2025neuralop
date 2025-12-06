from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.fft as fft

from .fno import FNO


class SpectralBandEmphasis(nn.Module):
    """Apply learnable per-channel emphasis across broad spectral bands.

    This is a light-weight, robust alternative to per-mode learnable weights.
    It computes the real-FFT of the activation, multiplies frequency bands by
    per-channel learnable gains, and transforms back.
    """

    def __init__(self, channels: int, n_bands: int = 3, fft_norm: str = "forward"):
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        # one gain per channel per band
        self.gains = nn.Parameter(torch.zeros(channels, n_bands))
        self.fft_norm = fft_norm

    def forward(self, x: torch.Tensor):
        # x: (B, C, d1, d2, ...)
        batch, channels = x.shape[:2]
        assert channels == self.channels

        fft_dims = list(range(- (x.ndim - 2), 0))
        # real FFT along spatial dims
        Xf = torch.fft.rfftn(x, dim=fft_dims, norm=self.fft_norm)

        # build radial frequency grid (normalized)
        mode_sizes = [x.shape[2 + i] for i in range(len(fft_dims))]
        grids = []
        for i, n in enumerate(mode_sizes):
            if i == len(mode_sizes) - 1:
                freqs = torch.fft.rfftfreq(n)
            else:
                freqs = torch.fft.fftfreq(n)
            grids.append(freqs.to(x.device))

        # produce a meshgrid of squared radii
        mesh = torch.meshgrid(*grids, indexing="ij")
        rad2 = torch.zeros_like(mesh[0])
        for g in mesh:
            rad2 = rad2 + (g ** 2)
        rad = torch.sqrt(rad2)
        rad = rad / (rad.max() + 1e-12)

        # define band edges uniformly in [0, 1]
        edges = torch.linspace(0.0, 1.0, steps=self.n_bands + 1, device=x.device)
        band_masks = []
        for b in range(self.n_bands):
            m = ((rad >= edges[b]) & (rad < edges[b + 1])).to(Xf.dtype)
            # expand to include batch and channel dims when applying
            band_masks.append(m)

        # compute weighted mask per channel: shape (C, *fft_shape)
        # first stack masks -> (n_bands, *fft_shape)
        stacked = torch.stack(band_masks, dim=0)
        # gains: (C, n_bands) -> (C, n_bands, 1,1,...)
        expand_dims = [1] * (stacked.ndim - 1)
        gains = self.gains.view(self.channels, self.n_bands, *expand_dims)
        # channel-wise multiplier over FFT domain: (C, *fft_shape)
        channel_multiplier = (gains * stacked.unsqueeze(0)).sum(dim=1)

        # apply multiplier: Xf shape (B, C, *fft_shape)
        # ensure types align (Xf is complex)
        multiplier = 1.0 + channel_multiplier.unsqueeze(0).to(Xf.real.dtype)
        Xf = Xf * multiplier

        # inverse transform: keep spatial size
        x_out = torch.fft.irfftn(Xf, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        return x_out


class FNOCardiac(FNO, name="FNOCardiac"):
    """Fourier Neural Operator variant tuned for cardiac MCG/MIoG signals.

    Features added on top of standard FNO:
    - Spectral band emphasis module (learnable gains per channel/band)
    - Optional small classification head for disease detection
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        freq_bands: int = 3,
        classifier: bool = False,
        n_classes: int = 2,
        **kwargs,
    ):
        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **kwargs,
        )

        self.spectral_emphasis = SpectralBandEmphasis(hidden_channels, n_bands=freq_bands)

        self.classifier = classifier
        if classifier:
            # global pooling + linear head
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1) if self.n_dim == 1 else nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels, max(2, n_classes)),
            )

    def forward(self, x, output_shape=None, return_features=False, **kwargs):
        # copy of FNO.forward with spectral emphasis inserted before projection
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # spectral emphasis in latent space
        x = self.spectral_emphasis(x)

        features = x
        x = self.projection(x)

        if self.classifier:
            # classification head uses pooled features
            # expect features shape (B, C, ...)
            if self.n_dim == 1:
                cls_in = features
            elif self.n_dim == 2:
                cls_in = features
            else:
                # fallback: global average over spatial dims
                cls_in = features
            logits = self.class_head(cls_in)
            if return_features:
                return x, logits, features
            return x, logits

        if return_features:
            return x, features

        return x
