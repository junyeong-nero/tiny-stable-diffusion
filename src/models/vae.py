"""VAE (Variational AutoEncoder) for Stable Diffusion 3 style latent space.

Implements an AutoencoderKL with:
- 64x64 input -> 8x8 latent (f8 compression)
- 16 latent channels (SD3 style)
- Reconstruction + KL divergence loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """
    ResNet Block: VAE의 기본 구성 요소
    (GroupNorm -> Swish -> Conv) x 2 구조에 Skip Connection 추가
    """

    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 입력과 출력 채널이 다르면 맞춰주기 위한 1x1 conv
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)  # Swish activation
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    """
    Self-Attention Block: 이미지의 전역적인 문맥(Global Context)을 파악
    보통 해상도가 가장 낮은 Bottleneck 구간에서 사용
    """

    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for Attention: (B, C, H, W) -> (B, C, H*W)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # (B, H*W, C)
        k = k.reshape(b, c, h * w)  # (B, C, H*W)
        v = v.reshape(b, c, h * w)  # (B, C, H*W)

        # Dot Product Attention
        attn = torch.bmm(q, k) * (c**-0.5)
        attn = F.softmax(attn, dim=2)

        # Weighted Sum
        h_ = torch.bmm(v, attn.permute(0, 2, 1))
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        in_channels=3,
        z_channels=16,
        dropout=0.0,
    ):
        super().__init__()

        # 기본 설정: SD3의 Latent Channel = 16
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # 초기 Convolution
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = 1
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        # Downsampling Layers
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                # 해상도가 낮아졌을 때(Bottleneck 근처) Attention 적용 여부 등을 결정할 수 있음 (여기서는 생략)

            down = nn.ModuleList()
            down.append(block)

            # 마지막 레벨이 아니면 Downsampling (Stride 2 Conv)
            if i_level != self.num_resolutions - 1:
                down.append(nn.Conv2d(block_out, block_out, kernel_size=3, stride=2, padding=1))
                curr_res = curr_res // 2

            self.down.append(down)

        # Middle Block (ResNet -> Attention -> ResNet)
        mid_ch = ch * ch_mult[-1]
        self.mid_block = nn.Sequential(
            ResnetBlock(mid_ch, mid_ch, dropout=dropout),
            AttnBlock(mid_ch),
            ResnetBlock(mid_ch, mid_ch, dropout=dropout),
        )

        # End: Norm -> Swish -> Conv (Output channel * 2 for Mean & LogVar)
        self.norm_out = nn.GroupNorm(32, mid_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(mid_ch, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level][0][i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level][1](h)  # Downsample

        h = self.mid_block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        out_channels=3,
        z_channels=16,
        dropout=0.0,
    ):
        super().__init__()

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[-1]
        curr_res = 2 ** (self.num_resolutions - 1)

        # z_channels(16) -> Conv -> Initial Channels
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle Block
        self.mid_block = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        # Upsampling Layers
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out

            up = nn.ModuleList()
            up.append(block)

            # 첫 번째 레벨이 아니면 Upsampling
            if i_level != 0:
                up.append(nn.Upsample(scale_factor=2.0, mode="nearest"))
                up.append(nn.Conv2d(block_out, block_out, kernel_size=3, stride=1, padding=1))
                curr_res = curr_res * 2

            self.up.append(up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[len(self.up) - 1 - i_level][0][i_block](h)
            if i_level != 0:
                h = self.up[len(self.up) - 1 - i_level][1](h)  # Upsample
                h = self.up[len(self.up) - 1 - i_level][2](h)  # Conv after upsample

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class AutoencoderKL(nn.Module):
    """
    SD3 Style VAE
    - z_channels: 16 (핵심)
    - channel multipliers: 보통 [1, 2, 4, 4] 사용 (f8 compression)
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        z_channels=16,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
    ):
        super().__init__()
        self.z_channels = z_channels

        self.encoder = Encoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
        )

        self.decoder = Decoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            out_channels=out_channels,
            z_channels=z_channels,
        )

        # VAE에서 Latent로 변환하기 전 1x1 Conv (선택적 사용, 보통 Encoder output이 곧바로 mean/logvar가 됨)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        moments = h
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, sample_posterior=True):
        mean, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        dec = self.decode(z)
        return dec, mean, logvar

    def training_loss(
        self,
        x: torch.Tensor,
        kl_weight: float = 1e-6,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate VAE training loss.

        Args:
            x: Input images (B, C, H, W) normalized to [-1, 1]
            kl_weight: Weight for KL divergence loss (beta-VAE style)

        Returns:
            Tuple of (total_loss, loss_dict with individual components and latent stats)
        """
        # Forward pass
        reconstruction, mean, logvar = self.forward(x, sample_posterior=True)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")

        # KL divergence loss
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        # Compute latent statistics for monitoring
        std = torch.exp(0.5 * logvar)

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            # Latent space statistics (should converge to mean≈0, std≈1)
            "latent_mean": mean.mean().item(),
            "latent_mean_std": mean.std().item(),
            "latent_std": std.mean().item(),
            "latent_std_std": std.std().item(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space (for diffusion training).

        Uses mean (deterministic) encoding for stable diffusion training.

        Args:
            x: Input images (B, C, H, W) normalized to [-1, 1]

        Returns:
            Latent tensor (B, z_channels, H/8, W/8)
        """
        mean, _ = self.encode(x)
        return mean

    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to images (for generation).

        Args:
            z: Latent tensor (B, z_channels, H/8, W/8)

        Returns:
            Decoded images (B, C, H, W) in [-1, 1]
        """
        return self.decode(z)


def create_vae(
    image_size: int = 64,
    z_channels: int = 16,
    ch: int = 64,
    ch_mult: tuple[int, ...] = (1, 2, 4, 4),
) -> AutoencoderKL:
    """Create a VAE model optimized for the given image size.

    Args:
        image_size: Input image size (64 recommended)
        z_channels: Latent channels (16 for SD3 style)
        ch: Base channel count (64 for lightweight)
        ch_mult: Channel multipliers for each resolution level

    Returns:
        AutoencoderKL model
    """
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        z_channels=z_channels,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=2,
    )


# ------------------------------------------------------------------------------
# 모델 테스트 코드 (64x64 해상도용)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 64x64 이미지 학습을 위한 경량화 설정
    model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        z_channels=16,  # SD3의 핵심인 16채널 Latent는 유지합니다.
        ch=64,  # GPU 메모리 절약을 위해 기본 채널을 128 -> 64로 줄였습니다.
        ch_mult=(1, 2, 4, 4),  # 압축 비율 유지: 64 -> 32 -> 16 -> 8 (f8 압축)
        num_res_blocks=2,
    ).to(device)

    # 1. 입력 이미지: 64x64 크기
    # Batch Size=1, RGB=3, H=64, W=64
    x = torch.randn(1, 3, 64, 64).to(device)

    print(f"1. Input Image shape: {x.shape}")
    # 출력: torch.Size([1, 3, 64, 64])

    # Forward Pass (인코딩 -> 디코딩)
    reconstruction, mean, logvar = model(x)

    # Latent Shape 확인
    with torch.no_grad():
        mu, _ = model.encode(x)

    # 64 / 8 = 8 이므로, 8x8 크기의 Feature Map이 나옵니다.
    print(f"2. Latent shape (Compressed): {mu.shape}")
    # 출력 예상: torch.Size([1, 16, 8, 8])
    # 설명: 공간(픽셀)은 8x8로 매우 작아졌지만, 깊이(채널)가 16이라 정보가 압축되어 있습니다.

    print(f"3. Reconstructed Image shape: {reconstruction.shape}")
    # 출력 예상: torch.Size([1, 3, 64, 64])

    # 파라미터 개수 확인 (모델이 얼마나 가벼운지 확인)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"4. Total Parameters: {num_params / 1_000_000:.2f} M")
