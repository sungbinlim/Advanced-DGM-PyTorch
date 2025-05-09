import torch
import torch.nn as nn
import lightning as pl
from torch.nn import functional as F
from .layers import ResnetBlock, Residual, PreNorm, LinearAttention, Upsample, Downsample
from functools import partial
from util import default
from copy import copy

class VQVAE(pl.LightningModule):
    def __init__(
        self, dim, encoder, decoder, quantize, dim_mults=(1, 2, 4, 8),
        in_channel=3, 
        embed_dim=64, # dimension of latent vector: d
        n_embed=512, # number of embedding vectors: K
        loss_fn=None,
        optimizer=None,
        lr=1e-3
    ):
        super().__init__()
        self.encoder = encoder(dim, dim_mults, in_channel)
        decoder_dims, encoder_last_dim = self.extract_decoder_dims(self.encoder.dims, embed_dim, in_channel)
        self.quantize_conv = nn.Conv2d(encoder_last_dim, embed_dim, 1)
        self.quantize = quantize(embed_dim, n_embed)
        self.decoder = decoder(decoder_dims)
        self.loss_fn = loss_fn if loss_fn else F.mse_loss
        self.optimizer = optimizer if optimizer else torch.optim.Adam
        self.lr = lr

    def training_step(self, batch, batch_idx):
        
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch    

        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        z = self.quantize_conv(z).transpose(1, 3)
        z, diff, _ = self.quantize(z)
        x_recon = self.decoder(z.permute(0, 3, 1, 2))
        recon_loss = self.loss_fn(x_recon, x)
        loss = recon_loss + diff
        
        self.log("train/loss", recon_loss, prog_bar=True, sync_dist=True)
        self.log("train/embed_loss", diff, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch

        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        z = self.quantize_conv(z).transpose(1, 3)
        z, diff, _ = self.quantize(z)
        x_recon = self.decoder(z.permute(0, 3, 1, 2))
        val_loss = self.loss_fn(x_recon, x)
        self.log("val/recon_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log('val/embed_loss', diff, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
    
    def extract_decoder_dims(self, encoder_dims, embed_dim, in_channels):
        decoder_dims = copy(encoder_dims)
        encoder_last_dim = decoder_dims.pop()
        decoder_dims.append(embed_dim)
        decoder_dims.reverse()
        decoder_dims.pop()
        decoder_dims.append(in_channels)

        return decoder_dims, encoder_last_dim
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        z = self.quantize_conv(z).transpose(1, 3)
        z = self.quantize(z)[0]
        x_recon = self.decoder(z.permute(0, 3, 1, 2))
        return x_recon

class Encoder(nn.Module):
    """
    Encoder for VQ-VAE
    dim: int
        Dimension of the input
    dim_mults: Tuple[int]
        Multipliers for the dimensions
    channels: int
        Number of channels
    self_condition: bool
        Whether to condition on the input
    resnet_block_groups: int
        Number of groups in the ResNet block
    """
    def __init__(self, dim, dim_mults=(1, 2, 4, 8), channels=3, 
                 self_condition=False,
                 resnet_block_groups=4,
                 ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = self.channels * (2 if self_condition else 1)

        self.init_conv = nn.Conv2d(input_channels, dim, 1, padding=0)
        self.dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(self.dims[:-1], self.dims[1:]))

        block_class = partial(ResnetBlock, groups=resnet_block_groups)
        num_resolutions = len(in_out)
        self.blocks = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_in, time_emb_dim=None),
                        block_class(dim_in, dim_in, time_emb_dim=None),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

    def forward(self, x, x_self_cond=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)

        for block1, block2, attn, downsample in self.blocks:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = downsample(x)

        return x

class Decoder(nn.Module):
    def __init__(self, dims, resnet_block_groups=4):
        super().__init__()

        in_out = list(zip(dims[:-1], dims[1:]))
        block_class = partial(ResnetBlock, groups=resnet_block_groups)
        num_resolutions = len(in_out)
        self.blocks = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_in, time_emb_dim=None),
                        block_class(dim_in, dim_in, time_emb_dim=None),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

    def forward(self, x):
        
        for block1, block2, attn, upsample in self.blocks:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = upsample(x)

        return x

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):
        flatten = x.reshape(-1, self.dim) # (B, h, w, dim) -> (B*h*w, dim)
        
        # compute L2 distance between x and embed
        dist = flatten.pow(2).sum(1, keepdim=True) - 2*flatten@self.embed + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # update codebook using EMA
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

if __name__ == "__main__":
    from copy import copy

    encoder = Encoder(dim=8,  
                      dim_mults=(1, 2, 4, 8), 
                      channels=3, 
                      self_condition=False, 
                      resnet_block_groups=4)
    x = torch.randn(size=(5, 3, 32, 32))
    y = encoder(x) # (B, 3, 32, 32) -> (B, 64, 4, 4)
    print(f"shape of Encoder output: {y.shape}")

    embed_dim = 64
    n_embed = 512

    decoder_dims = copy(encoder.dims)
    encoder_last_dim = decoder_dims.pop()
    decoder_dims.append(embed_dim) # embed_dim=64
    decoder_dims.reverse()
    decoder_dims.pop()
    decoder_dims.append(encoder.channels)

    quantize_conv = nn.Conv2d(encoder_last_dim, embed_dim, 1) # (B, 64, 4, 4) -> (B, 64, 4, 4)
    quantizer = Quantize(embed_dim, n_embed)
    
    z = quantize_conv(y).transpose(1, 3) # (B, 64, 4, 4) -> (B, 4, 4, 64)
    z, diff, embed_ind = quantizer(z) # (B, 4, 4, 64), 1, (B, 4, 4)
    z = z.permute(0, 3, 1, 2) # (B, 4, 4, 64) -> (B, 64, 4, 4)

    print(f"embed_ind: {embed_ind}")    
    print(f"shape of latent: {z.shape}, diff: {diff}, shape of embed_ind: {embed_ind.shape}")

    decoder = Decoder(dims=decoder_dims) # (B, 64, 4, 4) -> (B, 3, 32, 32) 

    x_hat = decoder(z)
    print(x_hat.shape)

    VQVAE = VQVAE(dim=8, 
                  dim_mults=(1, 2, 4, 8), 
                  in_channel=3, 
                  embed_dim=64, 
                  n_embed=512, 
                  decay=0.99)
    
    vq_x_hat, vq_diff, vq_embed_ind = VQVAE(x)
    print(f"shape of recon x: {vq_x_hat.shape}, diff: {vq_diff}, shape of embed_ind: {vq_embed_ind.shape}")
