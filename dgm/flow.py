import torch
import torch.nn as nn
import lightning as pl
from tqdm.auto import tqdm
from torch.nn import functional as F
from torch.distributions import Normal
from functools import partial
from util import default, exists
from .layers import Residual, ResnetBlock, PreNorm, LinearAttention, Attention, Downsample, Upsample, SinusoidalPositionEmbeddings
from flow_matching.solver import ODESolver
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

class FlowModel(pl.LightningModule):
    def __init__(
            self, dim, flow, time_dim=None,
            dim_mults=(1, 2, 4, 8),
            in_channels=3, 
            loss_fn=None,
            optimizer=None,
            lr=1e-3,
            scheduler=None,           
    ):
        super().__init__()
        self.flow = flow(dim, time_dim, dim_mults=dim_mults, in_channels=in_channels)
        self.loss_fn = loss_fn if loss_fn else F.mse_loss
        self.optimizer = optimizer if optimizer else torch.optim.Adam
        self.lr = lr
        self.scheduler = scheduler() if scheduler else CondOTScheduler()
        self.path = AffineProbPath(scheduler=self.scheduler)

    def forward(self, x, t):
        return self.flow(x=x, t=t)

    def training_step(self, batch, batch_idx):

        if isinstance(batch, list):
            x = batch[0] # CelebA
        else:
            x = batch

        if x.dim() == 3: # MNIST
            x = x.unsqueeze(1)

        if self.data_shape is None:
            self.data_shape = x.shape[1:]

        time = torch.rand((x.shape[0], ), device=self.device)
        noise = torch.randn_like(x)
        path_sample = self.path.sample(t=time, x_0=noise, x_1=x)
        target = path_sample.dx_t
        output = self.flow(path_sample.x_t, time)
        loss = self.loss_fn(output, target)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch

        if x.dim() == 3:
            x = x.unsqueeze(1)

        time = torch.rand((x.shape[0], ), device=self.device)
        noise = torch.randn_like(x)
        path_sample = self.path.sample(t=time, x_0=noise, x_1=x)
        target = path_sample.dx_t
        output = self.flow(path_sample.x_t, time)
        loss = self.loss_fn(output, target)

        self.log("valid/loss", loss, prog_bar=True, sync_dist=True)

        return loss   
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
    
    @torch.no_grad()
    def sampling_step(self, batch, time, dt, modified=True):

        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch

        if x.dim() == 3:
            x = x.unsqueeze(1)

        dx = self.flow(x, time) * dt

        if modified: # Heun's method (2nd-order Runge-Kutta)
            dx += self.flow(x + dx, time + dt) * dt
            x += 0.5 * dx
        else: # Euler's method
            x += dx

        return x
    
    @torch.no_grad()
    def sampling(self, sample_size, num_step=100, modified=True):
        self.flow.eval()
        x = torch.randn((sample_size, *self.flow.data_shape), device=self.device)
        t = torch.zeros((sample_size, ), device=self.device)
        delta_t = 1 / num_step

        for i in tqdm(range(num_step), desc='sampling loop time step', total=num_step):
            x = self.sampling_step(x, t, delta_t, modified)
            t += delta_t
        return x
    
    @torch.no_grad()
    def score(self, x, t=None):
         """Esimate the score function using the model."""
         if t is None:
             t = torch.tensor([0.99] * x.shape[0], device=self.device)
         alpha_t = self.scheduler(t).alpha_t.unsqueeze(1)
         sigma_t = self.scheduler(t).sigma_t.unsqueeze(1)
         d_alpha_t = self.scheduler(t).d_alpha_t.unsqueeze(1)
         d_sigma_t = self.scheduler(t).d_sigma_t.unsqueeze(1)
         coefficient = (alpha_t / (d_sigma_t * sigma_t * alpha_t - d_alpha_t * (sigma_t**2)))
         self.flow.eval()
         score = coefficient * ((d_alpha_t / alpha_t) * x  - self.flow(x, t))
         
         return score

    def compute_likelihood(self, x, step_size=0.01):
        time_grid = torch.tensor([1.0, 0.0]).to(x.device)
        log_p0 = Normal(
            loc=torch.zeros_like(x)[1:],
            scale=torch.ones_like(x)[1:]).log_prob
        solver = ODESolver(velocity_model=self.flow)
        sol, likelihood = solver.compute_likelihood(x_1=x,
                                                    log_p0=log_p0,
                                                    step_size=step_size,
                                                    time_grid=time_grid,
                                                    method='midpoint')
        result = {'likelihood': likelihood, 'solution': sol}
        return result

# UNet for DDPM and Flow Models
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        time_dim=None,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        self_condition=False,
        in_channels=3,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.data_shape = None
        self.channels = in_channels
        self.self_condition = self_condition
        input_channels = self.channels * (2 if self_condition else 1)
        time_dim = default(time_dim, 4*dim)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)) if exists(time_dim) else None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, in_channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, t=None, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        if self.data_shape is None:
            self.data_shape = x.shape[1:]

        x = self.init_conv(x)
        r = x.clone()

        if exists(self.time_mlp) and exists(t):
            if len(t.shape) == 0:
                t = t[None]
            t = self.time_mlp(t)
        else:
            t = None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

if __name__ == "__main__":
    x = torch.randn(size=(5, 3, 32, 32)).to('cuda')
    t = torch.rand(size=(5, )).to('cuda')
    flow_model = FlowModel(dim=8, flow=Unet, time_dim=64).to('cuda')
    y = flow_model(x, t)
    print(f"output shape: {y.shape}")

    x0 = torch.randn(size=(1, 3, 32, 32)).to('cuda')
    likelihood_result = flow_model.compute_likelihood(x0, step_size=0.01)
    print(f"Likelihood: {likelihood_result['likelihood']}")
    print(f"Solution: {likelihood_result['solution']}")