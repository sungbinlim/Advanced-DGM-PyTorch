import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vision_utils
from inspect import isfunction


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def plot_batch(ax, batch, title=None, **kwargs):
    imgs = vision_utils.make_grid(batch, padding=2, normalize=True)
    imgs = np.moveaxis(imgs.numpy(), 0, -1)
    ax.set_axis_off()
    if title is not None: ax.set_title(title)
    return ax.imshow(imgs, **kwargs)

def save_images(batch, title):
    batch_size = batch.shape[0]
    row = int(np.sqrt(batch_size))
    col = batch_size // row
    fig = plt.figure(figsize=(row, col))
    ax = fig.add_subplot(111)
    plot_batch(ax, batch, title)
    file_name = title + '_generated images.png'
    plt.savefig(fname=file_name)