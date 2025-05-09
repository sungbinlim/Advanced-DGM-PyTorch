import os, yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vision_utils
from inspect import isfunction
from data import reverse_transform

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

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config

def plot_batch(ax, batch, title=None, **kwargs):
    reverse_transform_fn = reverse_transform()
    batch = torch.tensor(np.array([reverse_transform_fn(img) for img in batch]))
    imgs = vision_utils.make_grid(batch, padding=2, normalize=True)
    imgs = np.moveaxis(imgs.numpy(), 0, -1)
    ax.set_axis_off()
    if title is not None: ax.set_title(title)
    return ax.imshow(imgs, **kwargs)

def save_images(batch, title, file_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_batch(ax, batch, title)

    if file_path is None:
        file_path = os.path.join(os.getcwd(), 'samples/')
        file_name = file_path + title + '.png'
    elif os.path.exists(file_path):
        file_name = os.path.join(file_path, title + '.png')
    else:
        os.makedirs(file_path)
        file_name = os.path.join(file_path, title + '.png')

    plt.savefig(fname=file_name, dpi=300)
    print(f"Image sampling is Done at {file_name}.")