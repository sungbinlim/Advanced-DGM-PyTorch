import os
import argparse
import torch.nn as nn
from dgm.vqvae import VQVAE, Encoder, Decoder, Quantize
from dgm.flow import FlowModel, Unet, AffineProbPath
from data import get_loader
from util import *
import lightning as pl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Deep Generative Model')
    parser.add_argument('--dgm', type=str, default='vqvae', required=True, choices=['vqvae', 'ddpm'], help = 'path of config file')
    parser.add_argument('--data', type=str, default='mnist', required=True, choices=['mnist', 'celeba', 'afhq', 'cifar'], help = 'path of config file')
    parser.add_argument('--ckpt_name', type=str, default=None, required=True, help = 'path of checkpoint file')

    args = parser.parse_args()
    cfg_path = os.path.join(*[os.getcwd(), 'config', f"{args.dgm}_{args.data}.yaml"])

    if not os.path.exists(cfg_path):
        raise ValueError(f"{cfg_path} does not exist. Please check your configuration file.")
    else:
        setting = load_config(cfg_path)
        paths = setting['path']
        config = setting['config']

    # Data loader for Visual data
    dataloader = get_loader(train_valid_split=0.8, num_workers=8, **setting)
    val_loader = dataloader['valid']

    if config['DGM'] == 'FlowModel':
        model = FlowModel.load_from_checkpoint(args.ckpt_name,
                                        strict=False,
                                        dim=config['init_Channel'],
                                        flow=Unet,
                                        time_dim=config['Time_dim'],
                                        dim_mults=tuple(config['Channel_mults']),
                                        in_channels=config['in_Channel'],
                                        lr=config['Learning rate'],
                                        scheduler=config['Scheduler'],
                                        )
    elif config['DGM'] == 'VQ-VAE':
        model = VQVAE.load_from_checkpoint(args.ckpt_name,
                                        strict=False,
                                        dim=config['init_Channel'], 
                                        encoder=Encoder,
                                        decoder=Decoder,
                                        quantize=Quantize,
                                        dim_mults=tuple(config['Channel_mults']),
                                        embed_dim=config['Embed_dim'],
                                        n_embed=config['Number_embed'],
                                       )

    print("Image Sampling...")
    sample_batch = next(iter(val_loader))
    if isinstance(sample_batch, list):
        x = sample_batch[0][0:8].to('cuda')
    else:
        x = sample_batch[0:8].to('cuda')
    
    if config['DGM'] == 'VQ-VAE':
        result = model(x)
    elif config['DGM'] == 'FlowModel':
        result = model.sampling(sample_size=8, resolution=config['Resolution'], num_step=config['Time steps'])
    
    stacked_imgs = torch.vstack([x.cpu(), result.cpu()])
    sample_path = paths['sample_path']
    save_images(stacked_imgs, f"{config['Data']}_recon_imgs", file_path=sample_path)
    