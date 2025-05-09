import os
import argparse
import torch.nn as nn
from dgm.vqvae import VQVAE, Encoder, Decoder, Quantize
from dgm.flow import FlowModel, Unet
from data import get_loader
from util import *
from time import time
import lightning as pl
from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Deep Generative Model')
    parser.add_argument('--dgm', type=str, default='vqvae', required=True, choices=['vqvae', 'fm'], help = 'path of config file')
    parser.add_argument('--data', type=str, default='mnist', required=True, choices=['mnist', 'celeba', 'afhq', 'cifar'], help = 'path of config file')
    parser.add_argument('--gpu', type=int, default=None, choices=[num for num in range(torch.cuda.device_count())], help = 'device number')
    parser.add_argument('--save', type=bool, default=False, help = 'save the model or not')

    args = parser.parse_args()
    cfg_path = os.path.join(*[os.getcwd(), 'config', f"{args.dgm}_{args.data}.yaml"])

    if not os.path.exists(cfg_path):
        raise ValueError(f"{cfg_path} does not exist. Please check your configuration file.")
    else:
        setting = load_config(cfg_path)
        paths = setting['path']
        config = setting['config']

    if (config['Device'] is None) or (config['Device'] != 'auto'):
        if args.gpu is not None:
            config['Device'] = args.gpu
            print(f"Device is set to {config['Device']}.")
        elif args.gpu is None:
            config['Device'] = 'auto'
        elif config['Device'] not in [num for num in range(torch.cuda.device_count())]:
            raise ValueError(f"GPU number {config['Device']} is not available. Please check your GPU number.")

    import wandb
    wandb.login()    
    identifier = str(time())[5:10]
    wandb_logger = WandbLogger(project=f"DGM-PyTorch-{config['DGM']}-{config['Data']}",
                               name=f"{config['DGM']}-{config['Data']}-{identifier}")

    # Data loader for Visual data
    dataloader = get_loader(train_valid_split=0.8, num_workers=32, **setting)
    train_loader = dataloader['train']
    val_loader = dataloader['valid']

    if config['DGM'] == 'FlowModel':
        model = FlowModel(dim=config['init_Channel'],
                          flow=Unet,
                          time_dim=config['Time_dim'],
                          dim_mults=tuple(config['Channel_mults']),
                          in_channels=config['in_Channel'],
                          loss_fn=nn.MSELoss(),
                          lr=config['Learning rate'],
                          scheduler=config['Scheduler'],
                          )
    elif config['DGM'] == 'VQ-VAE':
        model = VQVAE(dim=config['init_Channel'], 
                      encoder=Encoder,
                      decoder=Decoder,
                      quantize=Quantize,
                      dim_mults=tuple(config['Channel_mults']),
                      embed_dim=config['Embed_dim'],
                      n_embed=config['Number_embed'],
                      in_channel=config['in_Channel'],
                      loss_fn=nn.MSELoss(),
                      lr=config['Learning rate'],
                      )
    
    trainer = pl.Trainer(devices=config['Device'], 
                         max_epochs=config['Epochs'], 
                         logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)

    sample_batch = next(iter(val_loader))
    
    if isinstance(sample_batch, list):
        x = sample_batch[0][0:8].to('cuda:0')
    else:
        x = sample_batch[0:8].to('cuda:0')

    if config['DGM'] == 'VQ-VAE':
        model.eval()
        result = model(x)
    elif config['DGM'] == 'FlowModel':
        print("Image Sampling...")
        result = model.sampling(sample_size=8, num_step=config['Time steps'])
    
    save_images(torch.vstack([x.cpu(), result.cpu()]), f"{config['DGM']}_{config['Data']}_imgs_{identifier}", file_path=paths['sample_path'])
    print("Image sampling is Done.")