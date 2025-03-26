import os
import argparse
import torch.optim as optim
import torch.nn as nn
from dgm.vqvae import VQVAE
from data import get_loader
from trainer import VQVAETrainer
from util import *

if __name__ == "__main__":

    import wandb
    wandb.login()

    parser = argparse.ArgumentParser(description='Train Deep Generative Model')
    parser.add_argument('--dgm', type=str, default='vqvae', required=True, choices=['vqvae', 'ddpm'], help = 'path of config file')
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

    if (config['Device'] is None) and (args.gpu is not None):
        config['Device'] = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Device is set to {config['Device']}.")
    elif args.gpu is None:
        config['Device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError(f"GPU number {args.gpu} is not available. Please check your GPU number.")
        
    wandb.init(
        project="DGM-PyTorch",
        name=f"{config['DGM']}-{config['Data']}",
        config=config,
    )
    
    # Data loader for Visual data
    dataloader = get_loader(train_valid_split=None, num_workers=8, **setting)
    train_loader = dataloader['train']
    val_loader = dataloader['valid']

    if config['DGM'] == 'DDPM':
        raise NotImplementedError
    elif config['DGM'] == 'VQ-VAE':
        model = VQVAE(dim=config['init_Channel'], 
                      dim_mults=tuple(config['Channel_mults']), 
                      in_channel=config['in_Channel']).to(config['Device'])
        Trainer = VQVAETrainer
    
    print("Model is uploaded to GPU.")

    # Training setting
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['Learning rate'])
    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, device=config['Device'], model_path=paths['model_path'] if args.save else None)
    trainer.fit(config['Epochs'])
    print("Training is Done.")

    print("Image Sampling...")
    sample_batch = next(iter(val_loader))[0]
    if isinstance(sample_batch, list):
        x = sample_batch[0][0:16].to(config['Device'])
    else:
        x = sample_batch[0:16].to(config['Device'])
    result = model(x) 
    
    save_images(x.cpu(), f"{config['Data']}_original_imgs")
    save_images(result['output'].cpu(), f"{config['Data']}_recon_imgs")
    print("Image sampling is Done.")
