import torch
import wandb
import time
import os
from dgm.score import *
from dgm.vqvae import *
from tqdm.auto import tqdm

# Training code
def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):

        model.train()
        result = model(x)
        
        if isinstance(result, dict): # VQ-VAE
            loss = loss_fn(result['output'], y) + result['diff']
        else: 
            loss = loss_fn(result, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step_fn

# Evaluation code
def make_valid_step(model, loss_fn):
    def valid_sten_fn(x, y):

        model.eval()
        result = model(x)

        if isinstance(result, dict): # VQ-VAE
            loss = loss_fn(result['output'], y) + result['diff']
        else: 
            loss = loss_fn(result, y)        
        
        return loss.item()
    return valid_sten_fn

class VQVAETrainer:
    def __init__(self, model, 
                 loss_fn , 
                 optimizer, 
                 train_loader, 
                 val_loader=None, 
                 device='cpu', 
                 model_path=None
                 ):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_path = model_path
        self.device = device
        
    def train(self, epoch):
        self.model.train()
        train_step = make_train_step(self.model, self.loss_fn, self.optimizer)
        
        for batch in tqdm(self.train_loader, desc='training loop', total=len(self.train_loader)):
            if isinstance(batch, list):
                x_minibatch = batch[0].to(self.device)
            else:
                x_minibatch = batch.to(self.device)
            y_minibatch = x_minibatch
            loss = train_step(x_minibatch, y_minibatch)

        wandb.log({'Training loss': loss}, step=epoch)

    def validate(self, epoch):
        self.model.eval()
        valid_step = make_valid_step(self.model, self.loss_fn)

        for batch in self.val_loader:
            if isinstance(batch, list):
                x_minibatch = batch[0].to(self.device)
            else:
                x_minibatch = batch.to(self.device)
            y_minibatch = x_minibatch
            loss = valid_step(x_minibatch, y_minibatch)

        wandb.log({'Validation loss': loss}, step=epoch)
        print("Validation loss: {} at epoch {}".format(loss, epoch))

    def fit(self, epochs):
        print("Start Training...")
        for epoch in range(epochs):
            self.train(epoch)
            
            if self.val_loader is not None:
                self.validate(epoch)

            if (self.model_path is not None) and ((epoch+1) % 10 == 0):
                identifier = str(time.time())[6:10] + str(epoch) + 'pth'
                file_name = [self.model_path, identifier]
                file_name = os.path.join(*file_name)
                torch.save(self.model.state_dict(), file_name)
                print("Model saved at epoch {}.".format(epoch+1))
        print("Finished Training!")

class DDPMTrainer:
    def __init__(self, model, loss_fn , optimizer, train_loader, timesteps=1000, val_loader=None, device='cpu', model_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_path = model_path
        self.device = device
        self.timesteps = timesteps
        
    def train(self, epoch):
        self.model.train()
        train_step = make_train_step(self.model, self.loss_fn, self.optimizer)
        
        for x_minibatch in tqdm(self.train_loader, desc='training loop', total=len(self.train_loader)):
            x_minibatch = x_minibatch[0].to(self.device) # CelebA
            batch_size = x_minibatch.shape[0]
            t_minibatch = torch.randint(0, self.timesteps, (batch_size, ), device=self.device, dtype=torch.long)
            q = forwardProcess(timesteps=self.timesteps, device=self.device)
            x_minibatch, y_minibatch, _ = q(x_minibatch, t_minibatch)
            loss = train_step(x_minibatch, t_minibatch, y_minibatch)

        wandb.log({'Training loss': loss}, step=epoch)

    def validate(self, epoch):
        self.model.eval()
        valid_step = make_valid_step(self.model, self.loss_fn)

        for x_minibatch in self.val_loader:
            x_minibatch = x_minibatch[0].to(self.device)
            batch_size = x_minibatch.shape[0]
            t_minibatch = torch.randint(0, self.timesteps, (batch_size, ), device=self.device, dtype=torch.long)
            q = forwardProcess(timesteps=self.timesteps, device=self.device)
            x_minibatch, y_minibatch, _ = q(x_minibatch, t_minibatch)
            loss = valid_step(x_minibatch, t_minibatch, y_minibatch)

        wandb.log({'Validation loss': loss}, step=epoch)
        print("Validation loss: {} at epoch {}".format(loss, epoch))

    def fit(self, epochs):
        print("Start Training...")
        for epoch in range(epochs):
            self.train(epoch)
            self.validate(epoch)
        
            if (self.model_path is not None) and ((epoch+1) % 10 == 0):
                file_name = [self.model_path, 'ddpm', str(epoch), '.pth']
                file_name = '_'.join(file_name) 
                torch.save(self.model.state_dict(), file_name)
                print("Model saved at epoch {}.".format(epoch+1))
        print("Finished Training!")