import os
import numpy as np
import torchvision.datasets as datasets
from torch.utils import data
from torchvision import transforms
from PIL import Image

class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader."""
    def __init__(self, root, transform=None):
        """Initialize image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
    
    def __len__(self):
        """Return the total number of image files."""
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """Read an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image    

def get_loader(train_valid_split=None, num_workers=2, **setting):
    """Create and return Dataloader."""
    config = setting['config']
    image_path = setting['path']['data_root']
    data_loader = None
    image_size = (config['Resolution'], config['Resolution'])
    transform = basic_transform(image_size)

    if config['Data'] in ['MNIST', 'CIFAR10']:
        train_dataset = datasets.__dict__[config['Data']](root=image_path, 
                                                    train=True, 
                                                    download=True, 
                                                    transform=transform)
        valid_dataset = datasets.__dict__[config['Data']](root=image_path,
                                                    train=False,
                                                    download=True,
                                                    transform=transform)
        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=config['Batch size'],
                                       shuffle=True,
                                       num_workers=num_workers)
        valid_loader = data.DataLoader(dataset=valid_dataset,
                                       batch_size=config['Batch size'],
                                       shuffle=False,
                                       num_workers=num_workers)
        data_loader = {'train': train_loader, 'valid': valid_loader}

    elif config['Data'] in ['CelebA', 'AFHQ']:
        dataset = ImageFolder(image_path, transform)
    
        if (train_valid_split is None) and (data_loader is None):
            train_loader = data.DataLoader(dataset=dataset,
                                    batch_size=config['Batch size'],
                                    shuffle=True,
                                    num_workers=num_workers)
            valid_loader = None
        else:
            train_size = int(train_valid_split * len(dataset))
            valid_size = len(dataset) - train_size
            train_dataset, valid_dataset = data.random_split(dataset, [train_size, valid_size])
            train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=config['Batch size'],
                                    shuffle=True,
                                    num_workers=num_workers)
            valid_loader = data.DataLoader(dataset=valid_dataset,
                                    batch_size=config['Batch size'],
                                    shuffle=False,
                                    num_workers=num_workers)
        data_loader = {'train': train_loader, 'valid': valid_loader}
    else:
        raise ValueError(f"Dataset {config['Data']} is not supported.")
    return data_loader

def basic_transform(image_size):
    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                    ])
    return transform

def reverse_transform():
    reverse_transform = transforms.Compose([
     transforms.Lambda(lambda t: (t + 1) / 2),
     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     transforms.Lambda(lambda t: t * 255.),
     transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
     transforms.ToPILImage(),
    ])
    return reverse_transform