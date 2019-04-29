import torch
import os
from torchvision import transforms, datasets, models

data_dir = '/home/mhmd/Projects/Udacity/flowers/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Define your transforms for the training, validation, and testing sets
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
} 

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, 
                                  transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir,
                                  transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir,
                                  transform=data_transforms['test'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'],
                                        batch_size=32,
                                        shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                        batch_size=32,
                                        shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'],
                                        batch_size=32,
                                        shuffle=True) 
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
