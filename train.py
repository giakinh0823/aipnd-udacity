import argparse
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, VGG13_Weights
import os


def get_input_args():
    parser = argparse.ArgumentParser(
        description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Model architecture (vgg16, vgg13, etc.)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')
    return parser.parse_args()


def validate_args(input_args):
    print("Validate input!")
    print(torch.__version__)
    print(torch.cuda.is_available())

    if input_args.gpu and not torch.cuda.is_available():
        raise Exception("Error! not find gpu")

    if not os.path.isdir(input_args.data_dir):
        raise Exception('Error! Directory does not exist')

    data_dir = os.listdir(input_args.data_dir)
    if not set(data_dir).issubset({'test', 'train', 'valid'}):
        raise Exception('Error! test, train or valid sub director are missing')

    if input_args.arch not in ('vgg16', 'vgg13'):
        raise Exception('Error! Please choose vgg16 or vgg13')


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir,
                                      transform=data_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_dir,
                                      transform=data_transforms['valid']),
        'test': datasets.ImageFolder(root=test_dir,
                                     transform=data_transforms['test']),
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64,
                            shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=64,
                            shuffle=False),
        'test': DataLoader(image_datasets['test'], batch_size=64, shuffle=False)
    }

    return dataloaders, image_datasets['train'].class_to_idx


def build_model(arch, hidden_units, output_size, device):
    if arch == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif arch == 'vgg13':
        model = models.vgg13(weights=VGG13_Weights.DEFAULT)
    else:
        raise ValueError(
            "Unsupported architecture! Please choose 'vgg16' or 'vgg13'.")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model = model.to(device)  # Move the model to the device (GPU or CPU)

    return model


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=5):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data from dataloaders
            for inputs, labels in dataloaders[phase]:
                # move data to device (cpu or gpu)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicted == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_accuracy = running_corrects.double() / len(
                dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

    print("Done training!")

    return model


def save_checkpoint(model, save_dir, class_to_idx, arch, hidden_units,
        learning_rate, num_epochs):

    checkpoint = {
        'arch': arch,
        'input_size': 25088,
        'output_size': len(class_to_idx),
        'hidden_layers': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'learning_rate': learning_rate,
        'epochs': num_epochs
    }

    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')


def main():
    # get input
    input_args = get_input_args()
    validate_args(input_args)

    device = torch.device(
        "cuda" if input_args.gpu and torch.cuda.is_available() else "cpu")

    # load data
    dataloaders, class_to_idx = load_data(input_args.data_dir)

    model = build_model(input_args.arch, input_args.hidden_units,
                        len(class_to_idx), device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=input_args.learning_rate)

    # train model
    model = train_model(model, criterion, optimizer, dataloaders, device,
                        input_args.epochs)

    # save checkpoint
    save_checkpoint(model, input_args.save_dir, class_to_idx, input_args.arch,
                    input_args.hidden_units, input_args.learning_rate,
                    input_args.epochs)


if __name__ == '__main__':
    main()
