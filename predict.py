import argparse
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import VGG16_Weights, VGG13_Weights


def get_input_args():
    parser = argparse.ArgumentParser(
        description="Predict the class for an input image using a trained network.")
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str,
                        help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')
    return parser.parse_args()


def load_checkpoint(filepath):
    # Load the checkpoint from the specified file
    checkpoint_load = torch.load(filepath, weights_only=False)

    # Load the pre-trained VGG16 model
    if checkpoint_load['arch'] == 'vgg16':
        model_load = models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif checkpoint_load['arch'] == 'vgg13':
        model_load = models.vgg13(weights=VGG13_Weights.DEFAULT)
    else:
        raise ValueError(
            "Unsupported architecture! Please choose 'vgg16' or 'vgg13'.")

    # Freeze parameters so we don't backprop through them
    for param_load in model_load.parameters():
        param_load.requires_grad = False

    # Define a new classifier based on the checkpoint information    
    classifier_load = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint_load['input_size'],
                          checkpoint_load['hidden_layers'])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(checkpoint_load['hidden_layers'],
                          checkpoint_load['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replace the VGG classifier with our new classifier
    model_load.classifier = classifier_load

    # Load the state dictionary into the model
    model_load.load_state_dict(checkpoint_load['state_dict'], strict=False)

    # Load the class to index mapping
    model_load.class_to_idx = checkpoint_load['class_to_idx']

    return model_load


def process_image(image_path):
    # Load the image
    pil_image = Image.open(image_path)

    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio
    if pil_image.size[0] < pil_image.size[1]:
        pil_image.thumbnail((256, 256 * pil_image.size[1] // pil_image.size[0]))
    else:
        pil_image.thumbnail((256 * pil_image.size[0] // pil_image.size[1], 256))

    # Crop out the center 224x224 portion of the image
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = left + 224
    bottom = top + 224

    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert the image to a numpy array
    np_image = np.array(pil_image) / 255.0

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions to match PyTorch's expectations
    np_image = np_image.transpose((2, 0, 1))

    # Convert to a PyTorch tensor
    tensor_image = torch.from_numpy(np_image).float()

    return tensor_image


def predict(image_path, model, topk=5):
    # Process the image
    image = process_image(image_path)

    # Add batch dimension
    image = image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model and image to the appropriate device
    model.eval()
    model.to(device)

    image = image.to(device)

    # Forward pass
    with torch.no_grad():
        output = model.forward(image)

    # Calculate probabilities
    ps = torch.exp(output)

    # Get the top k probabilities and indices
    top_p, top_class = ps.topk(topk, dim=1)

    # Convert to lists
    top_p = top_p.cpu().numpy().tolist()[0]
    top_class = top_class.cpu().numpy().tolist()[0]

    # Invert the class_to_idx dictionary
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Get the actual class labels
    top_class = [idx_to_class[i] for i in top_class]

    return top_p, top_class


def main():
    # get input args
    input_args = get_input_args()

    # load check point
    model = load_checkpoint(input_args.checkpoint)

    # predict
    probs, classes = predict(input_args.input, model, input_args.top_k)

    # get classes
    if input_args.category_names:
        with open(input_args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]

    # Result
    print(f'Probabilities: {probs}')
    print(f'Classes: {classes}')


if __name__ == '__main__':
    main()
