import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
from pacs_dataset import PACSDataset
from models.resnet_simclr import ResNetSimCLR
import os
import yaml
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Evaluate SimCLR on PACS dataset')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to the SimCLR checkpoint')
parser.add_argument('--target-domain', type=str, required=True,
                    help='Target domain for evaluation')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

def main():
    args = parser.parse_args()
    
    # Check for CUDA
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    # Load config from checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, 'config.yml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded config from {config_path}")
    else:
        config = {'arch': args.arch, 'out_dim': 128}
        print(f"Config not found at {config_path}, using default values")
    
    # Create the target dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_dataset = PACSDataset(
        args.data,
        domains=[args.target_domain],
        transform=transform
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Create the model - just the backbone without projection head
    if config['arch'] == 'resnet18':
        backbone = models.resnet18(pretrained=False)
        feature_dim = 512
    elif config['arch'] == 'resnet50':
        backbone = models.resnet50(pretrained=False)
        feature_dim = 2048
    else:
        raise ValueError(f"Unsupported architecture: {config['arch']}")
    
    # Remove the final fully connected layer
    backbone.fc = nn.Identity()
    backbone = backbone.to(args.device)
    
    # Load the SimCLR model
    simclr_model = ResNetSimCLR(base_model=config['arch'], out_dim=config['out_dim'])
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    simclr_model.load_state_dict(checkpoint['state_dict'])
    
    # Extract the backbone weights and load them into our backbone model
    for name, param in simclr_model.backbone.named_parameters():
        if name.startswith('fc'):
            continue  # Skip the projection head
        backbone_name = name
        backbone.state_dict()[backbone_name].copy_(param.data)
    
    # Create a linear classifier
    num_classes = len(target_dataset.class_to_idx)
    classifier = nn.Linear(feature_dim, num_classes).to(args.device)
    
    # Train linear classifier on features
    backbone.eval()
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training linear classifier on features from {args.target_domain} domain")
    print(f"Feature dimension: {feature_dim}, Number of classes: {num_classes}")
    
    # Training loop for the linear classifier
    for epoch in range(100):  # 100 epochs should be enough for the linear classifier
        correct = 0
        total = 0
        train_loss = 0.0
        
        for images, labels in target_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Extract features
            with torch.no_grad():
                features = backbone(images)
            
            # Forward pass through classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        # Print epoch statistics
        accuracy = 100 * correct / total
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {train_loss/len(target_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    # Evaluate the final model
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in target_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Extract features
            features = backbone(images)
            
            # Forward pass through classifier
            outputs = classifier(features)
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    final_accuracy = 100 * correct / total
    print(f'Final Accuracy on {args.target_domain} domain: {final_accuracy:.2f}%')
    
    # Save results
    results_dir = 'evaluation_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
    result_file = os.path.join(results_dir, f'{checkpoint_name}_{args.target_domain}.txt')
    
    with open(result_file, 'w') as f:
        f.write(f'Target Domain: {args.target_domain}\n')
        f.write(f'Accuracy: {final_accuracy:.2f}%\n')
    
    print(f'Results saved to {result_file}')

if __name__ == '__main__':
    main()