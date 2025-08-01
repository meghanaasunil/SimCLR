import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.protodacl_dataset import ProtoDACLDatasetWrapper
from models.resnet_protodacl import ResNetProtoDACL
from protodacl import ProtoDACL
import os
import warnings
warnings.filterwarnings("ignore")
import yaml

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ProtoDACL')
parser.add_argument('-data', metavar='DIR', default='/kaggle/input/pacs-dataset/PACS',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='pacs',
                    help='dataset name', choices=['pacs'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

# Specific arguments for ProtoDACL
parser.add_argument('--memory-size', default=4096, type=int,
                    help='Size of memory bank')
parser.add_argument('--prototype-weight', default=0.5, type=float,
                    help='Weight for prototype alignment loss')

# Domain-specific training arguments
parser.add_argument('--source-domains', nargs='+', type=str, default=None,
                    help='Source domains for training (e.g., Photo Art Cartoon)')
parser.add_argument('--target-domain', type=str, default=None,
                    help='Target domain for testing (e.g., Sketch)')
parser.add_argument('--experiment-name', type=str, default='default',
                    help='Name for the experiment (used for saving checkpoints)')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    
    # Set experiment name for saving checkpoints
    if args.experiment_name == 'default' and args.source_domains and args.target_domain:
        args.experiment_name = f"ProtoDACL_{'_'.join(args.source_domains)}-to-{args.target_domain}"
        
    print(f"Running experiment: {args.experiment_name}")
    
    # Check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ProtoDACLDatasetWrapper(args.data)

    # Get dataset with domain-specific filtering
    train_dataset = dataset.get_dataset(
        args.dataset_name, 
        args.n_views, 
        source_domains=args.source_domains,
        target_domain=[args.target_domain] if args.target_domain else None
    )
    
    # Check if the dataset is empty
    if len(train_dataset) == 0:
        print("ERROR: Dataset is empty! Please check your dataset path and structure.")
        print("Current dataset path:", args.data)
        print("Make sure the PACS dataset is correctly organized with domain folders.")
        sys.exit(1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetProtoDACL(
        base_model=args.arch,
        out_dim=args.out_dim,
        memory_size=args.memory_size
    )

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    # It's a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        protodacl = ProtoDACL(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        protodacl.train(train_loader)
        
    # Save experiment results in a specific directory
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
        
    # Copy the final checkpoint to the experiment results directory
    checkpoint_dir = protodacl.writer.log_dir
    final_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_{args.epochs:04d}.pth.tar')
    
    if os.path.exists(final_checkpoint):
        experiment_dir = os.path.join('experiment_results', args.experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        
        import shutil
        # Copy the checkpoint
        shutil.copy2(final_checkpoint, os.path.join(experiment_dir, 'model.pth.tar'))
        
        # Create a clean config file with only necessary parameters
        config_dict = {
            'arch': args.arch,
            'out_dim': args.out_dim,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'temperature': args.temperature,
            'prototype_weight': args.prototype_weight,
            'memory_size': args.memory_size,
            'dataset_name': args.dataset_name,
            'source_domains': args.source_domains,
            'target_domain': args.target_domain
        }
        
        with open(os.path.join(experiment_dir, 'config.yml'), 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=False)
    
    print(f"Saved experiment results to {experiment_dir}")


if __name__ == '__main__':
    main()