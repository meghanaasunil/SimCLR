import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss.protodacl_loss import ProtoDACLLoss
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class ProtoDACL(object):
    """
    ProtoDACL training framework.
    """
    
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, 'training.log'),
            level=logging.DEBUG
        )
        
        self.criterion = ProtoDACLLoss(
            temperature=self.args.temperature,
            prototype_weight=self.args.prototype_weight
        ).to(self.args.device)
    
    def train(self, train_loader):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
        """
        scaler = GradScaler(enabled=self.args.fp16_precision)
        
        # Save config file
        save_config_file(self.writer.log_dir, self.args)
        
        n_iter = 0
        logging.info(f"Start ProtoDACL training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        
        for epoch_counter in range(self.args.epochs):
            for data in tqdm(train_loader):
                # Print data format for debugging
                print(f"Data type: {type(data)}")
                if isinstance(data, list):
                    print(f"List length: {len(data)}")
                    for i, item in enumerate(data):
                        print(f"Item {i} type: {type(item)}")
                
                # Handle different data formats
                if isinstance(data, tuple) and len(data) == 2:
                    # Format: (images, labels)
                    if isinstance(data[0], list) and isinstance(data[1], tuple) and len(data[1]) == 2:
                        # Format: ([view1, view2, ...], (class_labels, domain_labels))
                        images = torch.cat(data[0], dim=0)
                        class_labels, domain_labels = data[1]
                        # Repeat labels for each view
                        class_labels = class_labels.repeat(self.args.n_views)
                        domain_labels = domain_labels.repeat(self.args.n_views)
                    else:
                        # Legacy format
                        images, labels = data
                        if isinstance(images, list):
                            images = torch.cat(images, dim=0)
                        
                        # Handle different possible formats of labels
                        if isinstance(labels, tuple) and len(labels) == 2:
                            # Tuple with (class_labels, domain_labels)
                            class_labels, domain_labels = labels
                            # Repeat labels for each view if images were a list
                            if isinstance(data[0], list):
                                class_labels = class_labels.repeat(self.args.n_views)
                                domain_labels = domain_labels.repeat(self.args.n_views)
                        elif isinstance(labels, list) and len(labels) == 2:
                            # List with [class_labels, domain_labels]
                            class_labels, domain_labels = labels[0], labels[1]
                            # Repeat labels for each view if images were a list
                            if isinstance(data[0], list):
                                class_labels = class_labels.repeat(self.args.n_views)
                                domain_labels = domain_labels.repeat(self.args.n_views)
                        else:
                            # Only class labels provided, create dummy domain labels
                            class_labels = labels
                            domain_labels = torch.zeros_like(class_labels)
                            # Repeat labels for each view if images were a list
                            if isinstance(data[0], list):
                                class_labels = class_labels.repeat(self.args.n_views)
                                domain_labels = domain_labels.repeat(self.args.n_views)
                elif isinstance(data, list):
                    # Direct list format from some data loaders
                    if len(data) >= 2:
                        # Assume first item is images, second is labels
                        images_part = data[0]
                        labels_part = data[1] if len(data) > 1 else None
                        
                        # Handle images that might be a list of tensors
                        if isinstance(images_part, list) and all(isinstance(img, torch.Tensor) for img in images_part):
                            images = torch.cat(images_part, dim=0)
                        else:
                            images = images_part
                            
                        # Handle labels
                        if labels_part is not None:
                            if isinstance(labels_part, tuple) and len(labels_part) == 2:
                                class_labels, domain_labels = labels_part
                            elif isinstance(labels_part, list) and len(labels_part) == 2:
                                class_labels, domain_labels = labels_part[0], labels_part[1]
                            elif isinstance(labels_part, torch.Tensor):
                                class_labels = labels_part
                                domain_labels = torch.zeros_like(class_labels)
                            else:
                                # Fallback
                                print(f"WARNING: Unexpected labels format: {type(labels_part)}")
                                class_labels = torch.zeros(images.size(0), dtype=torch.long)
                                domain_labels = torch.zeros(images.size(0), dtype=torch.long)
                        else:
                            # No labels provided
                            class_labels = torch.zeros(images.size(0), dtype=torch.long)
                            domain_labels = torch.zeros(images.size(0), dtype=torch.long)
                            
                        # Repeat labels if necessary (for multiple views)
                        if isinstance(images_part, list) and len(images_part) > 1:
                            if len(class_labels) * len(images_part) == images.size(0):
                                class_labels = class_labels.repeat(len(images_part))
                                domain_labels = domain_labels.repeat(len(images_part))
                    else:
                        # Just images with no labels
                        images = data[0]
                        class_labels = torch.zeros(images.size(0), dtype=torch.long)
                        domain_labels = torch.zeros(images.size(0), dtype=torch.long)
                else:
                    # Unexpected format
                    print(f"Unexpected data format. Type: {type(data)}")
                    if hasattr(data, '__dict__'):
                        print(f"Data attributes: {data.__dict__}")
                    raise ValueError(f"Unexpected data format: {type(data)}")
                
                # Move data to device
                images = images.to(self.args.device)
                class_labels = class_labels.to(self.args.device)
                domain_labels = domain_labels.to(self.args.device)
                
                with autocast(enabled=self.args.fp16_precision):
                    # Forward pass with memory bank update
                    features_dict = self.model.forward_memory(
                        images, class_labels, domain_labels, update_memory=True)
                    
                    # Compute loss
                    loss, contrastive_loss, proto_loss = self.criterion(
                        features_dict, class_labels, domain_labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # Log statistics
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss/total', loss, global_step=n_iter)
                    self.writer.add_scalar('loss/contrastive', contrastive_loss, global_step=n_iter)
                    self.writer.add_scalar('loss/prototype', proto_loss, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                
                n_iter += 1
            
            # Scheduler step
            if epoch_counter >= 10:  # Warmup for the first 10 epochs
                self.scheduler.step()
            
            # Log epoch statistics
            logging.debug(
                f"Epoch: {epoch_counter}\t"
                f"Loss: {loss.item():.4f}\t"
                f"Contrastive Loss: {contrastive_loss.item():.4f}\t"
                f"Prototype Loss: {proto_loss.item():.4f}"
            )
            
            # Save checkpoint every 20 epochs
            if (epoch_counter + 1) % 20 == 0 or epoch_counter == self.args.epochs - 1:
                checkpoint_name = f'checkpoint_{epoch_counter+1:04d}.pth.tar'
                save_checkpoint(
                    {
                        'epoch': epoch_counter + 1,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    },
                    is_best=False,
                    filename=os.path.join(self.writer.log_dir, checkpoint_name)
                )
        
        logging.info("Training has finished.")
        
        # Save final model checkpoint
        checkpoint_name = f'checkpoint_{self.args.epochs:04d}.pth.tar'
        save_checkpoint(
            {
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(self.writer.log_dir, checkpoint_name)
        )
        
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")