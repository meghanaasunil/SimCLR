import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from data_aug.view_generator import ContrastiveLearningViewGenerator
from data_aug.gaussian_blur import GaussianBlur  # Use the custom GaussianBlur from original SimCLR

class ProtoDACLDataset(Dataset):
    """
    Dataset for ProtoDACL that provides class and domain labels.
    
    Enhances PACSDataset by returning domain labels along with class labels.
    """
    
    def __init__(self, root_dir, domains=None, exclude_domains=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the PACS domains
            domains (list, optional): If specified, only loads images from these domains
            exclude_domains (list, optional): If specified, excludes these domains
            transform (callable, optional): Transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # List of all domains
        all_domains = ['Photo', 'Art', 'Cartoon', 'Sketch']
        
        # Filter domains based on include/exclude parameters
        if domains:
            domains_to_use = [d for d in domains if d in all_domains]
        elif exclude_domains:
            domains_to_use = [d for d in all_domains if d not in exclude_domains]
        else:
            domains_to_use = all_domains
            
        print(f"Using domains: {domains_to_use}")
        
        # Collect all image paths
        self.images = []
        self.class_labels = []
        self.domain_labels = []
        
        # Map class names to indices
        self.class_to_idx = {}
        # Map domain names to indices
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(all_domains)}
        
        class_idx = 0
        
        for domain in domains_to_use:
            domain_dir = os.path.join(root_dir, domain)
            domain_idx = self.domain_to_idx[domain]
            
            # Check if domain directory exists
            if not os.path.isdir(domain_dir):
                print(f"Warning: Domain directory {domain_dir} not found")
                continue
            
            # Iterate through class directories
            for class_name in sorted(os.listdir(domain_dir)):
                class_dir = os.path.join(domain_dir, class_name)
                
                # Skip if not a directory
                if not os.path.isdir(class_dir):
                    continue
                
                # Add class to mapping if not already present
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    class_idx += 1
                
                # Iterate through images
                for img_name in os.listdir(class_dir):
                    # Skip if not an image file
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.class_labels.append(self.class_to_idx[class_name])
                    self.domain_labels.append(domain_idx)
        
        print(f"Loaded {len(self.images)} images from {len(domains_to_use)} domains")
        print(f"Classes: {self.class_to_idx}")
        print(f"Domains: {self.domain_to_idx}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        class_label = self.class_labels[idx]
        domain_label = self.domain_labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        else:
            # If no transform is provided, at least convert to tensor
            img = transforms.ToTensor()(img)
        
        return img, (torch.tensor(class_label), torch.tensor(domain_label))


class ProtoDACLDatasetWrapper:
    """
    Wrapper for ProtoDACL dataset to provide a consistent interface for training.
    """
    
    def __init__(self, root_folder):
        self.root_folder = root_folder
    
    @staticmethod
    def get_protodacl_pipeline_transform(size, s=1):
        """
        Return a set of data augmentation transformations for ProtoDACL.
        
        Similar to SimCLR but with potentially stronger augmentations.
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # Use the custom GaussianBlur implementation from the original SimCLR
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),  # Using custom GaussianBlur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return data_transforms
    
    def get_dataset(self, name, n_views, source_domains=None, target_domain=None):
        """
        Get the appropriate dataset for training or evaluation.
        
        Args:
            name: Dataset name
            n_views: Number of views for contrastive learning
            source_domains: Source domains for training
            target_domain: Target domain for evaluation
            
        Returns:
            Dataset for training or evaluation
        """
        if name == 'pacs':
            return ProtoDACLDataset(
                self.root_folder,
                domains=source_domains,
                exclude_domains=target_domain,
                transform=ContrastiveLearningViewGenerator(
                    self.get_protodacl_pipeline_transform(224),
                    n_views
                )
            )
        else:
            raise ValueError(f"Dataset {name} not supported by ProtoDACL")