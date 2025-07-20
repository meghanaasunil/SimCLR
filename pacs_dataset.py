import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PACSDataset(Dataset):
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
        self.labels = []
        self.domains = []
        
        # Map class names to indices
        self.class_to_idx = {}
        idx = 0
        
        for domain in domains_to_use:
            domain_dir = os.path.join(root_dir, domain)
            
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
                    self.class_to_idx[class_name] = idx
                    idx += 1
                
                # Iterate through images
                for img_name in os.listdir(class_dir):
                    # Skip if not an image file
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
                    self.domains.append(domain)
        
        print(f"Loaded {len(self.images)} images from {len(domains_to_use)} domains")
        print(f"Classes: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        else:
            # If no transform is provided, at least convert to tensor
            img = transforms.ToTensor()(img)
        
        return img, label