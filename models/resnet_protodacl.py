import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from exceptions.exceptions import InvalidBackboneError


class MemoryBank(nn.Module):
    """Memory bank to store embeddings with class and domain labels."""
    
    def __init__(self, feature_dim, max_size=4096):
        """
        Initialize memory bank.
        
        Args:
            feature_dim: Dimension of feature embeddings
            max_size: Maximum number of samples to store in memory bank
        """
        super(MemoryBank, self).__init__()
        self.max_size = max_size
        self.feature_dim = feature_dim
        
        # Initialize empty bank
        self.features = torch.zeros(0, feature_dim)
        self.class_labels = torch.zeros(0, dtype=torch.long)
        self.domain_labels = torch.zeros(0, dtype=torch.long)
        self.ptr = 0
        self.is_full = False
        
    def update(self, features, class_labels, domain_labels):
        """
        Update memory bank with new features and labels.
        
        Args:
            features: Feature embeddings (N, feature_dim)
            class_labels: Class labels (N)
            domain_labels: Domain labels (N)
        """
        batch_size = features.size(0)
        
        # Initialize memory bank if first update
        if self.features.size(0) == 0:
            self.features = torch.zeros(self.max_size, self.feature_dim).to(features.device)
            self.class_labels = torch.zeros(self.max_size, dtype=torch.long).to(class_labels.device)
            self.domain_labels = torch.zeros(self.max_size, dtype=torch.long).to(domain_labels.device)
        
        # Calculate indices to update
        if self.ptr + batch_size <= self.max_size:
            # Enough space for the entire batch
            idxs = torch.arange(self.ptr, self.ptr + batch_size)
        else:
            # Not enough space, wrap around
            remaining = self.max_size - self.ptr
            idxs = torch.cat([torch.arange(self.ptr, self.max_size), 
                             torch.arange(0, batch_size - remaining)])
            self.is_full = True
        
        # Update memory bank
        self.features[idxs] = features.detach()
        self.class_labels[idxs] = class_labels.detach()
        self.domain_labels[idxs] = domain_labels.detach()
        
        # Update pointer
        self.ptr = (self.ptr + batch_size) % self.max_size
    
    def get_class_prototypes(self):
        """
        Compute class prototypes as the average embedding for each class.
        
        Returns:
            Dictionary mapping class labels to prototype embeddings
        """
        num_samples = self.max_size if self.is_full else self.ptr
        if num_samples == 0:
            return {}
        
        # Use only filled portion of memory bank
        features = self.features[:num_samples]
        class_labels = self.class_labels[:num_samples]
        
        # Get unique class labels
        unique_classes = torch.unique(class_labels)
        
        # Compute prototypes
        prototypes = {}
        for cls in unique_classes:
            mask = (class_labels == cls)
            if mask.sum() > 0:
                prototype = features[mask].mean(dim=0)
                prototypes[cls.item()] = F.normalize(prototype, dim=0)
        
        return prototypes
    
    def get_domain_adversarial_negatives(self, features, class_labels, domain_labels, k=16):
        """
        Mine hard negatives from different domains but same class.
        
        Args:
            features: Feature embeddings (N, feature_dim)
            class_labels: Class labels (N)
            domain_labels: Domain labels (N)
            k: Number of hard negatives to mine per sample
            
        Returns:
            Tensor of hard negative indices for each sample
        """
        num_samples = self.max_size if self.is_full else self.ptr
        if num_samples == 0:
            return None
        
        # Use only filled portion of memory bank
        bank_features = self.features[:num_samples]
        bank_class_labels = self.class_labels[:num_samples]
        bank_domain_labels = self.domain_labels[:num_samples]
        
        batch_size = features.size(0)
        
        # Normalize query features
        query_features = F.normalize(features, dim=1)
        
        # Compute similarity scores
        sim_scores = torch.mm(query_features, bank_features.t())  # (batch_size, num_samples)
        
        # Prepare masks for different domains and different classes
        hard_negative_indices = []
        
        for i in range(batch_size):
            query_class = class_labels[i]
            query_domain = domain_labels[i]
            
            # Find samples with different class
            diff_class_mask = (bank_class_labels != query_class)
            
            # Find samples with different domain
            diff_domain_mask = (bank_domain_labels != query_domain)
            
            # Combined mask for domain-adversarial negatives
            valid_negatives_mask = diff_class_mask
            
            # If no valid negatives found, use any different class
            if valid_negatives_mask.sum() == 0:
                valid_negatives_mask = diff_class_mask
            
            # Get similarity scores for valid negatives
            valid_scores = sim_scores[i].clone()
            valid_scores[~valid_negatives_mask] = -float('inf')
            
            # Get top-k hard negatives
            if valid_scores.size(0) > 0:
                _, hard_neg_idx = torch.topk(valid_scores, min(k, valid_scores.size(0)))
                hard_negative_indices.append(hard_neg_idx)
            else:
                # Fallback: use random indices if no valid negatives
                hard_negative_indices.append(torch.randint(0, num_samples, (k,)))
        
        return hard_negative_indices


class ResNetProtoDACL(nn.Module):
    """ProtoDACL model based on ResNet architecture."""
    
    def __init__(self, base_model, out_dim, memory_size=4096):
        """
        Initialize ProtoDACL model.
        
        Args:
            base_model: Base model architecture ('resnet18', 'resnet50', etc.)
            out_dim: Output dimension of projection head
            memory_size: Size of memory bank
        """
        super(ResNetProtoDACL, self).__init__()
        
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                           "resnet50": models.resnet50(pretrained=False)}
        
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        
        # Replace final fc layer with identity
        self.backbone.fc = nn.Identity()
        
        # Add MLP projection head
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )
        
        # Initialize memory bank
        self.memory_bank = MemoryBank(feature_dim=out_dim, max_size=memory_size)
        
    def _get_basemodel(self, model_name):
        """Get base model architecture."""
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model
    
    def forward(self, x):
        """Forward pass through model."""
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections
    
    def forward_memory(self, x, class_labels=None, domain_labels=None, update_memory=True):
        """
        Forward pass with memory bank update.
        
        Args:
            x: Input tensor
            class_labels: Class labels
            domain_labels: Domain labels
            update_memory: Whether to update memory bank
            
        Returns:
            Dictionary containing features, projections, prototypes, etc.
        """
        # Forward pass
        features, projections = self.forward(x)
        
        # Normalize projections
        normalized_projections = F.normalize(projections, dim=1)
        
        result = {
            'features': features,
            'projections': projections,
            'normalized_projections': normalized_projections,
        }
        
        # Update memory bank and get class prototypes
        if class_labels is not None and domain_labels is not None:
            if update_memory:
                self.memory_bank.update(normalized_projections, class_labels, domain_labels)
            
            prototypes = self.memory_bank.get_class_prototypes()
            result['prototypes'] = prototypes
            
            # Get domain-adversarial negatives
            hard_negatives = self.memory_bank.get_domain_adversarial_negatives(
                normalized_projections, class_labels, domain_labels)
            result['hard_negatives'] = hard_negatives
        
        return result