import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoDACLLoss(nn.Module):
    """
    ProtoDACL loss combining contrastive loss with prototype alignment loss.
    """
    
    def __init__(self, temperature=0.07, prototype_weight=0.5, contrast_mode='all'):
        """
        Initialize ProtoDACL loss.
        
        Args:
            temperature: Temperature parameter for contrastive loss
            prototype_weight: Weight for prototype alignment loss
            contrast_mode: 'all' or 'one', whether to contrast all pairs or one pair
        """
        super(ProtoDACLLoss, self).__init__()
        self.temperature = temperature
        self.prototype_weight = prototype_weight
        self.contrast_mode = contrast_mode
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features_dict, class_labels, domain_labels):
        """
        Compute ProtoDACL loss.
        
        Args:
            features_dict: Dictionary containing normalized_projections, prototypes, hard_negatives
            class_labels: Class labels
            domain_labels: Domain labels
            
        Returns:
            Total loss, contrastive loss, prototype alignment loss
        """
        normalized_projections = features_dict['normalized_projections']
        prototypes = features_dict['prototypes']
        hard_negatives = features_dict['hard_negatives']
        
        device = normalized_projections.device
        batch_size = normalized_projections.size(0)
        
        # Standard contrastive loss (similar to SimCLR)
        contrastive_loss = self._compute_contrastive_loss(
            normalized_projections, hard_negatives, class_labels, domain_labels)
        
        # Prototype alignment loss
        proto_loss = self._compute_prototype_alignment_loss(
            normalized_projections, prototypes, class_labels)
        
        # Combine losses
        total_loss = contrastive_loss + self.prototype_weight * proto_loss
        
        return total_loss, contrastive_loss, proto_loss
    
    def _compute_contrastive_loss(self, features, hard_negatives, class_labels, domain_labels):
        """
        Compute contrastive loss with domain-adversarial negative mining.
        
        Args:
            features: Normalized feature embeddings
            hard_negatives: Indices of hard negative samples
            class_labels: Class labels
            domain_labels: Domain labels
            
        Returns:
            Contrastive loss
        """
        device = features.device
        batch_size = features.size(0)
        
        # Generate positive pairs (using augmentations)
        # In this case, we assume features already contain augmented views
        # For a typical batch of N images with 2 views each, features would be 2N
        # We need to reshape to get the positive pairs
        if self.contrast_mode == 'all':
            # Assume the first half of the batch are the first views
            # and the second half are the second views
            # Create mask for positive pairs
            mask = torch.zeros((batch_size, batch_size), dtype=torch.float32).to(device)
            for i in range(batch_size):
                # Identify samples from same class
                same_class = (class_labels == class_labels[i])
                # Exclude self
                same_class[i] = False
                mask[i, same_class] = 1.0
        else:
            # For one-to-one contrast, just use diagonal mask
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Remove self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _compute_prototype_alignment_loss(self, features, prototypes, class_labels):
        """
        Compute prototype alignment loss.
        
        Args:
            features: Normalized feature embeddings
            prototypes: Dictionary mapping class labels to prototype embeddings
            class_labels: Class labels
            
        Returns:
            Prototype alignment loss
        """
        if not prototypes:  # Empty prototypes dictionary
            return torch.tensor(0.0, device=features.device)
        
        batch_size = features.size(0)
        loss = torch.tensor(0.0, device=features.device)
        count = 0
        
        # Compute distance between each feature and its class prototype
        for i in range(batch_size):
            cls = class_labels[i].item()
            if cls in prototypes:
                prototype = prototypes[cls].to(features.device)
                # Use cosine similarity (features are already normalized)
                sim = torch.dot(features[i], prototype)
                # Convert to distance: 1 - sim
                dist = 1.0 - sim
                loss += dist
                count += 1
        
        # Average loss
        if count > 0:
            loss = loss / count
        
        return loss