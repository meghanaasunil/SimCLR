import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class ProtoDACLViewGenerator(object):
    """
    Generate multiple views of the same image for contrastive learning,
    while preserving label information.
    """

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, sample):
        """
        Args:
            sample: tuple of (image, labels) where labels can be a single tensor or a tuple
        
        Returns:
            List of transformed images and the original labels
        """
        if isinstance(sample, tuple) and len(sample) == 2:
            # Unpack the sample
            image, labels = sample
            # Generate multiple views of the image
            views = [self.base_transform(image) for _ in range(self.n_views)]
            return views, labels
        else:
            # Just the image without labels
            return [self.base_transform(sample) for _ in range(self.n_views)]