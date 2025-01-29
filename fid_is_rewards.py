import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import mahalanobis
import pickle
from train_cifar_classifier import CIFAR100Classifier


class FIDISRewardNet(nn.Module):
    def __init__(self, classifier_path="saved_models/cifar100_classifier.pt"):
        super().__init__()

        # Load inception model for FID
        self.inception_model = models.inception_v3(
            pretrained=True, transform_input=False
        )
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()

        # Load classifier for IS
        self.classifier = CIFAR100Classifier()
        self.classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
        self.classifier.eval()

        # Load CIFAR-100 statistics for FID
        with open("cifar100_fid_stats.pkl", "rb") as f:
            cifar100_stats = pickle.load(f)
        self.mu_real = cifar100_stats["mu_cifar100"]
        self.sigma_real = cifar100_stats["sigma_cifar100"]

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Define transforms
        self.inception_transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def fid_single(self, image):
        image = self.inception_transform(image)
        device = next(self.inception_model.parameters()).device
        image = image.to(device)

        with torch.no_grad():
            feats = self.inception_model(image).cpu().numpy()

        sigma_inv = np.linalg.inv(self.sigma_real)
        fid_scores = np.array(
            [mahalanobis(feat, self.mu_real, sigma_inv) for feat in feats]
        )
        return fid_scores

    def is_single(self, image):
        # Expect images to be tensor of shape [B, C, H, W]
        images = self.inception_transform(image)
        device = next(self.classifier.parameters()).device
        images = images.to(device)

        with torch.no_grad():
            logits = self.classifier(images)
            probs = (
                torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            )  # [B, num_classes]

        # Compute IS score for each image
        entropies = -np.sum(probs * np.log(probs + 1e-9), axis=1)
        is_scores = np.exp(entropies)
        return is_scores

    def forward(self, image):
        fid_score = self.fid_single(image)
        is_score = self.is_single(image)
        return fid_score + is_score
