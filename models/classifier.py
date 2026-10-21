import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, bottleneck_dim=256):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim
        self.head = torch.nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        if type(x) == tuple:
            x = x[0]
        x = self.head(x)
        return x

class Classifier_new(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim=512):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim
        self.head = torch.nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        if type(x) == tuple:
            x = x[0]
        x = self.head(x)
        return x

class DINOClassifier(nn.Module):
    def __init__(self, encoder, num_classes, bottleneck_dim=256):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim

        self.head = torch.nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x, is_training=True)["x_norm_patchtokens"]
        x = x.mean(1)
        x = self.head(x)
        return x