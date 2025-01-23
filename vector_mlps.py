import torch
import torch.nn as nn


class MLPFlow(nn.Module):
    """MLP-based flow model for continuous vector data"""

    def __init__(self, input_dim, hidden_dims=[256, 512, 256], num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Embedding for time
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # Embedding for class labels
        self.class_embed = nn.Embedding(num_classes, 64)

        # Main network
        layers = []
        prev_dim = input_dim + 128  # input_dim + time_embedding + class_embedding

        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = dim

        # Final layer outputs the velocity with same dimension as input
        layers.append(nn.Linear(prev_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t, x, y):
        # Embed time
        t_emb = self.time_embed(t.reshape(-1, 1))

        # Embed class labels
        y_emb = self.class_embed(y)

        # Concatenate input with time and class embeddings
        h = torch.cat([x, t_emb, y_emb], dim=1)

        # Get velocity prediction
        return self.net(h)


class MLPValue(nn.Module):
    """MLP-based value model for continuous vector data"""

    def __init__(self, input_dim, hidden_dims=[256, 256, 128], num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Embedding for time
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # Embedding for class labels
        self.class_embed = nn.Embedding(num_classes, 64)

        # Main network
        layers = []
        prev_dim = input_dim + 128  # input_dim + time_embedding + class_embedding

        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = dim

        # Final layer outputs scalar value prediction
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])  # Normalize to [0,1]

        self.net = nn.Sequential(*layers)

    def forward(self, t, x, y):
        # Embed time
        t_emb = self.time_embed(t.reshape(-1, 1))

        # Embed class labels
        y_emb = self.class_embed(y)

        # Concatenate input with time and class embeddings
        h = torch.cat([x, t_emb, y_emb], dim=1)

        # Get value prediction
        return self.net(h).squeeze(-1)
