
import torch
import torch.nn as nn

# Make results reproducible
torch.manual_seed(42)


# Model 1: Multi-Layer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# Model 2: Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)


# Model 3: Attention Model
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Convert each 7x7 patch into 64 features
        self.patch_projection = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.0
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Split 28x28 images into 16 patches of 7x7
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)

        # Shape: [batch, 16, 49]
        patches = patches.contiguous().view(x.size(0), 16, 49)

        # Project patches to 64 dimensions
        x = self.patch_projection(patches)

        # Apply attention layers
        x = self.transformer(x)

        # Average the patch representations
        x = x.mean(dim=1)

        return self.classifier(x)


# Create the three models
models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

# Create one sample batch
sample_input = torch.randn(1, 1, 28, 28)

# Test each model
for name, model in models.items():
    model.eval()

    with torch.no_grad():
        output = model(sample_input)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(f"\n{name} Model")
    print("-" * 40)
    print("Output shape:", output.shape)
    print("Total parameters:", parameters)

