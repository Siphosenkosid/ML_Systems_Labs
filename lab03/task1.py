
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Set a seed to make the dataset split reproducible
torch.manual_seed(42)

# Step 1: Define the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Step 2: Download the Fashion-MNIST datasets
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Step 3: Split the training dataset
# 55,000 images for training and 5,000 for validation
train_data, val_data = random_split(
    train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

# Step 4: Create DataLoaders
batch_size = 128

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Step 5: Display dataset information
print("Fashion-MNIST Dataset")
print("-" * 40)
print("Training samples:", len(train_data))
print("Validation samples:", len(val_data))
print("Testing samples:", len(test_dataset))
print("Batch size:", batch_size)

# Step 6: Check the shape of one batch
images, labels = next(iter(train_loader))

print("\nBatch Information")
print("-" * 40)
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
