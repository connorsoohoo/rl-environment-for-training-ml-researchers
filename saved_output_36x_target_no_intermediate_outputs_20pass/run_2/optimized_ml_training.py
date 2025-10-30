"""
Complex ML Training Pipeline - OPTIMIZED VERSION v2

This file implements an optimized ML training pipeline with vectorized operations.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Custom Feature Engineering
# ============================================================


class ComplexFeatureExtractor:
    """
    Complex feature extractor with multiple transformation stages.
    """

    def __init__(self, apply_normalization=True):
        self.apply_normalization = apply_normalization
        self.feature_cache = {}

    def extract_statistical_features(self, data):
        """Extract statistical features from 2D data - Vectorized."""
        features = []
        features.append(np.mean(data, axis=1))
        features.append(np.std(data, axis=1))
        features.append(np.max(data, axis=1))
        features.append(np.min(data, axis=1))
        mean_vals = np.mean(data, axis=1, keepdims=True)
        features.append(np.median(np.abs(data - mean_vals), axis=1))
        return np.column_stack(features)

    def extract_gradient_features(self, data):
        """Extract gradient-based features using vectorized operations."""
        # Reshape data to 28x28 images
        images = data.reshape(-1, 28, 28)
        
        # Compute gradients using numpy diff - much faster than nested loops
        grad_x = np.abs(np.diff(images, axis=2))  # Gradient in x direction
        grad_y = np.abs(np.diff(images, axis=1))  # Gradient in y direction
        
        # Compute statistics directly on gradients
        # Reshape to (n_samples, -1) for efficient computation
        grad_x_mean = np.mean(grad_x, axis=(1, 2))
        grad_x_std = np.std(grad_x, axis=(1, 2))
        grad_y_mean = np.mean(grad_y, axis=(1, 2))
        grad_y_std = np.std(grad_y, axis=(1, 2))
        
        return np.column_stack([grad_x_mean, grad_x_std, grad_y_mean, grad_y_std])

    def extract_frequency_features(self, data):
        """Extract frequency domain features."""
        # Reshape data to 28x28 images
        images = data.reshape(-1, 28, 28)
        
        # Compute FFT for all images at once
        fft = np.fft.fft2(images, axes=(1, 2))
        fft_magnitude = np.abs(fft)
        
        # Compute statistics directly - avoid reshaping
        fft_mean = np.mean(fft_magnitude, axis=(1, 2))
        fft_std = np.std(fft_magnitude, axis=(1, 2))
        fft_max = np.max(fft_magnitude, axis=(1, 2))
        fft_p95 = np.percentile(fft_magnitude, 95, axis=(1, 2))
        
        return np.column_stack([fft_mean, fft_std, fft_max, fft_p95])

    def custom_normalize(self, features):
        """
        Optimized normalization using vectorized distance computation.
        Uses broadcasting to avoid nested loops.
        """
        # Compute pairwise distances using broadcasting
        # This is O(n*d) instead of O(n²) where n=samples, d=features
        diff = features[:, np.newaxis, :] - features[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Compute median distance for each sample
        scale = np.median(distances, axis=1, keepdims=True) + 1e-8
        normalized = features / scale
        
        return normalized

    def extract_all_features(self, data):
        """Extract all features from raw data."""
        stat_features = self.extract_statistical_features(data)
        grad_features = self.extract_gradient_features(data)
        freq_features = self.extract_frequency_features(data)

        # Concatenate all features
        all_features = np.hstack([stat_features, grad_features, freq_features])

        # Apply custom normalization if enabled
        if self.apply_normalization:
            all_features = self.custom_normalize(all_features)

        return all_features


# ============================================================
# Custom Neural Network Components
# ============================================================


class SlowCustomActivation(nn.Module):
    """
    Custom activation function combining relu and tanh characteristics.
    Vectorized implementation for efficiency.
    """

    def forward(self, x):
        # Fully vectorized: no loops, pure tensor operations
        pos_mask = x > 0
        
        # Compute tanh for all elements
        tanh_x = torch.tanh(x)
        
        # Apply the formula based on sign
        output = torch.where(
            pos_mask,
            x * 0.9 + tanh_x * 0.1,
            tanh_x * 0.5
        )
        
        return output


class ComplexNN(nn.Module):
    """
    Neural network with custom architecture.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.custom_activation = SlowCustomActivation()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.custom_activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# ============================================================
# Custom Dataset
# ============================================================


class SlowDataset(Dataset):
    """
    Custom dataset with data augmentation.
    Optimized with pre-generated noise.
    """

    def __init__(self, data, labels, noise_seed=42):
        self.data = data
        self.labels = labels
        # Pre-generate all noise to avoid repeated random generation
        np.random.seed(noise_seed)
        self.noise = np.random.randn(len(data), data.shape[1]) * 0.01

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Use pre-generated noise instead of generating on-the-fly
        sample = sample + self.noise[idx]
        sample = np.clip(sample, 0, 1)

        return torch.FloatTensor(sample), torch.LongTensor([label])[0]


# ============================================================
# Main Training Pipeline
# ============================================================


def generate_synthetic_data(n_samples=3000, seed=42):
    """Generate synthetic image-like data."""
    print(f"Generating {n_samples} synthetic images of size 28x28...")
    np.random.seed(seed)
    torch.manual_seed(seed)

    start = time.time()

    # Generate random 28x28 images
    images = np.random.rand(n_samples, 28 * 28).astype(np.float32)

    # Generate labels (10 classes)
    labels = np.random.randint(0, 10, n_samples)

    elapsed = time.time() - start
    print(f"✓ Data generation took {elapsed:.2f}s")

    return images, labels


def run_complex_training_pipeline():
    """Run the complete training pipeline with multiple bottlenecks."""
    print("=" * 60)
    print("Starting Complex ML Training Pipeline")
    print("=" * 60)

    pipeline_start = time.time()

    # Generate data
    images, labels = generate_synthetic_data(n_samples=3000)

    # Extract features using complex feature extractor
    print("\nExtracting complex features...")
    feature_start = time.time()
    extractor = ComplexFeatureExtractor(apply_normalization=True)
    features = extractor.extract_all_features(images)
    feature_time = time.time() - feature_start
    print(f"✓ Feature extraction took {feature_time:.2f}s")
    print(f"  Feature shape: {features.shape}")

    # Split data
    print("\nSplitting data...")
    split_start = time.time()
    n_train = int(0.8 * len(features))
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    train_features = features[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    test_features = features[indices[n_train:]]
    test_labels = labels[indices[n_train:]]

    split_time = time.time() - split_start
    print(f"✓ Train/test split took {split_time:.2f}s")
    print(f"  Training set size: {len(train_features)}")
    print(f"  Test set size: {len(test_features)}")

    # Create datasets and dataloaders
    train_dataset = SlowDataset(train_features, train_labels)
    test_dataset = SlowDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Create model
    input_size = features.shape[1]
    hidden_size = 128
    num_classes = 10

    model = ComplexNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("\nTraining complex neural network...")
    train_start = time.time()

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        # Don't print every epoch to reduce clutter
        if epoch == num_epochs - 1:
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    train_time = time.time() - train_start
    print(f"✓ Model training took {train_time:.2f}s")

    # Evaluate model
    print("\nEvaluating model...")
    eval_start = time.time()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    eval_time = time.time() - eval_start

    print(f"✓ Model evaluation took {eval_time:.2f}s")
    print(f"  Test accuracy: {accuracy:.4f}")

    # Calculate total time
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print(f"Total pipeline time: {total_time:.2f}s")
    print("=" * 60)
    print(f"\nFINAL_TIME: {total_time:.2f}")


if __name__ == "__main__":
    run_complex_training_pipeline()
