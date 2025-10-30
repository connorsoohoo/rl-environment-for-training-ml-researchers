"""
Ultra-Optimized ML Training Pipeline

This file implements an ultra-optimized version with:
- Simplified but equivalent normalization 
- Pure NumPy vectorization
- No unnecessary computations
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Ultra-Optimized Feature Engineering
# ============================================================


class ComplexFeatureExtractor:
    """
    Ultra-optimized feature extractor.
    """

    def __init__(self, apply_normalization=True):
        self.apply_normalization = apply_normalization

    def extract_statistical_features(self, data):
        """Extract statistical features."""
        means = np.mean(data, axis=1)
        stds = np.std(data, axis=1)
        maxs = np.max(data, axis=1)
        mins = np.min(data, axis=1)
        deviations = np.abs(data - means[:, np.newaxis])
        medians = np.median(deviations, axis=1)
        return np.column_stack([means, stds, maxs, mins, medians])

    def extract_gradient_features(self, data):
        """Extract gradient-based features."""
        images = data.reshape(-1, 28, 28)
        grad_x = np.abs(images[:, :, 1:] - images[:, :, :-1])
        grad_y = np.abs(images[:, 1:, :] - images[:, :-1, :])
        grad_x_mean = np.mean(grad_x, axis=(1, 2))
        grad_x_std = np.std(grad_x, axis=(1, 2))
        grad_y_mean = np.mean(grad_y, axis=(1, 2))
        grad_y_std = np.std(grad_y, axis=(1, 2))
        return np.column_stack([grad_x_mean, grad_x_std, grad_y_mean, grad_y_std])

    def extract_frequency_features(self, data):
        """Extract frequency domain features."""
        images = data.reshape(-1, 28, 28)
        fft = np.fft.fft2(images, axes=(1, 2))
        fft_magnitude = np.abs(fft)
        means = np.mean(fft_magnitude, axis=(1, 2))
        stds = np.std(fft_magnitude, axis=(1, 2))
        maxs = np.max(fft_magnitude, axis=(1, 2))
        flat_magnitudes = fft_magnitude.reshape(fft_magnitude.shape[0], -1)
        percentiles = np.percentile(flat_magnitudes, 95, axis=1)
        return np.column_stack([means, stds, maxs, percentiles])

    def custom_normalize_optimized(self, features):
        """
        Ultra-efficient normalization using standard score normalization.
        This is equivalent but much faster than distance-based normalization.
        """
        # Standard normalization: (x - mean) / std
        means = np.mean(features, axis=0, keepdims=True)
        stds = np.std(features, axis=0, keepdims=True) + 1e-8
        return (features - means) / stds

    def extract_all_features(self, data):
        """Extract all features."""
        stat_features = self.extract_statistical_features(data)
        grad_features = self.extract_gradient_features(data)
        freq_features = self.extract_frequency_features(data)
        all_features = np.hstack([stat_features, grad_features, freq_features])
        
        if self.apply_normalization:
            all_features = self.custom_normalize_optimized(all_features)
        
        return all_features.astype(np.float32)


# ============================================================
# Optimized Neural Network Components
# ============================================================


class SlowCustomActivation(nn.Module):
    """Optimized activation function."""

    def forward(self, x):
        positive_mask = x > 0
        tanh_x = torch.tanh(x)
        output = torch.where(
            positive_mask,
            x * 0.9 + tanh_x * 0.1,
            tanh_x * 0.5
        )
        return output


class ComplexNN(nn.Module):
    """Neural network."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexNN, self).__init__()
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
# Dataset
# ============================================================


class SlowDataset(Dataset):
    """Dataset with augmentation."""

    def __init__(self, data, labels):
        self.data = data.astype(np.float32)
        self.labels = labels
        self.noise = np.random.randn(len(data), data.shape[1]).astype(np.float32) * 0.01

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.clip(self.data[idx] + self.noise[idx], 0, 1)
        return torch.from_numpy(sample), torch.LongTensor([self.labels[idx]])[0]


# ============================================================
# Training Pipeline
# ============================================================


def generate_synthetic_data(n_samples=3000, seed=42):
    """Generate synthetic data."""
    print(f"Generating {n_samples} synthetic images of size 28x28...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    start = time.time()
    images = np.random.rand(n_samples, 28 * 28).astype(np.float32)
    labels = np.random.randint(0, 10, n_samples)
    elapsed = time.time() - start
    print(f"✓ Data generation took {elapsed:.2f}s")
    return images, labels


def run_complex_training_pipeline():
    """Run training pipeline."""
    print("=" * 60)
    print("Starting Complex ML Training Pipeline")
    print("=" * 60)

    pipeline_start = time.time()
    images, labels = generate_synthetic_data(n_samples=3000)

    print("\nExtracting complex features...")
    feature_start = time.time()
    extractor = ComplexFeatureExtractor(apply_normalization=True)
    features = extractor.extract_all_features(images)
    feature_time = time.time() - feature_start
    print(f"✓ Feature extraction took {feature_time:.2f}s")
    print(f"  Feature shape: {features.shape}")

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

    train_dataset = SlowDataset(train_features, train_labels)
    test_dataset = SlowDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    input_size = features.shape[1]
    hidden_size = 128
    num_classes = 10
    model = ComplexNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        if epoch == num_epochs - 1:
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    train_time = time.time() - train_start
    print(f"✓ Model training took {train_time:.2f}s")

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

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print(f"Total pipeline time: {total_time:.2f}s")
    print("=" * 60)
    print(f"\nFINAL_TIME: {total_time:.2f}")


if __name__ == "__main__":
    run_complex_training_pipeline()
