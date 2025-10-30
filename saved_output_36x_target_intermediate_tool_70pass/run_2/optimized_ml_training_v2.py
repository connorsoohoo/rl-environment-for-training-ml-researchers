"""
Complex ML Training Pipeline - OPTIMIZED VERSION V2

This file implements an optimized ML training pipeline with multiple stages:
- Feature engineering pipeline (heavily vectorized)
- Custom neural network layers (vectorized)
- Training loop
- Data preprocessing
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
    Optimized with vectorization and efficient normalization.
    """

    def __init__(self, apply_normalization=True):
        self.apply_normalization = apply_normalization
        self.feature_cache = {}

    def extract_statistical_features(self, data):
        """Extract statistical features from 2D data - VECTORIZED."""
        features = np.column_stack([
            np.mean(data, axis=1),
            np.std(data, axis=1),
            np.max(data, axis=1),
            np.min(data, axis=1),
            np.median(np.abs(data - np.mean(data, axis=1, keepdims=True)), axis=1)
        ])
        return features

    def extract_gradient_features(self, data):
        """Extract gradient-based features - VECTORIZED."""
        images = data.reshape(-1, 28, 28)
        
        # Vectorized gradient computation
        grad_x = np.abs(np.diff(images, axis=2))
        grad_y = np.abs(np.diff(images, axis=1))
        
        # Pad to match dimensions and compute statistics
        grad_x_padded = np.pad(grad_x, ((0, 0), (0, 1), (0, 0)), mode='constant')
        grad_y_padded = np.pad(grad_y, ((0, 0), (0, 0), (0, 1)), mode='constant')
        
        features = np.column_stack([
            np.mean(grad_x_padded, axis=(1, 2)),
            np.std(grad_x_padded, axis=(1, 2)),
            np.mean(grad_y_padded, axis=(1, 2)),
            np.std(grad_y_padded, axis=(1, 2))
        ])
        return features

    def extract_frequency_features(self, data):
        """Extract frequency domain features - VECTORIZED."""
        images = data.reshape(-1, 28, 28)
        
        # Compute FFT for all images at once
        fft = np.fft.fft2(images, axes=(1, 2))
        fft_magnitude = np.abs(fft)
        
        # Vectorized feature extraction
        features = np.column_stack([
            np.mean(fft_magnitude, axis=(1, 2)),
            np.std(fft_magnitude, axis=(1, 2)),
            np.max(fft_magnitude, axis=(1, 2)),
            np.percentile(fft_magnitude.reshape(fft_magnitude.shape[0], -1), 95, axis=1)
        ])
        return features

    def custom_normalize(self, features):
        """
        Custom normalization using distance-based scaling - HEAVILY OPTIMIZED.
        Uses efficient distance computation with minimal memory overhead.
        """
        n_samples = features.shape[0]
        
        # Compute squared norms: ||x||^2 for each sample
        sq_norms = np.sum(features ** 2, axis=1, keepdims=True)  # Shape: (n, 1)
        
        # Compute pairwise squared distances using broadcasting:
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
        dot_product = np.dot(features, features.T)  # Shape: (n, n)
        sq_distances = sq_norms + sq_norms.T - 2 * dot_product
        
        # Clip to avoid numerical errors and compute distances
        sq_distances = np.maximum(sq_distances, 0)
        distances = np.sqrt(sq_distances)
        
        # Compute median distance for each sample
        scales = np.median(distances, axis=1) + 1e-8
        
        # Normalize
        normalized = features / scales[:, np.newaxis]
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
    OPTIMIZED: Vectorized computation instead of element-wise iteration.
    """

    def forward(self, x):
        # Vectorized computation
        pos_mask = x > 0
        output = torch.zeros_like(x)
        
        # For positive values: val * 0.9 + tanh(val) * 0.1
        output[pos_mask] = x[pos_mask] * 0.9 + torch.tanh(x[pos_mask]) * 0.1
        
        # For negative values: tanh(val) * 0.5
        output[~pos_mask] = torch.tanh(x[~pos_mask]) * 0.5
        
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
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Add slight noise for data augmentation
        sample = sample + np.random.randn(*sample.shape) * 0.01
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
