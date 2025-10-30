"""
Complex ML Training Pipeline - OPTIMIZED

This file implements an optimized ML training pipeline with multiple stages:
- Feature engineering pipeline (vectorized)
- Custom neural network layers (optimized)
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
    OPTIMIZED: Uses vectorized numpy operations instead of loops.
    """

    def __init__(self, apply_normalization=True):
        self.apply_normalization = apply_normalization
        self.feature_cache = {}

    def extract_statistical_features(self, data):
        """Extract statistical features from 2D data - VECTORIZED."""
        # Vectorized approach: compute all stats at once
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        max_val = np.max(data, axis=1, keepdims=True)
        min_val = np.min(data, axis=1, keepdims=True)
        
        # Median absolute deviation
        mad = np.median(np.abs(data - mean), axis=1, keepdims=True)
        
        features = np.hstack([mean, std, max_val, min_val, mad])
        return features

    def extract_gradient_features(self, data):
        """Extract gradient-based features - VECTORIZED."""
        # Reshape to 28x28 images
        img = data.reshape(-1, 28, 28)  # (n_samples, 28, 28)
        
        # Compute gradients using numpy diff (vectorized)
        grad_x = np.abs(np.diff(img, axis=2))  # Gradient in x direction
        grad_y = np.abs(np.diff(img, axis=1))  # Gradient in y direction
        
        # Extract statistics from gradients (vectorized)
        mean_gx = np.mean(grad_x, axis=(1, 2), keepdims=True)
        std_gx = np.std(grad_x, axis=(1, 2), keepdims=True)
        mean_gy = np.mean(grad_y, axis=(1, 2), keepdims=True)
        std_gy = np.std(grad_y, axis=(1, 2), keepdims=True)
        
        # Flatten to (n_samples, 4)
        features = np.hstack([mean_gx.reshape(-1, 1), 
                             std_gx.reshape(-1, 1), 
                             mean_gy.reshape(-1, 1), 
                             std_gy.reshape(-1, 1)])
        return features

    def extract_frequency_features(self, data):
        """Extract frequency domain features - VECTORIZED."""
        # Reshape to 28x28 images
        img = data.reshape(-1, 28, 28)
        
        # Compute FFT for all images at once
        fft = np.fft.fft2(img, axes=(1, 2))
        fft_magnitude = np.abs(fft)
        
        # Extract statistics (vectorized)
        mean_fft = np.mean(fft_magnitude, axis=(1, 2), keepdims=True)
        std_fft = np.std(fft_magnitude, axis=(1, 2), keepdims=True)
        max_fft = np.max(fft_magnitude, axis=(1, 2), keepdims=True)
        
        # Percentile computation vectorized
        fft_flat = fft_magnitude.reshape(fft_magnitude.shape[0], -1)
        perc_95 = np.percentile(fft_flat, 95, axis=1, keepdims=True)
        
        # Flatten and concatenate
        features = np.hstack([mean_fft.reshape(-1, 1), 
                             std_fft.reshape(-1, 1), 
                             max_fft.reshape(-1, 1), 
                             perc_95.reshape(-1, 1)])
        return features

    def custom_normalize(self, features):
        """
        Custom normalization using distance-based scaling - OPTIMIZED.
        Uses vectorized distance computation instead of nested loops.
        """
        # Compute pairwise distances using vectorized operations
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b (for efficiency)
        sq_sum = np.sum(features ** 2, axis=1, keepdims=True)  # (n, 1)
        
        # Compute all pairwise distances at once: ||X - X||^2
        # dist_sq = sq_sum + sq_sum.T - 2 * features @ features.T
        # Then take sqrt to get actual distances
        distances_sq = sq_sum + sq_sum.T - 2.0 * np.dot(features, features.T)
        distances_sq = np.clip(distances_sq, 0, None)  # Clip negative due to numerical errors
        distances = np.sqrt(distances_sq)
        
        # Get median distance for each sample (more efficient than loop)
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
    OPTIMIZED: Uses vectorized torch operations instead of element-wise loops.
    """

    def forward(self, x):
        # Vectorized computation - avoid loops and .item() calls
        positive_mask = x > 0
        output = torch.zeros_like(x)
        
        # For positive values: val * 0.9 + tanh(val) * 0.1
        output[positive_mask] = x[positive_mask] * 0.9 + torch.tanh(x[positive_mask]) * 0.1
        
        # For non-positive values: tanh(val) * 0.5
        output[~positive_mask] = torch.tanh(x[~positive_mask]) * 0.5
        
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
