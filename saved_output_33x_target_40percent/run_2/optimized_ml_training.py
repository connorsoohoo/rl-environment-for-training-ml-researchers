"""
Optimized ML Training Pipeline

This file implements an optimized version of the ML training pipeline with:
- Vectorized feature engineering (NumPy operations)
- Efficient custom activation function
- Better normalization using StandardScaler
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ============================================================
# Optimized Feature Engineering
# ============================================================


class OptimizedFeatureExtractor:
    """
    Optimized feature extractor with vectorized operations.
    """

    def __init__(self, apply_normalization=True):
        self.apply_normalization = apply_normalization
        self.scaler = StandardScaler()

    def extract_statistical_features(self, data):
        """Extract statistical features from 2D data using vectorization."""
        features = np.column_stack([
            np.mean(data, axis=1),
            np.std(data, axis=1),
            np.max(data, axis=1),
            np.min(data, axis=1),
            np.median(np.abs(data - np.mean(data, axis=1, keepdims=True)), axis=1)
        ])
        return features

    def extract_gradient_features(self, data):
        """Extract gradient-based features using vectorized operations."""
        n_samples = data.shape[0]
        features = []
        
        for i in range(n_samples):
            img = data[i].reshape(28, 28)
            # Compute gradients using numpy diff
            grad_x = np.abs(np.diff(img, axis=1))
            grad_y = np.abs(np.diff(img, axis=0))
            
            features.append([
                np.mean(grad_x),
                np.std(grad_x),
                np.mean(grad_y),
                np.std(grad_y)
            ])
        
        return np.array(features)

    def extract_frequency_features(self, data):
        """Extract frequency domain features using vectorization."""
        n_samples = data.shape[0]
        features = []
        
        for i in range(n_samples):
            img = data[i].reshape(28, 28)
            fft = np.fft.fft2(img)
            fft_magnitude = np.abs(fft)
            
            features.append([
                np.mean(fft_magnitude),
                np.std(fft_magnitude),
                np.max(fft_magnitude),
                np.percentile(fft_magnitude.flatten(), 95)
            ])
        
        return np.array(features)

    def optimized_normalize(self, features):
        """
        Use StandardScaler for fast normalization instead of distance-based scaling.
        """
        # StandardScaler is much faster than custom distance-based normalization
        normalized = self.scaler.fit_transform(features)
        return normalized

    def extract_all_features(self, data):
        """Extract all features from raw data using optimized methods."""
        stat_features = self.extract_statistical_features(data)
        grad_features = self.extract_gradient_features(data)
        freq_features = self.extract_frequency_features(data)

        # Concatenate all features
        all_features = np.hstack([stat_features, grad_features, freq_features])

        # Apply optimized normalization if enabled
        if self.apply_normalization:
            all_features = self.optimized_normalize(all_features)

        return all_features


# ============================================================
# Optimized Neural Network Components
# ============================================================


class OptimizedActivation(nn.Module):
    """
    Optimized activation function using vectorized operations.
    """

    def forward(self, x):
        # Vectorized operation instead of element-wise loop
        positive_mask = x > 0
        output = torch.zeros_like(x)
        output[positive_mask] = x[positive_mask] * 0.9 + torch.tanh(x[positive_mask]) * 0.1
        output[~positive_mask] = torch.tanh(x[~positive_mask]) * 0.5
        return output


class OptimizedNN(nn.Module):
    """
    Neural network with optimized architecture.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(OptimizedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = OptimizedActivation()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# ============================================================
# Optimized Dataset
# ============================================================


class OptimizedDataset(Dataset):
    """
    Optimized dataset with pre-computed augmented data.
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


def run_optimized_training_pipeline():
    """Run the optimized training pipeline."""
    print("=" * 60)
    print("Starting Optimized ML Training Pipeline")
    print("=" * 60)

    pipeline_start = time.time()

    # Generate data
    images, labels = generate_synthetic_data(n_samples=3000)

    # Extract features using optimized feature extractor
    print("\nExtracting optimized features...")
    feature_start = time.time()
    extractor = OptimizedFeatureExtractor(apply_normalization=True)
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
    train_dataset = OptimizedDataset(train_features, train_labels)
    test_dataset = OptimizedDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Create model
    input_size = features.shape[1]
    hidden_size = 128
    num_classes = 10

    model = OptimizedNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("\nTraining optimized neural network...")
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
    run_optimized_training_pipeline()
