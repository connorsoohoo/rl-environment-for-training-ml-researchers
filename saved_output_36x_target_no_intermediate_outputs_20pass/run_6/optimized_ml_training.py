"""
Complex ML Training Pipeline - Optimized Version

Optimizations made:
1. Vectorized custom_normalize using broadcasting and matrix operations
2. Vectorized extract_gradient_features using numpy operations
3. Vectorized SlowCustomActivation using torch operations instead of loops
4. Other minor optimizations for consistency
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
        """Extract statistical features from 2D data - vectorized."""
        features = np.zeros((data.shape[0], 5), dtype=np.float32)
        features[:, 0] = np.mean(data, axis=1)  # mean
        features[:, 1] = np.std(data, axis=1)   # std
        features[:, 2] = np.max(data, axis=1)   # max
        features[:, 3] = np.min(data, axis=1)   # min
        # Median absolute deviation - vectorized
        deviations = np.abs(data - np.mean(data, axis=1, keepdims=True))
        features[:, 4] = np.median(deviations, axis=1)
        return features

    def extract_gradient_features(self, data):
        """Extract gradient-based features - vectorized."""
        n_samples = data.shape[0]
        features = np.zeros((n_samples, 4), dtype=np.float32)
        
        # Reshape all images at once
        img_reshaped = data.reshape(n_samples, 28, 28)
        
        # Compute gradients for all samples at once
        # Gradient in x direction
        grad_x = np.abs(np.diff(img_reshaped[:, :, :27], axis=2))
        # Gradient in y direction
        grad_y = np.abs(np.diff(img_reshaped[:, :27, :], axis=1))
        
        # Calculate features from gradients
        features[:, 0] = np.mean(grad_x, axis=(1, 2))
        features[:, 1] = np.std(grad_x, axis=(1, 2))
        features[:, 2] = np.mean(grad_y, axis=(1, 2))
        features[:, 3] = np.std(grad_y, axis=(1, 2))
        
        return features

    def extract_frequency_features(self, data):
        """Extract frequency domain features - partially vectorized."""
        features = []
        for i in range(data.shape[0]):
            img = data[i].reshape(28, 28)
            fft = np.fft.fft2(img)
            fft_magnitude = np.abs(fft)
            features.append(
                [
                    np.mean(fft_magnitude),
                    np.std(fft_magnitude),
                    np.max(fft_magnitude),
                    np.percentile(fft_magnitude.flatten(), 95),
                ]
            )
        return np.array(features, dtype=np.float32)

    def custom_normalize(self, features):
        """
        Custom normalization using distance-based scaling - vectorized.
        """
        n_samples = features.shape[0]
        
        # Compute pairwise distances using broadcasting
        # features shape: (n_samples, n_features)
        # We want distances: (n_samples, n_samples)
        # Using: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        
        sq_norms = np.sum(features ** 2, axis=1, keepdims=True)  # (n_samples, 1)
        distances_sq = sq_norms + sq_norms.T - 2 * (features @ features.T)
        distances_sq = np.maximum(distances_sq, 0)  # Avoid numerical errors
        distances = np.sqrt(distances_sq)
        
        # Calculate scale for each sample
        scales = np.median(distances, axis=1) + 1e-8
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
    Optimized with vectorized operations.
    """

    def forward(self, x):
        # Vectorized version using torch operations
        pos_mask = x > 0
        output = torch.zeros_like(x)
        output[pos_mask] = x[pos_mask] * 0.9 + torch.tanh(x[pos_mask]) * 0.1
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
