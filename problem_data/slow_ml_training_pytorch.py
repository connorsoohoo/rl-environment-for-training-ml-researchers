"""
ML training pipeline for image classification using PyTorch.

This script trains a simple neural network classifier on synthetic image data.
Contains intentional inefficiencies for optimization practice.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def generate_synthetic_images(n_samples: int, img_size: int = 28):
    """Generate synthetic grayscale images (like MNIST)."""
    print(f"Generating {n_samples} synthetic images of size {img_size}x{img_size}...")
    images = []
    labels = []

    for _ in range(n_samples):
        # Generate a random image
        img = np.random.rand(img_size, img_size)
        # Simple pattern: if mean > 0.5, label as 1, else 0
        label = 1 if np.mean(img) > 0.5 else 0
        images.append(img)
        labels.append(label)

    return images, labels


def extract_features_slow(images):
    """Extract features from images (inefficiently)."""
    print("Extracting features...")
    features_list = []

    for img in images:
        # Convert to numpy if needed
        if isinstance(img, list):
            img = np.array(img)

        # Extract multiple features inefficiently
        feature_vector = []

        # Feature 1: Mean pixel value (calculated inefficiently)
        pixel_sum = 0
        pixel_count = 0
        for row in img:
            for pixel in row:
                pixel_sum += pixel
                pixel_count += 1
        mean_val = pixel_sum / pixel_count
        feature_vector.append(mean_val)

        # Feature 2: Std deviation (calculated inefficiently)
        variance_sum = 0
        for row in img:
            for pixel in row:
                variance_sum += (pixel - mean_val) ** 2
        std_val = (variance_sum / pixel_count) ** 0.5
        feature_vector.append(std_val)

        # Feature 3: Max value (inefficient)
        max_val = 0
        for row in img:
            for pixel in row:
                if pixel > max_val:
                    max_val = pixel
        feature_vector.append(max_val)

        # Feature 4: Min value (inefficient)
        min_val = 1.0
        for row in img:
            for pixel in row:
                if pixel < min_val:
                    min_val = pixel
        feature_vector.append(min_val)

        # Feature 5-8: Quadrant means (inefficient)
        h, w = img.shape
        mid_h, mid_w = h // 2, w // 2

        # Top-left quadrant
        q1_sum = 0
        q1_count = 0
        for i in range(mid_h):
            for j in range(mid_w):
                q1_sum += img[i, j]
                q1_count += 1
        feature_vector.append(q1_sum / q1_count if q1_count > 0 else 0)

        # Top-right quadrant
        q2_sum = 0
        q2_count = 0
        for i in range(mid_h):
            for j in range(mid_w, w):
                q2_sum += img[i, j]
                q2_count += 1
        feature_vector.append(q2_sum / q2_count if q2_count > 0 else 0)

        # Bottom-left quadrant
        q3_sum = 0
        q3_count = 0
        for i in range(mid_h, h):
            for j in range(mid_w):
                q3_sum += img[i, j]
                q3_count += 1
        feature_vector.append(q3_sum / q3_count if q3_count > 0 else 0)

        # Bottom-right quadrant
        q4_sum = 0
        q4_count = 0
        for i in range(mid_h, h):
            for j in range(mid_w, w):
                q4_sum += img[i, j]
                q4_count += 1
        feature_vector.append(q4_sum / q4_count if q4_count > 0 else 0)

        features_list.append(feature_vector)

    return np.array(features_list)


def normalize_features_slow(features):
    """Normalize features to zero mean and unit variance (inefficiently)."""
    print("Normalizing features...")
    normalized = np.zeros_like(features)

    # For each feature dimension
    for col_idx in range(features.shape[1]):
        # Calculate mean
        col_sum = 0
        for row_idx in range(features.shape[0]):
            col_sum += features[row_idx, col_idx]
        col_mean = col_sum / features.shape[0]

        # Calculate std
        variance_sum = 0
        for row_idx in range(features.shape[0]):
            variance_sum += (features[row_idx, col_idx] - col_mean) ** 2
        col_std = (variance_sum / features.shape[0]) ** 0.5

        # Normalize
        for row_idx in range(features.shape[0]):
            if col_std > 0:
                normalized[row_idx, col_idx] = (
                    features[row_idx, col_idx] - col_mean
                ) / col_std
            else:
                normalized[row_idx, col_idx] = 0

    return normalized


class SimpleNN(nn.Module):
    """Simple neural network classifier."""

    def __init__(self, input_size, hidden_size=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model_slow(X_train, y_train, X_test, y_test, n_features):
    """Train a PyTorch neural network (inefficiently)."""
    print("Training Neural Network model...")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create model
    model = SimpleNN(input_size=n_features, hidden_size=32, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train for a few epochs (inefficiently - one sample at a time)
    n_epochs = 20
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        # INEFFICIENT: Train one sample at a time instead of batching
        for i in range(len(X_train_tensor)):
            # Get single sample
            inputs = X_train_tensor[i : i + 1]
            labels = y_train_tensor[i : i + 1]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(X_train_tensor)
            print(f"  Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())

    return model, accuracy


def run_training_pipeline():
    """Run the complete training pipeline and return timing results."""
    print("=" * 60)
    print("Starting ML training pipeline (PyTorch Version)")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Generate data (2000 samples)
    step_start = time.time()
    images, labels = generate_synthetic_images(n_samples=2000, img_size=28)
    labels = np.array(labels)
    step1_time = time.time() - step_start
    print(f"✓ Data generation took {step1_time:.2f}s\n")

    # Step 2: Extract features (8 features)
    step_start = time.time()
    features = extract_features_slow(images)
    step2_time = time.time() - step_start
    print(f"✓ Feature extraction took {step2_time:.2f}s")
    print(f"  Feature shape: {features.shape}\n")

    # Step 3: Normalize features
    step_start = time.time()
    features_normalized = normalize_features_slow(features)
    step3_time = time.time() - step_start
    print(f"✓ Feature normalization took {step3_time:.2f}s\n")

    # Step 4: Train/test split
    step_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        features_normalized, labels, test_size=0.2, random_state=42
    )
    step4_time = time.time() - step_start
    print(f"✓ Train/test split took {step4_time:.2f}s")
    print(f"  Training set size: {len(X_train)}")
    print(f"  Test set size: {len(X_test)}\n")

    # Step 5: Train model and evaluate
    step_start = time.time()
    model, accuracy = train_model_slow(
        X_train, y_train, X_test, y_test, n_features=features.shape[1]
    )
    step5_time = time.time() - step_start
    print(f"✓ Model training took {step5_time:.2f}s\n")

    print("✓ Model evaluation took 0.00s")
    print(f"  Test accuracy: {accuracy:.4f}\n")

    total_time = time.time() - start_time

    print("=" * 60)
    print(f"Total pipeline time: {total_time:.2f}s")
    print("=" * 60)

    return total_time


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Run the pipeline and print the total time
    total_time = run_training_pipeline()
    print(f"\nFINAL_TIME: {total_time:.2f}")
