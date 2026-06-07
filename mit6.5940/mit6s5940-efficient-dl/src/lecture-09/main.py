"""
Knowledge Distillation on CIFAR-10 (Lecture 09)

Implements Knowledge Distillation (Hinton et al., 2015) where a larger
teacher CNN transfers knowledge to a smaller student CNN via softened
logits. The student is trained with a combined loss:

    KD_Loss = alpha * CE(student_logits, targets)
            + (1 - alpha) * T^2 * KL(softmax(teacher_logits/T), softmax(student_logits/T))

We compare:
    - Student trained from scratch (baseline, no distillation)
    - Student trained with KD at temperatures T = [1, 2, 4, 8, 16]

All training runs on CPU only.
"""

import copy
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Device configuration – CPU only
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
TEACHER_EPOCHS = 30
STUDENT_EPOCHS = 30
KD_EPOCHS = 30
LEARNING_RATE = 0.001
ALPHA = 0.5  # weight for hard-label CE loss in KD
TEMPERATURES = [1, 2, 4, 8, 16]
NUM_WORKERS = 2  # DataLoader workers (CPU-friendly)


# ===========================================================================
# Model Definitions
# ===========================================================================


class TeacherCNN(nn.Module):
    """Larger teacher network: 4 conv blocks + 2 FC layers."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Block 1: 3 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )
        # Block 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        # Block 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        # Block 4: 128 -> 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4 -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


class StudentCNN(nn.Module):
    """Smaller student network: 3 conv blocks + 1 FC layer."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Block 1: 3 -> 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )
        # Block 2: 16 -> 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        # Block 3: 32 -> 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# Data Loading
# ===========================================================================


def get_cifar10_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader]:
    """Return CIFAR-10 train and test DataLoaders."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


# ===========================================================================
# Training & Evaluation Utilities
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train the model for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate model accuracy on the given loader."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float = LEARNING_RATE,
) -> float:
    """Train a model with standard cross-entropy. Returns final test accuracy."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        acc = evaluate(model, test_loader)
        if acc > best_acc:
            best_acc = acc
    return best_acc


# ===========================================================================
# Knowledge Distillation Training
# ===========================================================================


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    alpha: float,
) -> torch.Tensor:
    """Compute the knowledge-distillation loss.

    KD_loss = alpha * CE(student_logits, labels)
            + (1-alpha) * T^2 * KL(softmax(teacher_logits/T), softmax(student_logits/T))
    """
    # Hard-label cross-entropy
    ce_loss = F.cross_entropy(student_logits, labels)

    # KL divergence between softened distributions.
    # PyTorch F.kl_div(input, target) computes KL(target || exp(input)):
    #   target * (log(target) - input)
    # We want KL(teacher_soft || student_soft), so:
    #   target = teacher probabilities (soft)
    #   input  = student log-probabilities (log_soft)
    teacher_prob = F.softmax(teacher_logits / T, dim=1)
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")

    return alpha * ce_loss + (1.0 - alpha) * (T**2) * kl_loss


def train_kd(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    T: float,
    alpha: float = ALPHA,
    lr: float = LEARNING_RATE,
) -> float:
    """Train the student via knowledge distillation. Returns best test accuracy."""
    teacher.eval()  # teacher is frozen during KD
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch_idx in range(epochs):
        student.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(images)

            optimizer.zero_grad()
            student_logits = student(images)
            loss = kd_loss_fn(student_logits, teacher_logits, labels, T, alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        acc = evaluate(student, test_loader)
        if acc > best_acc:
            best_acc = acc

    return best_acc


# ===========================================================================
# Main
# ===========================================================================


def print_results_table(
    student_scratch_acc: float,
    kd_results: dict,
) -> None:
    """Print results in a clean table format."""
    print("\n" + "=" * 72)
    print("  KNOWLEDGE DISTILLATION RESULTS ON CIFAR-10")
    print("=" * 72)
    print(f"  {'Method':<30} {'Temperature':>12} {'Test Accuracy':>16}")
    print("  " + "-" * 60)
    print(
        f"  {'Student (no KD, from scratch)':<30} {'---':>12} {student_scratch_acc:>15.2f}%"
    )
    for T in sorted(kd_results.keys()):
        print(f"  {'Student (KD)':<30} {f'T={T}':>12} {kd_results[T]:>15.2f}%")
    print("=" * 72)
    print()


def main() -> None:
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders()

    # -- Step 1: train the teacher -------------------------------------------------
    print(f"\nTraining TeacherCNN ({TEACHER_EPOCHS} epochs)...")
    teacher = TeacherCNN(num_classes=10).to(DEVICE)
    t_start = time.time()
    teacher_acc = train_model(teacher, train_loader, test_loader, epochs=TEACHER_EPOCHS)
    t_elapsed = time.time() - t_start
    print(f"Teacher test accuracy: {teacher_acc:.2f}%  (took {t_elapsed:.1f}s)")

    # -- Step 2: train student from scratch (baseline, no KD) -----------------------
    print(f"\nTraining StudentCNN from scratch ({STUDENT_EPOCHS} epochs)...")
    student_scratch = StudentCNN(num_classes=10).to(DEVICE)
    s_start = time.time()
    scratch_acc = train_model(
        student_scratch,
        train_loader,
        test_loader,
        epochs=STUDENT_EPOCHS,
    )
    s_elapsed = time.time() - s_start
    print(f"Student (no KD) test accuracy: {scratch_acc:.2f}%  (took {s_elapsed:.1f}s)")

    # -- Step 3: train student with KD at different temperatures --------------------
    kd_accuracies: dict = {}
    for T in TEMPERATURES:
        print(f"\nTraining StudentCNN with KD at T={T} ({KD_EPOCHS} epochs)...")
        # Fresh student for each temperature
        student_kd = StudentCNN(num_classes=10).to(DEVICE)
        kd_start = time.time()
        acc = train_kd(
            teacher,
            student_kd,
            train_loader,
            test_loader,
            epochs=KD_EPOCHS,
            T=T,
        )
        kd_elapsed = time.time() - kd_start
        kd_accuracies[T] = acc
        print(
            f"Student (KD, T={T}) test accuracy: {acc:.2f}%  (took {kd_elapsed:.1f}s)"
        )

    # -- Step 4: print final results table -----------------------------------------
    print_results_table(scratch_acc, kd_accuracies)


if __name__ == "__main__":
    main()
