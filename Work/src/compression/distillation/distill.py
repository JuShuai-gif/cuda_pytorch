"""Knowledge distillation: teacher -> student with soft (logit) targets.

The industrial motivation: a large model (teacher) is too slow/big for the
edge, so we train a small model (student) to imitate it.  The key trick is
that the teacher's *soft logits* carry more information than one-hot labels -
they encode "a 3 looks a bit like an 8" - so a temperature-softened KL loss
teaches the student a richer decision boundary than hard labels alone.

Uses MNIST (the classic Hinton distillation benchmark).
"""

from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, n_classes: int):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [n_classes]
        self.net = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.net.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.net:
            x = m(x)
        return x


def _mnist_path() -> str:
    return os.environ.get("MNIST_DIR", "/tmp/opencode/mnist")


def load_mnist(device, batch: int = 256):
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), lambda t: t.view(-1)])
    train = datasets.MNIST(_mnist_path(), train=True, download=True, transform=transform)
    test = datasets.MNIST(_mnist_path(), train=False, download=True, transform=transform)

    x = train.data.float().view(-1, 784) / 255.0
    y = train.targets
    xt = test.data.float().view(-1, 784) / 255.0
    yt = test.targets

    train_dl = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)
    return train_dl, xt.to(device), yt.to(device)


def _step(model, xb, yb, opt, *, teacher_logits=None, T=4.0, alpha=0.9):
    opt.zero_grad()
    logits = model(xb)
    if teacher_logits is None:
        loss = F.cross_entropy(logits, yb)
    else:
        soft = F.kl_div(
            F.log_softmax(logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * T * T
        hard = F.cross_entropy(logits, yb)
        loss = alpha * soft + (1 - alpha) * hard
    loss.backward()
    opt.step()


def train_model(model, train_dl, *, epochs, device, lr=1e-3,
                teacher=None, T=4.0, alpha=0.9):
    """Train on mini-batches; if teacher is given, use logit distillation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            teacher_logits = None
            if teacher is not None:
                teacher.eval()
                with torch.no_grad():
                    teacher_logits = teacher(xb)
            _step(model, xb, yb, opt, teacher_logits=teacher_logits, T=T, alpha=alpha)
    return model


def accuracy(model, x, y, device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
    return (pred == y).float().mean().item()


def _few_shot_dl(train_dl, n: int):
    x = train_dl.dataset.tensors[0][:n]
    y = train_dl.dataset.tensors[1][:n]
    return DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)


def run_distillation(device, seed: int = 0, n_train=2000) -> dict:
    torch.manual_seed(seed)
    train_dl, xt, yt = load_mnist(device)
    few_dl = _few_shot_dl(train_dl, n_train)

    # Teacher trains on the full dataset (it is the expensive, high-accuracy
    # model we cannot deploy to the edge).
    teacher = MLP(784, 512, 3, 10).to(device)
    train_model(teacher, train_dl, epochs=6, device=device)
    teacher_acc = accuracy(teacher, xt, yt, device)

    # Students train on few data (the realistic edge-training scenario where
    # the teacher's soft labels act as a regularizer / data augmentation).
    student = MLP(784, 256, 2, 10).to(device)
    train_model(student, few_dl, epochs=40, device=device)
    direct_acc = accuracy(student, xt, yt, device)

    student_distilled = MLP(784, 256, 2, 10).to(device)
    train_model(student_distilled, few_dl, epochs=40, device=device,
                teacher=teacher, T=4.0, alpha=0.9)
    distilled_acc = accuracy(student_distilled, xt, yt, device)

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    return {
        "n_train": n_train,
        "teacher_acc": teacher_acc,
        "student_direct_acc": direct_acc,
        "student_distilled_acc": distilled_acc,
        "teacher_params": n_params(teacher),
        "student_params": n_params(student),
    }


def temperature_sweep(device, seed: int = 0, n_train=2000) -> list[dict]:
    torch.manual_seed(seed)
    train_dl, xt, yt = load_mnist(device)
    few_dl = _few_shot_dl(train_dl, n_train)

    teacher = MLP(784, 512, 3, 10).to(device)
    train_model(teacher, train_dl, epochs=6, device=device)

    out = []
    for T in [1.0, 2.0, 4.0, 8.0, 20.0]:
        s = MLP(784, 256, 2, 10).to(device)
        train_model(s, few_dl, epochs=40, device=device, teacher=teacher, T=T)
        out.append({"T": T, "accuracy": accuracy(s, xt, yt, device)})
    return out
