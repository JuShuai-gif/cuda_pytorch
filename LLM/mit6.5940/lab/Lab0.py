import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

transforms = {
    "train": Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    ),
    "test": ToTensor(),
}

dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="/home/ghr/code/data/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )

samples = [[] for _ in range(10)]
for image, label in dataset["test"]:
    if len(samples[label]) < 4:
        samples[label].append(image)

plt.figure(figsize=(20, 9))
for index in range(40):
    label = index % 10
    image = samples[label][index // 10]

    # Convert from CHW to HWC for visualization
    image = image.permute(1, 2, 0)

    # Convert from class index to class name
    label = dataset["test"].classes[label]

    # Visualize the image
    plt.subplot(4, 10, index + 1)
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")
plt.show()

dataflow = {}
for split in ["train", "test"]:
    dataflow[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
    )

for inputs, targets in dataflow["train"]:
    print("[inputs] dtype: {}, shape: {}".format(inputs.dtype, inputs.shape))
    print("[targets] dtype: {}, shape: {}".format(targets.dtype, targets.shape))
    break


class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != "M":
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x


model = VGG().cuda()

print(model.backbone)

print(model.classifier)

num_params = 0
for param in model.parameters():
    if param.requires_grad:
        num_params += param.numel()
print("#Params:", num_params)

num_macs = profile_macs(model, torch.zeros(1, 3, 32, 32).cuda())
print("#MACs:", num_macs)

criterion = nn.CrossEntropyLoss()

optimizer = SGD(
    model.parameters(),
    lr=0.4,
    momentum=0.9,
    weight_decay=5e-4,
)

num_epochs = 20
steps_per_epoch = len(dataflow["train"])

# Define the piecewise linear scheduler
lr_lambda = lambda step: np.interp(
    [step / steps_per_epoch], [0, num_epochs * 0.3, num_epochs], [0, 1, 0]
)[0]

# Visualize the learning rate schedule
steps = np.arange(steps_per_epoch * num_epochs)
plt.plot(steps, [lr_lambda(step) * 0.4 for step in steps])
plt.xlabel("Number of Steps")
plt.ylabel("Learning Rate")
plt.grid("on")
plt.show()

scheduler = LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    dataflow: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
) -> None:
    model.train()

    for inputs, targets in tqdm(dataflow, desc="train", leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()


@torch.inference_mode()
def evaluate(model: nn.Module, dataflow: DataLoader) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()


for epoch_num in tqdm(range(1, num_epochs + 1)):
    train(model, dataflow["train"], criterion, optimizer, scheduler)
    metric = evaluate(model, dataflow["test"])
    print(f"epoch {epoch_num}:", metric)


plt.figure(figsize=(20, 10))
for index in range(40):
    image, label = dataset["test"][index]

    # Model inference
    model.eval()
    with torch.inference_mode():
        pred = model(image.unsqueeze(dim=0).cuda())
        pred = pred.argmax(dim=1)

    # Convert from CHW to HWC for visualization
    image = image.permute(1, 2, 0)

    # Convert from class indices to class names
    pred = dataset["test"].classes[pred]
    label = dataset["test"].classes[label]

    # Visualize the image
    plt.subplot(4, 10, index + 1)
    plt.imshow(image)
    plt.title(f"pred: {pred}" + "\n" + f"label: {label}")
    plt.axis("off")
plt.show()
