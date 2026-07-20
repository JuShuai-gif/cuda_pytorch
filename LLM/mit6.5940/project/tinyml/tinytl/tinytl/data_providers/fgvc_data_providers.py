import os
import torchvision
from ofa.imagenet_classification.data_providers import ImagenetDataProvider

__all__ = [
    "FGVCDataProvider",
    "AircraftDataProvider",
    "CarDataProvider",
    "Flowers102DataProvider",
    "CUB200DataProvider",
    "PetsDataProvider",
    "Food101DataProvider",
    "CIFAR10DataProvider",
    "CIFAR100DataProvider",
]


# FGVC（Fine-Grained Visual Classification，细粒度视觉分类）数据集基类
# 继承自 OFA 的 ImagenetDataProvider，复用其数据加载、增强、批次处理等基础设施
# 每个子类只需提供：数据集名称（name）、类别数量（n_classes）、本地存储路径（save_path）
class FGVCDataProvider(ImagenetDataProvider):
    @staticmethod
    def name():
        """数据集名称标识符，子类必须覆写"""
        raise not NotImplementedError

    @property
    def n_classes(self):
        """分类类别数量，子类必须覆写"""
        raise not NotImplementedError

    @property
    def save_path(self):
        """数据集本地存储路径，子类必须覆写"""
        raise not NotImplementedError


# ---- 细粒度图像分类数据集 ----


class AircraftDataProvider(FGVCDataProvider):
    """FGVC-Aircraft: 飞机类型细粒度分类，100 类"""

    @staticmethod
    def name():
        return "aircraft"

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/aircraft")


class CarDataProvider(FGVCDataProvider):
    """Stanford Cars: 汽车型号细粒度分类，196 类"""

    @staticmethod
    def name():
        return "car"

    @property
    def n_classes(self):
        return 196

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/stanford_car")


class Flowers102DataProvider(FGVCDataProvider):
    """Oxford Flowers-102: 花卉细粒度分类，102 类"""

    @staticmethod
    def name():
        return "flowers102"

    @property
    def n_classes(self):
        return 102

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/flowers102")


class Food101DataProvider(FGVCDataProvider):
    """Food-101: 食物图像分类，101 类"""

    @staticmethod
    def name():
        return "food101"

    @property
    def n_classes(self):
        return 101

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/food101")


class CUB200DataProvider(FGVCDataProvider):
    """CUB-200-2011: 鸟类细粒度分类（经典基准），200 类"""

    @staticmethod
    def name():
        return "cub200"

    @property
    def n_classes(self):
        return 200

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/cub200")


class PetsDataProvider(FGVCDataProvider):
    """Oxford-IIIT Pet: 猫狗品种细粒度分类，37 类"""

    @staticmethod
    def name():
        return "pets"

    @property
    def n_classes(self):
        return 37

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/pets")


# ---- 通用视觉分类数据集 ----
# CIFAR-10/100 不属于细粒度分类，但同样用于评估 TinyTL 的迁移学习效果


class CIFAR10DataProvider(FGVCDataProvider):
    """CIFAR-10: 通用物体分类，10 类（32×32 小图）"""

    @staticmethod
    def name():
        return "cifar10"

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/cifar10")

    # CIFAR 需要使用 torchvision 内置的下载机制（其他数据集需手动下载到 save_path）
    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR10(
            self.save_path, train=True, transform=_transforms, download=True
        )
        return dataset

    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR10(
            self.save_path, train=False, transform=_transforms, download=True
        )
        return dataset


class CIFAR100DataProvider(CIFAR10DataProvider):
    """CIFAR-100: 通用物体分类，100 类（32×32 小图）"""

    @staticmethod
    def name():
        return "cifar100"

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        return os.path.expanduser("~/dataset/cifar100")

    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(
            self.save_path, train=True, transform=_transforms, download=True
        )
        return dataset

    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(
            self.save_path, train=False, transform=_transforms, download=True
        )
        return dataset
