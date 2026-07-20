from ofa.imagenet_classification.run_manager import ImagenetRunConfig

from .fgvc_data_providers import (
    AircraftDataProvider,
    Flowers102DataProvider,
    CarDataProvider,
)
from .fgvc_data_providers import (
    Food101DataProvider,
    CUB200DataProvider,
    PetsDataProvider,
)
from .fgvc_data_providers import CIFAR10DataProvider, CIFAR100DataProvider

__all__ = ["FGVCRunConfig"]


# FGVC 数据集的运行配置管理器
# 继承自 OFA 的 ImagenetRunConfig，复用其训练/验证/测试的数据加载、学习率调度等基础设施
# 扩展功能：(1) 支持 8 种 FGVC/通用数据集的数据提供器按名称分发；
#          (2) fast_evaluation 模式将验证集和测试集完整加载到内存，加速评估
class FGVCRunConfig(ImagenetRunConfig):
    def __init__(
        self,
        n_epochs=50,
        init_lr=0.01,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="flowers102",
        train_batch_size=256,
        test_batch_size=500,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0,
        no_decay_keys=None,
        mixup_alpha=None,
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=32,
        resize_scale=0.08,
        distort_color="tf",
        image_size=224,
        fast_evaluation=True,
        **kwargs,
    ):
        super(FGVCRunConfig, self).__init__(
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
            n_worker,
            resize_scale,
            distort_color,
            image_size,
            **kwargs,
        )
        # 快速评估模式：将验证/测试数据一次性加载到内存，避免每个 epoch 反复读磁盘
        # TinyTL 实验中数据集规模不大（通常几千到几万张），加载到内存可行且大幅加速评估
        self.fast_evaluation = fast_evaluation

    @property
    def data_provider(self):
        """根据 dataset 名称字符串延迟创建对应的 DataProvider 实例。

        支持 8 种数据集：aircraft、flowers102、car、food101、cub200、pets、cifar10、cifar100。
        结果缓存在 _data_provider 中以避免重复创建。
        """
        if self.__dict__.get("_data_provider", None) is None:
            # 按名称匹配 DataProvider 类
            if self.dataset == AircraftDataProvider.name():
                DataProviderClass = AircraftDataProvider
            elif self.dataset == Flowers102DataProvider.name():
                DataProviderClass = Flowers102DataProvider
            elif self.dataset == CarDataProvider.name():
                DataProviderClass = CarDataProvider
            elif self.dataset == Food101DataProvider.name():
                DataProviderClass = Food101DataProvider
            elif self.dataset == CUB200DataProvider.name():
                DataProviderClass = CUB200DataProvider
            elif self.dataset == PetsDataProvider.name():
                DataProviderClass = PetsDataProvider
            elif self.dataset == CIFAR10DataProvider.name():
                DataProviderClass = CIFAR10DataProvider
            elif self.dataset == CIFAR100DataProvider.name():
                DataProviderClass = CIFAR100DataProvider
            else:
                raise ValueError("Do not support: %s" % self.dataset)

            # 创建 DataProvider 实例并缓存
            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
            )
        return self.__dict__["_data_provider"]

    @property
    def valid_loader(self):
        """验证集数据加载器。

        快速评估模式（fast_evaluation=True）：将验证集所有 (images, labels) 对一次性加载到内存列表，
        后续每个 epoch 遍历内存列表即可，避免反复读磁盘和图像解码。
        缓存 key 包含 active_img_size 以区分不同图像尺寸的加载结果。
        """
        if not self.fast_evaluation:
            # 非快速模式：每次从磁盘动态加载
            return self.data_provider.valid

        # valid_size=None 时直接回退到测试集的快速加载逻辑
        if self.valid_size is None:
            return self.test_loader

        # 检查缓存，未命中则加载全部验证数据到内存
        if (
            self.__dict__.get(
                "_in_memory_valid%d" % self.data_provider.active_img_size, None
            )
            is None
        ):
            self.__dict__[
                "_in_memory_valid%d" % self.data_provider.active_img_size
            ] = []
            for images, labels in self.data_provider.valid:
                self.__dict__[
                    "_in_memory_valid%d" % self.data_provider.active_img_size
                ].append((images, labels))
        return self.__dict__["_in_memory_valid%d" % self.data_provider.active_img_size]

    @property
    def test_loader(self):
        """测试集数据加载器。

        与 valid_loader 逻辑相同：快速模式时将测试集完整缓存到内存。
        """
        if not self.fast_evaluation:
            return self.data_provider.test

        if (
            self.__dict__.get(
                "_in_memory_test%d" % self.data_provider.active_img_size, None
            )
            is None
        ):
            self.__dict__["_in_memory_test%d" % self.data_provider.active_img_size] = []
            for images, labels in self.data_provider.test:
                self.__dict__[
                    "_in_memory_test%d" % self.data_provider.active_img_size
                ].append((images, labels))
        return self.__dict__["_in_memory_test%d" % self.data_provider.active_img_size]
