# 微型机器学习 (Tiny Machine Learning) [[官网]](https://tinyml.mit.edu)

**[动态]** 我们已将 MCUNet 重构为独立仓库：https://github.com/mit-han-lab/mcunet 。TinyEngine 的后续更新请关注新仓库！

**[动态]** 我们正积极与产业伙伴合作，将 TinyML 技术落地到真实场景中。我们的技术已成功影响多款产品，部署在超过 10 万台物联网设备上。欢迎联系韩松教授了解更多信息。

**[动态]** 我们的项目获得了以下媒体报道： 
[MIT News](https://news.mit.edu/2020/iot-deep-learning-1113), 
[WIRED](https://www.wired.com/story/ai-algorithms-slimming-fit-fridge/), 
[Morning Brew](https://www.morningbrew.com/emerging-tech/stories/2020/12/07/researchers-figured-fit-ai-ever-onto-internet-things-microchips), 
[Stacey on IoT](https://staceyoniot.com/researchers-take-a-3-pronged-approach-to-edge-ai/), 
[Analytics Insight](https://www.analyticsinsight.net/amalgamating-ml-and-iot-in-smart-home-devices/), 
[Techable](https://techable.jp/archives/142462).

## TinyML 项目
| 项目                        | 关键词            |
| --------------------------- | :----------------------: |
| [MCUNet](https://github.com/mit-han-lab/mcunet) | 内存高效推理, 系统-算法协同设计 |
| [TinyTL](https://github.com/mit-han-lab/tinyml/tree/master/tinytl)   | 端侧学习, 内存高效迁移学习    |
| [NetAug](https://github.com/mit-han-lab/tinyml/tree/master/netaug)   | 微型神经网络训练技巧       |


## 关于 TinyML
搭载丰富传感器的智能边缘设备（如数十亿台手机和物联网设备）已经深入我们的日常生活。
将人工智能与这些边缘设备相结合，催生了海量的现实应用，如智能家居、智慧零售、自动驾驶等。
然而，当前最先进的深度学习 AI 系统在训练和推理阶段通常需要巨大的资源：
大规模标注数据集、海量算力、以及大量 AI 专业人才。
这严重阻碍了这些强大 AI 系统在边缘设备上的落地应用。

[TinyML 项目](https://tinyml.mit.edu) 的目标是提升深度学习 AI 系统的效率——用更少的计算、
更少的人工和更少的数据，来推动边缘 AI 和 AIoT（人工智能物联网）的广阔市场。

<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/figures/background1.png" width="100%" />
</p>
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/figures/background2.png" width="100%" />
</p>

## 演示
[![观看视频](https://hanlab.mit.edu/projects/tinyml/figures/mcunet_demo.png)](https://youtu.be/YvioBgtec4U)

## 相关论文

[MCUNet: 物联网设备上的微型深度学习](https://arxiv.org/abs/2007.10319) (NeurIPS'20, spotlight)

[TinyTL: 减少内存而非参数量，实现高效端侧学习](https://arxiv.org/abs/2007.11622) (NeurIPS'20)

[Once for All: 训练一个网络，为高效部署衍生专用子网](https://arxiv.org/abs/1908.09791) (ICLR'20)

[ProxylessNAS: 面向目标任务和硬件的直接神经架构搜索](https://arxiv.org/pdf/1812.00332.pdf) (ICLR'19)

[AutoML: 构建高效专用神经网络的自动化方法](https://ieeexplore.ieee.org/abstract/document/8897011) (IEEE Micro)

[AMC: 移动设备上的 AutoML 模型压缩与加速](https://arxiv.org/pdf/1802.03494.pdf) (ECCV'18)

[HAQ: 硬件感知的自动量化](https://arxiv.org/pdf/1811.08886.pdf) (CVPR'19, oral)
