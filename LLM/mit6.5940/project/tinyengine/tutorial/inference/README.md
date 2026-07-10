# TinyEngine：推理演示教程代码 (视觉唤醒词)

这是利用 **TinyEngine** 在 STM32F746G-DISCO 探索板上部署视觉唤醒词 (Visual Wake Words, VWW) 推理模型的官方演示教程。

## 代码结构

（完成下方的[详细步骤](#详细步骤)后，代码结构如下）

```
.
├── ...
├── Debug                                  # 调试文件夹
│   ├── TinyEngine_vww_tutorial.elf        # 将在 MCU 上运行的 ELF 可执行文件
│   └── ...
├── Drivers                                # 驱动文件夹
├── Inc                                    # 头文件文件夹
├── Src                                    # 源文件文件夹
│   ├── main.cpp                           # 主源文件
│   ├── TinyEngine                         # TinyEngine 文件夹
│   │   ├── codegen                        # 代码生成文件夹
│   │   │   ├── Include
│   │   │   │   ├── genModel.h             # 内存分配头文件
│   │   │   │   └── ...
│   │   │   └── Source
│   │   │       ├── genModel.c             # 计算图的代码生成
│   │   │       └── ...
│   │   ├── include
│   │   └── src/kernels
│   │       ├── fp_requantize_op           # 浮点重量化算子
│   │       ├── fp_backward_op             # 反向传播 FP32 算子
│   │       └── int_forward_op             # 前向传播 INT 算子
│   └── ...
└── ...
```

## 所需硬件

1. STM32F746G-DISCO 探索板
2. Arducam Shield Mini 2MP Plus *(可选)*
3. 公对母杜邦线 (x8) *(可选)*

## 使用概览

1. 下载并安装 STM32CubeIDE 1.5.0 版本。
2. 下载并将本项目导入你的 STM32CubeIDE。
3. 用杜邦线将 Arducam 连接到开发板。*(可选)*
4. 编译并将程序烧录到你的 STM32F746G-DISCO 探索板。
5. 完成！在演示中，STM32F746G-DISCO 探索板的 LCD 屏幕将显示人物检测结果（有人/无人）和每秒帧数（FPS）。

## 详细步骤

0. 准备一块 STM32F746G-DISCO 探索板（以及一个 Arducam，如果使用的话）。
1. 下载 STM32CubeIDE，一款面向 STM32 微控制器和微处理器的 C/C++ 开发平台，集成了外设配置、代码生成、代码编译和调试功能。

- 请下载 STM32CubeIDE **1.5.0 版本**。\[[下载链接](https://www.st.com/en/development-tools/stm32cubeide.html#get-software)\]
- 更详细的安装和使用指南请参考 "UM2563 STM32CubeIDE 安装指南" 和 "UM2553 STM32CubeIDE 快速入门指南"。\[[文档链接](https://www.st.com/en/development-tools/stm32cubeide.html#documentation)\]

2. 准备代码库。

- 首先，请按照 [`面向用户的设置`](https://github.com/mit-han-lab/tinyengine#setup-for-users) 中的说明配置环境。
- 复制 `tutorial/inference` 文件夹并重命名为 `TinyEngine_vww_tutorial`（供后续在 STM32CubeIDE 中使用）。

```bash
cp -r ./tutorial/inference ./tutorial/TinyEngine_vww_tutorial
```

- 设置 PYTHONPATH，然后运行 VWW 的代码生成示例：

```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)
python examples/vww.py  # 如需使用基于 patch 的推理，请运行 `example/vww_patchbased.py`
```

- 将新生成的 `codegen` 文件夹移动到以下路径：

```bash
mkdir ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine
mv codegen ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine
```

- 将 `TinyEngine` 文件夹复制到以下路径：

```bash
cp -r ./TinyEngine/include ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine
cp -r ./TinyEngine/src ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine
```

- 使用以下 Shell 脚本将所需的 Arm 文件复制到正确路径：

```bash
bash import_arm_inference.sh
```

3. 在 STM32CubeIDE 中配置编译和运行。

- 将 `TinyEngine_vww_tutorial` 代码库导入 STM32CubeIDE：通过 \[File\] -> \[Import...\] -> \[General\] -> \[Existing Projects into Workspace\]（导入整个 `TinyEngine_vww_tutorial` 文件夹）。

<img src="../../assets/figures/0_import_project_0.png" alt="0_import_project_0" width="47%"/>  <img src="../../assets/figures/1_import_project_1.png" alt="1_import_project_1" width="46%"/>

- 导入完成后，`TinyEngine_vww_tutorial` 应出现在 STM32CubeIDE 的 Project Explorer 中，如下图所示：

<img src="../../assets/figures/2_project_explorer.png" alt="2_project_explorer" width="30%"/>

- 打开 `TinyEngine_vww_tutorial/Src/main.cpp`。
  
  - 如果使用 Arducam，请将 `UseCamera` 宏设置为 1（第 32 行），如下图所示：

  <img src="../../assets/figures/3_main_cpp_UseCamera.png" alt="3_main_cpp_UseCamera" width="80%"/>

  - 如果不使用 Arducam，请将 `UseCamera` 宏设置为 0（第 32 行），并将 `NoCamera_Person` 宏设置为 0 或 1（第 33 行），如下图所示：
  
  <img src="../../assets/figures/3_main_cpp_NoUseCamera.png" alt="3_main_cpp_NoUseCamera" width="80%"/>

- 检查编译设置是否正确。（默认设置应该是正确的，但请按以下步骤确认）：

  - 通过 \[Project\] -> \[Properties\] -> \[C/C++ Build\] -> \[Settings\] -> \[Tool Settings\] -> \[MCU GCC Compiler\] -> \[Include paths\] 设置 GCC 编译器的头文件路径，如下图所示：

  <img src="../../assets/figures/4_gcc_include_paths.png" alt="4_gcc_include_paths" width="65%"/>

  - 通过 \[Project\] -> \[Properties\] -> \[C/C++ Build\] -> \[Settings\] -> \[Tool Settings\] -> \[MCU GCC Compiler\] -> \[Optimization\] 将 GCC 编译器的优化级别设为 `-Ofast`，如下图所示：

  <img src="../../assets/figures/5_gcc_optimization.png" alt="5_gcc_optimization" width="65%"/>

  - 通过 \[Project\] -> \[Properties\] -> \[C/C++ Build\] -> \[Settings\] -> \[Tool Settings\] -> \[MCU G++ Compiler\] -> \[Include paths\] 设置 G++ 编译器的头文件路径，如下图所示：

  <img src="../../assets/figures/6_gplusplus_include_paths.png" alt="6_gplusplus_include_paths" width="65%"/>

  - 通过 \[Project\] -> \[Properties\] -> \[C/C++ Build\] -> \[Settings\] -> \[Tool Settings\] -> \[MCU G++ Compiler\] -> \[Optimization\] 将 G++ 编译器的优化级别设为 `-Ofast`，如下图所示：

  <img src="../../assets/figures/7_gplusplus_optimization.png" alt="7_gplusplus_optimization" width="65%"/>

- 点击 \[Project\] -> \[Build Project\] 编译程序并生成二进制可执行文件。

- 通过 \[Run\] -> \[Run Configurations...\] -> \[STM32 Cortex-M C/C++ Application\] -> \[TinyEngine_vww_tutorial Debug\] -> \[C/C++ Application\] -> \[Browse...\] 设置运行/调试配置：

  - 指定正确的 elf 文件（文件路径：`Debug/TinyEngine_vww_tutorial.elf`）以确保程序能正确运行，如下图所示：

<img src="../../assets/figures/8_run_configurations_0.png" alt="8_run_configurations_0" width="49%"/>  <img src="../../assets/figures/9_run_configurations_1.png" alt="9_run_configurations_1" width="49%"/>

4. 设置你的 STM32F746G-DISCO 探索板，将 Arducam 连接到开发板，同时建立与开发板的 USB 连接。

- **(可选)** 根据以下引脚定义用杜邦线将 Arducam 连接到开发板：

  - SPI: MOSI->PB15(D11), MISO->PB14(D12), SCK->PI_1(D13), CS(NSS)->PI_0(D5), VCC-> 3.3V, GND->GND
  - I2C: SCL->PB8(D15). SDA->PB9(D14)

  <img src="../../assets/figures/10_mcu_top_view.png" alt="10_mcu_top_view" width="40%"/>
  <img src="../../assets/figures/11_mcu_side_view.png" alt="11_mcu_side_view" width="40%"/>

  ```
                  (俯视图)                                        (侧视图)
  ```

- 建立与 STM32F746G-DISCO 探索板的 USB 连接。

5. 现在，让我们运行演示。

- 点击 \[Run\] -> \[Run\] 在开发板上执行二进制可执行文件。
- 如果系统要求更新 ST-LINK 固件，请先点击 "OK"：

<img src="../../assets/figures/12_stlink_0.png" alt="12_stlink_0" width="40%"/>

- 点击 "Open in update mode"：

<img src="../../assets/figures/13_stlink_1.png" alt="13_stlink_1" width="40%"/>

- 点击 "Upgrade"：

<img src="../../assets/figures/14_stlink_2.png" alt="14_stlink_2" width="40%"/>

- 在 STM32CubeIDE 中再次点击 \[Run\] -> \[Run\]。

6. 如果你成功运行了演示，STM32F746G-DISCO 探索板的 LCD 屏幕将显示人物检测结果（有人/无人）和每秒帧数（FPS），如下图所示：

  - **使用 Arducam 时：**

   <img src="../../assets/figures/15_demo_person.png" alt="15_demo_person" width="40%"/> <img src="../../assets/figures/16_demo_no_person.png" alt="16_demo_no_person" width="40%"/>

   ```
                    (有人)                                         (无人)
   ```

  - **不使用 Arducam 时：**

   <img src="../../assets/figures/17_demo_person_noArducam.png" alt="17_demo_person_noArducam" width="40%"/> <img src="../../assets/figures/18_demo_no_person_noArducam.png" alt="18_demo_no_person_noArducam" width="40%"/>

   ```
                    (有人)                                         (无人)
   ```

## 已知限制

- 本演示仅在 STM32CubeIDE 1.5.0 版本上测试过。
