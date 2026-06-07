# MIT 6.5940 架构图集合

## 课程整体知识架构

```mermaid
graph TB
    subgraph "基础 (Lec 01-02)"
        A1[引言: 为什么需要高效DL]
        A2[基础: 效率指标与神经网络]
    end

    subgraph "高效推理核心技术 (Lec 03-09)"
        B1[剪枝 Pruning]
        B2[量化 Quantization]
        B3[神经架构搜索 NAS]
        B4[知识蒸馏 KD]
    end

    subgraph "TinyML 系统 (Lec 10-11)"
        C1[MCUNet: 模型设计]
        C2[TinyEngine: 推理引擎]
    end

    subgraph "LLM 优化 (Lec 12-15)"
        D1[Transformer/LLM 基础]
        D2[LLM 部署: AWQ/vLLM]
        D3[LLM 后训练: PEFT/LoRA]
        D4[长上下文 LLM]
    end

    subgraph "高级模型优化 (Lec 16-18)"
        E1[Vision Transformer]
        E2[GAN/视频/点云]
        E3[Diffusion Models]
    end

    subgraph "高效训练 (Lec 19-21)"
        F1[分布式训练]
        F2[梯度压缩]
        F3[端侧训练]
    end

    A1 --> A2 --> B1
    A2 --> B2
    A2 --> B3
    B1 --> B4
    B2 --> B4
    
    B1 --> C1
    B2 --> C1
    C1 --> C2
    
    A2 --> D1
    D1 --> D2
    D1 --> D3
    D1 --> D4
    
    D1 --> E1
    E1 --> E2
    E2 --> E3
    
    D1 --> F1
    F1 --> F2
    F2 --> F3

    style B1 fill:#ff6b6b
    style B2 fill:#ffd93d
    style B3 fill:#6bcb77
    style B4 fill:#4d96ff
    style C1 fill:#9b59b6
    style D2 fill:#e74c3c
```

## 模型压缩流水线

```mermaid
graph LR
    subgraph "输入"
        M0[("Baseline<br/>Model<br/>FP32")]
    end

    subgraph "压缩阶段"
        S1["① 剪枝<br/>Pruning<br/>━━━━━━<br/>· 通道剪枝<br/>· 幅度剪枝<br/>· 敏感度分析"]
        S2["② 量化<br/>Quantization<br/>━━━━━━<br/>· PTQ校准<br/>· QAT微调<br/>· INT8/INT4"]
        S3["③ 蒸馏<br/>Distillation<br/>━━━━━━<br/>· 软标签<br/>· 温度T<br/>· CE+KL"]
        S4["④ NAS<br/>Architecture<br/>━━━━━━<br/>· 搜索空间<br/>· 精度预测<br/>· 硬件约束"]
    end

    subgraph "导出与部署"
        E1[ONNX 导出]
        E2[TensorRT 优化]
        E3[ONNX Runtime]
        E4[MCU部署]
    end

    subgraph "评估"
        R1["准确率"]
        R2["延迟 ms"]
        R3["内存 MB"]
        R4["能耗 mJ"]
    end

    M0 --> S1
    S1 --> S2
    S2 --> S3
    S2 --> S4
    
    S1 --> E1
    S2 --> E1
    S3 --> E1
    
    E1 --> E2
    E1 --> E3
    E1 --> E4
    
    E2 --> R1
    E3 --> R1
    E4 --> R1
    E2 --> R2
    E3 --> R3
    E4 --> R4

    style M0 fill:#3498db,color:white
    style S1 fill:#e74c3c,color:white
    style S2 fill:#f39c12,color:white
    style S3 fill:#2ecc71,color:white
    style S4 fill:#9b59b6,color:white
```

## 量化原理: 从浮点到整数

```mermaid
graph TB
    subgraph "FP32 浮点范围"
        F[("r ∈ [r_min, r_max]<br/>4 bytes per value")]
    end
    
    subgraph "量化参数计算"
        S["Scale S = (r_max - r_min) / (q_max - q_min)"]
        Z["Zero Point Z = round(q_min - r_min/S)"]
    end
    
    subgraph "INT8 整数范围"
        Q["q = clamp(round(r/S) + Z, -128, 127)<br/>1 byte per value"]
    end
    
    subgraph "反量化恢复"
        D["r_hat = S × (q - Z)<br/>精度损失 ≈ Δ²/12"]
    end

    F --> S
    F --> Z
    S --> Q
    Z --> Q
    Q --> D

    style F fill:#3498db,color:white
    style Q fill:#e74c3c,color:white
    style D fill:#95a5a6,color:white
```

## 知识蒸馏流程

```mermaid
graph TB
    subgraph "Teacher (大模型)"
        T_IN[输入数据]
        T_NET[Teacher Network<br/>参数量大/已训练好]
        T_LOGITS["Logits z_t"]
        T_SOFT["软标签<br/>p_t = softmax(z_t / T)<br/>包含'暗知识'"]
    end

    subgraph "Student (小模型)"
        S_IN[输入数据]
        S_NET[Student Network<br/>参数量小/待训练]
        S_LOGITS["Logits z_s"]
        S_HARD["硬预测<br/>p_s = softmax(z_s)"]
        S_SOFT["软预测<br/>p_s^T = softmax(z_s / T)"]
    end

    subgraph "损失函数"
        L_CE["交叉熵损失<br/>L_CE = CE(y_true, p_s)"]
        L_KD["蒸馏损失<br/>L_KD = T² × KL(p_t^T || p_s^T)"]
        L_TOTAL["总损失<br/>L = α·L_CE + (1-α)·L_KD"]
    end

    T_IN --> T_NET --> T_LOGITS --> T_SOFT
    S_IN --> S_NET --> S_LOGITS --> S_HARD
    S_LOGITS --> S_SOFT
    
    S_HARD --> L_CE
    T_SOFT --> L_KD
    S_SOFT --> L_KD
    L_CE --> L_TOTAL
    L_KD --> L_TOTAL
    L_TOTAL --> S_NET

    style T_NET fill:#e74c3c,color:white
    style S_NET fill:#2ecc71,color:white
    style L_TOTAL fill:#f39c12,color:white
```

## 端侧AI部署流水线

```mermaid
graph TB
    subgraph "训练阶段 (云端GPU)"
        T1[大规模数据训练]
        T2[Teacher模型训练]
        T3[模型压缩<br/>剪枝+量化+蒸馏]
    end

    subgraph "优化阶段"
        O1[ONNX导出<br/>跨框架互操作]
        O2[TensorRT优化<br/>层融合/精度校准]
        O3[ONNX Runtime<br/>跨平台推理]
    end

    subgraph "部署阶段 (边缘设备)"
        D1[手机/NPU<br/>CoreML/TFLite]
        D2[嵌入式MCU<br/>TinyEngine]
        D3[IoT设备<br/>TFLite Micro]
        D4[浏览器<br/>ONNX Runtime Web]
    end

    subgraph "监控"
        M1[延迟监控]
        M2[准确率漂移]
        M3[模型更新]

    T1 --> T2 --> T3
    T3 --> O1
    O1 --> O2
    O1 --> O3
    O2 --> D1
    O3 --> D1
    O2 --> D2
    O3 --> D3
    O1 --> D4
    
    D1 --> M1
    D1 --> M2
    M2 --> M3
    M3 --> T3

    style T3 fill:#e74c3c,color:white
    style O2 fill:#f39c12,color:white
    style D1 fill:#2ecc71,color:white
    style D2 fill:#9b59b6,color:white
```

## 剪枝粒度层级

```mermaid
graph LR
    subgraph "最细粒度"
        FG["细粒度剪枝<br/>Fine-grained<br/>━━━━━━━<br/>单个权重<br/>非结构化<br/>压缩率最高<br/>需要稀疏硬件"]
    end

    subgraph ""
        VEC["向量级剪枝<br/>Vector-level<br/>━━━━━━━<br/>连续权重块<br/>中等结构化<br/>SIMD友好"]
    end

    subgraph ""
        KERNEL["卷积核级<br/>Kernel-level<br/>━━━━━━━<br/>整个卷积核<br/>部分结构化<br/>直接加速"]
    end

    subgraph "最粗粒度"
        CH["通道剪枝<br/>Channel-level<br/>━━━━━━━<br/>整个通道<br/>完全结构化<br/>任何硬件加速<br/>压缩率最低"]
    end

    FG --> VEC --> KERNEL --> CH

    style FG fill:#e74c3c,color:white
    style CH fill:#2ecc71,color:white
```

## MCU上的TinyML推理

```mermaid
graph TB
    subgraph "MCU硬件约束"
        H1["SRAM < 512KB"]
        H2["Flash < 2MB"]
        H3["算力 ~200 MOPS"]
        H4["无操作系统"]
    end

    subgraph "模型设计 (TinyNAS)"
        M1["搜索空间<br/>在内存预算内"]
        M2["自动搜索<br/>最优架构"]
        M3["MCUNet模型<br/>适配MCU"]
    end

    subgraph "推理引擎 (TinyEngine)"
        E1["内存优化<br/>In-place操作"]
        E2["计算优化<br/>Im2Col/Winograd"]
        E3["SIMD指令<br/>ARM CMSIS-NN"]
        E4["算子融合<br/>Conv+BN+ReLU"]
    end

    H1 --> M1
    H2 --> M1
    M1 --> M2 --> M3
    M3 --> E1
    
    H3 --> E2
    H4 --> E3
    E1 --> E4
    E2 --> E4
    E3 --> E4

    style H1 fill:#e74c3c,color:white
    style M3 fill:#f39c12,color:white
    style E4 fill:#2ecc71,color:white
```
