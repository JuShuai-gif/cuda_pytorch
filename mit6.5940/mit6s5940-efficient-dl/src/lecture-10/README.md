# Lecture 10 - MCUNet / TinyML Simulation

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: MCUNet — 将深度学习部署到微控制器

## Overview

MCUNet enables running deep neural networks on microcontrollers (MCUs) with severe memory constraints (SRAM ~256KB, Flash ~1MB). This simulation enforces hardware budget checks during model design.

## Key Concepts

- **SRAM Budget**: Limits peak activation memory during inference (~256KB typical for Cortex-M)
- **Flash Budget**: Limits total model size storage (~1MB for on-chip flash)
- **Memory Counting**: Analytical calculation of params, MACs, peak activations, model bytes
- **Budget Enforcement**: Architectures exceeding limits are rejected at design time

## Implementation

| Component      | Description                                          |
| -------------- | ---------------------------------------------------- |
| TinyCNN        | Configurable CNN builder with budget validation      |
| Memory Tracker | Calculates peak activation, param memory analytically |
| Budget Checker | 256KB SRAM / 1MB Flash limits enforced               |
| Report         | Formatted MCU memory budget table                    |

## Usage

```bash
cd src/lecture-10
python main.py
```

## Expected Output

```
============================================================
MCU Memory Budget Report
============================================================
SRAM Budget:  256.00 KB
Flash Budget: 1024.00 KB
------------------------------------------------------------
Architecture   Params    MACs       ActMem    ModelSize   Status
------------------------------------------------------------
TinyNet        48,234    1.2M       184.3 KB  188.4 KB    PASS
WideNet        124,560   4.8M       312.0 KB  486.6 KB    FAIL (SRAM)
FatFC          89,120    2.1M       156.0 KB  1375.0 KB   FAIL (Flash)
============================================================
```

## References

- Lin et al., "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020)
- MIT 6.5940 Lecture 10 Slides
