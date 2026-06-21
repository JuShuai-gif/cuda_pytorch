# Memory Report

```
# Edge Memory Planning Report

- Tensors: 7
- Naive peak (no reuse): 6144 bytes
- Planned peak (reuse) : 4608 bytes
- Saving: 25%

| id | shape | bytes | live[birth,death] | offset |
|----|-------|-------|-------------------|--------|
| 0 | tensor<8x16xf32> | 512 | [0,0] | 4096 |
| 1 | tensor<16x32xf32> | 2048 | [0,0] | 0 |
| 2 | tensor<32x8xf32> | 1024 | [0,2] | 2048 |
| 3 | tensor<8x32xf32> | 1024 | [0,1] | 3072 |
| 4 | tensor<8x32xf32> | 1024 | [1,2] | 0 |
| 5 | tensor<8x8xf32> | 256 | [2,3] | 1024 |
| 6 | tensor<8x8xf32> | 256 | [3,5] | 0 |
```

