# 如何添加 Attention 后端

> 实现一个新的 attention 后端并接入 selector。

## 1. 后端接口（backends/abstract.py）

一个后端需实现 3 个类：
```python
class MyAttentionBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls(): return MyAttentionImpl
    @staticmethod
    def get_metadata_cls(): return MyAttentionMetadata
    @staticmethod
    def get_builder_cls(): return MyAttentionMetadataBuilder

class MyAttentionImpl(AttentionImpl):
    def preprocess_qkv(self, qkv, metadata): return qkv
    def forward(self, q, k, v, attn_metadata): ...   # 核心
    def postprocess_output(self, output, metadata): return output
```

## 2. 最简后端（参考 SDPA）

```python
# backends/my_attn.py
class MyAttentionImpl(AttentionImpl):
    def forward(self, q, k, v, attn_metadata):
        # q,k,v: [B, L, H, D]
        query = q.transpose(1, 2)   # → [B, H, L, D]
        output = my_attention_kernel(query, key, value)
        return output.transpose(1, 2)   # → [B, L, H, D]
```

## 3. 稀疏后端（需 tile）

若是稀疏后端，用 `preprocess_qkv`/`postprocess_output` 做 tile 重排（参考 `video_sparse_attn.py`）：
```python
def preprocess_qkv(self, qkv, metadata):
    return self.tile(qkv, metadata)     # token → block
def forward(self, q, k, v, metadata):
    return my_sparse_kernel(q, k, v, metadata.topk)
def postprocess_output(self, output, metadata):
    return self.untile(output, metadata)
```
并在 `AttentionMetadata` 携带稀疏所需信息（tile 划分、block 大小）。

## 4. 注册到 selector

后端选择在 `attention/selector.py` + 平台 `get_attn_backend_cls`。需要：
1. 在 `AttentionBackendEnum` 加枚举值。
2. 在平台 `get_attn_backend_cls`（`platforms/`）映射枚举 → 后端类 qualname。
3. `backend_name_to_enum`（selector.py）支持名字。

（**具体注册位置待确认**：查看 `platforms/cuda.py` 的 `get_attn_backend_cls` 实现。）

## 5. 声明模型支持

在 DiT 的 `_supported_attention_backends` 加入你的后端枚举，否则不会被选中。

## 6. torch.compile 兼容

若 kernel 不可追踪，用 `torch.library.custom_op` 包装（参考 `flash_attn.py:65`）：
```python
@torch.library.custom_op("fastvideo::my_attn_forward", mutates_args=(), device_types="cuda")
def my_attn_forward(q, k, v): ...
```

## 7. 测试

```bash
FASTVIDEO_ATTENTION_BACKEND=MY_ATTN python examples/inference/basic/basic.py
```
对比 SDPA 输出验证正确性（数值接近）。

## 8. 检查清单

- [ ] 实现 Backend/Impl/Metadata 三类。
- [ ] `forward` 输入输出 `[B,L,H,D]`（或正确转置）。
- [ ] 加 `AttentionBackendEnum` 枚举。
- [ ] 平台 `get_attn_backend_cls` 映射。
- [ ] DiT `_supported_attention_backends` 声明。
- [ ] 稀疏后端处理 metadata + tile。
- [ ] torch.compile 兼容（如需）。

## 9. 参考
- `attention/backends/sdpa.py`（最简）。
- `attention/backends/flash_attn.py`（含 compile 包装）。
- `attention/backends/video_sparse_attn.py`（稀疏 tile）。
- `attention/selector.py`（选择逻辑）。
