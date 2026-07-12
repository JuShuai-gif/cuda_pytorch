"""Parallelism strategies demo: DP, DDP, Tensor Parallel, Pipeline Parallel, ZeRO.

Companion script for distributed_techniques/parallelism_strategies/README.md.
All experiments run on a single process, illustrating the concepts.
For actual multi-GPU runs, see collective_operations/ and torch_dist/.

Run:
    python test1.py              # full demo
    python test1.py dp           # DP vs DDP concept
    python test1.py tensor       # tensor parallelism
    python test1.py pipeline     # pipeline parallelism
    python test1.py zero         # ZeRO memory analysis
    python test1.py mfu          # MFU calculation
"""

import sys
import time

import torch
import torch.nn as nn


# ============ 1. DP vs DDP concept ============
def exp_dp_ddp():
    print("=" * 60)
    print("1. DataParallel (DP) vs DistributedDataParallel (DDP)")
    print("=" * 60)

    # Simulate 4 GPUs
    num_gpus = 4
    model_size = 1_000_000  # params
    batch_per_gpu = 32
    grad_size = model_size  # bytes of gradient
    bw_interconnect = 12e9  # 12 GB/s (NVLink-like)

    # --- DP: master GPU bottleneck ---
    # Step 1: scatter inputs (small, negligible)
    # Step 2: each GPU does forward+backward in parallel
    compute_time_dp = 0.01  # same for both

    # Step 3: all gradients sent to master -> master receives N-1 copies
    grad_data = grad_size * 4 * (num_gpus - 1)  # bytes (fp32)
    gather_time = grad_data / bw_interconnect

    # Step 4: master updates weights (serial)
    update_time = 0.005

    # Step 5: broadcast new weights
    scatter_time = model_size * 4 / bw_interconnect

    total_dp = compute_time_dp + gather_time + update_time + scatter_time

    # --- DDP: all-reduce ---
    # Each GPU does ring all-reduce: 2*(N-1)/N * data per GPU
    reduce_data_per_gpu = grad_size * 4 * 2 * (num_gpus - 1) / num_gpus
    allreduce_time = reduce_data_per_gpu / bw_interconnect

    total_ddp = compute_time_dp + allreduce_time

    print(f"  Simulated step time (4 GPUs, {model_size / 1e6:.0f}M params):")
    print(
        f"    DP:  compute={compute_time_dp * 1e3:.1f}ms  gather={gather_time * 1e3:.1f}ms  "
        f"update={update_time * 1e3:.1f}ms  scatter={scatter_time * 1e3:.1f}ms"
    )
    print(f"         total = {total_dp * 1e3:.1f} ms")
    print(
        f"    DDP: compute={compute_time_dp * 1e3:.1f}ms  allreduce={allreduce_time * 1e3:.1f}ms"
    )
    print(f"         total = {total_ddp * 1e3:.1f} ms")
    print(f"\n  DDP speedup over DP for grad sync: {gather_time / allreduce_time:.1f}x")
    print("  -> DP: master serializes gradient aggregation (sequential bottleneck)")
    print("     DDP: all-reduce parallelizes communication (ring algorithm)")
    print()


# ============ 2. Tensor parallelism concept ============
def exp_tensor_parallel():
    print("=" * 60)
    print("2. Tensor parallelism: splitting matrix multiplication")
    print("=" * 60)

    torch.manual_seed(42)
    D, F = 256, 512
    B = 128

    X = torch.randn(B, D)  # input  [B, D]
    A = torch.randn(D, F)  # weight [D, F]

    # Full matmul (reference)
    Y_full = X @ A  # [B, F]

    # --- Column-parallel: split A by columns, each GPU computes part of output ---
    num_devices = 4
    F_per_device = F // num_devices  # 128 each

    # GPU i holds A[:, i*Fper:(i+1)*Fper]
    Y_parts = []
    for i in range(num_devices):
        A_part = A[:, i * F_per_device : (i + 1) * F_per_device]  # [D, Fper]
        Y_parts.append(X @ A_part)  # [B, Fper]

    # All-gather to reconstruct full output
    Y_col = torch.cat(Y_parts, dim=-1)  # [B, F]
    col_error = (Y_full - Y_col).abs().max().item()
    print(f"  Column-parallel (split A by columns):")
    print(f"    each GPU: [B,{D}] @ [{D},{F_per_device}] -> [{B},{F_per_device}]")
    print(f"    max error: {col_error:.6f}")
    print(f"    after all-gather: [{B}, {F}]")

    # --- Row-parallel: split X by rows AND A by rows ---
    B_per_device = B // num_devices  # 32 each

    Y_parts2 = []
    for i in range(num_devices):
        X_part = X[i * B_per_device : (i + 1) * B_per_device, :]  # [Bper, D]
        A_part = A.clone()  # each GPU has full A, computes partial B
        # Actually row-parallel for linear: split B dim of output
        # Y = XA: split X by rows means each GPU computes part of output rows
        Y_parts2.append(X_part @ A_part)  # [Bper, F]

    Y_row = torch.cat(Y_parts2, dim=0)  # [B, F]
    row_error = (Y_full - Y_row).abs().max().item()
    print(f"\n  Row-parallel (split X by rows):")
    print(f"    each GPU: [{B_per_device},{D}] @ [{D},{F}] -> [{B_per_device},{F}]")
    print(f"    max error: {row_error:.6f}")
    print(f"    after concatenation: [{B}, {F}]")

    # --- True row-parallel (split A by rows, all-reduce partial sums) ---
    D_per_device = D // num_devices
    Y_parts3 = []
    for i in range(num_devices):
        X_part = X[:, i * D_per_device : (i + 1) * D_per_device]  # [B, Dper]
        A_part = A[i * D_per_device : (i + 1) * D_per_device, :]  # [Dper, F]
        Y_parts3.append(X_part @ A_part)  # partial [B, F]

    Y_partial = sum(Y_parts3)  # all-reduce sum -> [B, F]
    partial_error = (Y_full - Y_partial).abs().max().item()
    print(f"\n  Row-parallel (split A by rows, all-reduce):")
    print(
        f"    each GPU: [{B},{D_per_device}] @ [{D_per_device},{F}] -> partial [{B},{F}]"
    )
    print(f"    after all-reduce sum: [{B}, {F}]")
    print(f"    max error: {partial_error:.6f}")

    print("\n  -> Column-parallel: parallelizes output dimension (no reduction needed)")
    print("     Row-parallel:    parallelizes input dimension (all-reduce needed)")
    print("     Megatron-LM alternates column/row to minimize communication")
    print()


# ============ 3. Pipeline parallelism concept ============
def exp_pipeline():
    print("=" * 60)
    print("3. Pipeline parallelism: layer splitting across GPUs")
    print("=" * 60)

    num_stages = 4
    num_microbatches = 8

    # Simulate: each stage takes 1 unit of time for forward, 1 for backward
    fwd_per_stage = 1.0
    bwd_per_stage = 1.0
    comm_per_boundary = 0.1

    # --- Naive: all forward, all backward ---
    # All microbatches flow through pipeline forward, then backward
    # Pipeline bubble: stages wait while first microbatch reaches end
    bubble_fwd = (num_stages - 1) * (fwd_per_stage + comm_per_boundary)
    bubble_bwd = (num_stages - 1) * (bwd_per_stage + comm_per_boundary)
    total_naive = (
        num_microbatches
        * num_stages
        * (fwd_per_stage + bwd_per_stage + 2 * comm_per_boundary)
        + bubble_fwd
        + bubble_bwd
    )
    # This overcounts a bit but illustrates the point

    # Better model:
    # Forward: stage 1 starts at t=0, stage s starts at t=(s-1)*(fwd+comm)
    # Last microbatch finishes forward: (N-1)*fwd + (stages-1)*(fwd+comm) + stages*fwd
    # Then backward flows back similarly
    t_fwd_last = (num_microbatches - 1) * fwd_per_stage + num_stages * (
        fwd_per_stage + comm_per_boundary
    )
    t_bwd_end = (
        t_fwd_last
        + num_stages * (bwd_per_stage + comm_per_boundary)
        + (num_microbatches - 1) * bwd_per_stage
    )

    print(f"  Pipeline: {num_stages} stages, {num_microbatches} microbatches")
    print(
        f"  Forward:  {fwd_per_stage}s/stage, Backward: {bwd_per_stage}s/stage, Comm: {comm_per_boundary}s/boundary"
    )
    print()

    # Time per strategy
    # 1. Naive (AFAB): stages idle during forward fill + backward drain
    naive_time = (
        num_stages * (fwd_per_stage + comm_per_boundary) * num_microbatches
        + num_stages * (bwd_per_stage + comm_per_boundary) * num_microbatches
    )
    print(f"  1. All-Forward-All-Backward: {naive_time:.1f}s  (bubble: large)")

    # 2. 1F1B: interleave forward and backward
    # After first microbatch finishes forward at stage S, start its backward
    # Pipeline fill: (stages-1) idle before stage S gets work
    # Steady state: 1 forward + 1 backward per microbatch
    # Pipeline drain: last (stages-1) backward steps
    ofob_time = (
        (num_stages - 1) * (fwd_per_stage + comm_per_boundary)  # warmup
        + num_microbatches
        * (fwd_per_stage + bwd_per_stage + 2 * comm_per_boundary)  # steady
        + (num_stages - 1) * (bwd_per_stage + comm_per_boundary)  # drain
    )
    mem_ofob = num_stages  # only 'stages' activation sets live at once

    print(f"  2. 1F1B (One-Forward-One-Backward): {ofob_time:.1f}s")
    print(
        f"     activation memory: {mem_ofob} microbatch-activation-sets (vs {num_microbatches} for AFAB)"
    )

    # 3. Interleaved: model chunks per GPU
    chunks_per_gpu = 2
    n_rounds = num_stages * chunks_per_gpu  # total pipeline stages
    interleaved_time = (
        (n_rounds - 1) * (fwd_per_stage + comm_per_boundary)
        + num_microbatches * (fwd_per_stage + bwd_per_stage + 2 * comm_per_boundary)
        + (n_rounds - 1) * (bwd_per_stage + comm_per_boundary)
    )
    print(f"  3. Interleaved (2 chunks/GPU): {interleaved_time:.1f}s")
    print(
        f"     less bubble but more communication ({n_rounds - num_stages} extra boundaries)"
    )

    print("\n  -> More microbatches = better GPU utilization (less bubble)")
    print("     But activation memory grows with number of in-flight microbatches")
    print(
        "     1F1B keeps activation memory proportional to pipeline depth, not batch count"
    )
    print()


# ============ 4. ZeRO memory analysis ============
def exp_zero():
    print("=" * 60)
    print("4. ZeRO: memory savings across stages")
    print("=" * 60)

    params = 70e9  # 70B model
    bytes_per_param = 4  # FP32
    Nd = 8  # data-parallel degree

    param_mem = params * bytes_per_param  # 280 GB
    grad_mem = params * bytes_per_param  # 280 GB
    # Optimizer states (Adam): fp32 param copy + fp32 momentum + fp32 variance
    opt_mem = (
        params * bytes_per_param * 2
    )  # 560 GB (just momentum+variance, param copy is extra)
    total_base = param_mem + grad_mem + opt_mem

    def gb(x):
        return x / 1e9

    print(f"  Model: {params / 1e9:.0f}B params, FP32, DP={Nd}")
    print(f"    Base (no sharding):")
    print(f"      params:         {gb(param_mem):.0f} GB")
    print(f"      gradients:      {gb(grad_mem):.0f} GB")
    print(f"      opt states:     {gb(opt_mem):.0f} GB")
    print(f"      TOTAL:          {gb(total_base):.0f} GB")

    # ZeRO-1: shard optimizer states only
    opt_per_gpu = opt_mem / Nd
    total_z1 = param_mem + grad_mem + opt_per_gpu
    print(f"\n    ZeRO-1 (shard optimizer):")
    print(f"      opt states/gpu: {gb(opt_per_gpu):.0f} GB")
    print(
        f"      total/gpu:      {gb(total_z1):.0f} GB (saved {gb(opt_mem - opt_per_gpu):.0f} GB)"
    )

    # ZeRO-2: shard optimizer + gradients
    grad_per_gpu = grad_mem / Nd
    total_z2 = param_mem + grad_per_gpu + opt_per_gpu
    print(f"\n    ZeRO-2 (shard opt+grad):")
    print(
        f"      total/gpu:      {gb(total_z2):.0f} GB (saved {gb(grad_mem - grad_per_gpu):.0f} GB)"
    )

    # ZeRO-3: shard everything
    param_per_gpu = param_mem / Nd
    total_z3 = param_per_gpu + grad_per_gpu + opt_per_gpu
    print(f"\n    ZeRO-3 (shard all):")
    print(
        f"      total/gpu:      {gb(total_z3):.0f} GB (saved {gb(param_mem - param_per_gpu):.0f} GB)"
    )

    print(f"\n  Memory reduction from base to ZeRO-3: {total_base / total_z3:.1f}x")
    print(f"  Attention: activations are NOT sharded by ZeRO (need TP/SP for that)")
    print()


# ============ 5. MFU calculation ============
def exp_mfu():
    print("=" * 60)
    print("5. MFU (Model FLOPs Utilization)")
    print("=" * 60)

    # Estimate MFU for a simple transformer layer
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA device available")
        return

    device = "cuda"
    B, S, D, FF = 8, 1024, 4096, 16384  # batch, seq, model_dim, ff_dim

    # A simple transformer block: attention + FFN
    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_qkv = nn.Linear(D, 3 * D, bias=False)
            self.attn_out = nn.Linear(D, D, bias=False)
            self.ffn1 = nn.Linear(D, FF, bias=False)
            self.ffn2 = nn.Linear(FF, D, bias=False)
            self.ln1 = nn.LayerNorm(D)
            self.ln2 = nn.LayerNorm(D)

        def forward(self, x):
            # Attention
            qkv = self.attn_qkv(x)  # [B,S,3D] -> split into Q,K,V
            q, k, v = qkv.chunk(3, dim=-1)
            # Simplified: no actual attention, just matmul pattern
            attn = q @ k.transpose(-1, -2) / (D**0.5)  # [B,S,S]
            attn_out = attn @ v  # [B,S,D]
            out_attn = self.attn_out(attn_out)  # [B,S,D]
            x = self.ln1(x + out_attn)
            # FFN
            ffn = self.ffn2(torch.relu(self.ffn1(x)))  # [B,S,D]
            x = self.ln2(x + ffn)
            return x

    block = TransformerBlock().to(device).train()

    # Count FLOPs
    # QKV: 2*B*S*D*3D
    # Q@K^T: 2*B*S*S*D
    # Attn@V: 2*B*S*S*D
    # Out: 2*B*S*D*D
    # FFN1: 2*B*S*D*FF
    # FFN2: 2*B*S*FF*D
    flops_per_block = (
        2 * B * S * D * 3 * D  # QKV projection
        + 2 * B * S * S * D  # Q @ K^T
        + 2 * B * S * S * D  # attn @ V
        + 2 * B * S * D * D  # out projection
        + 2 * B * S * D * FF  # FFN gate
        + 2 * B * S * FF * D  # FFN down
    )

    x = torch.randn(B, S, D, device=device)

    # Warmup
    for _ in range(5):
        block(x)
    torch.cuda.synchronize()

    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        block(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / n_iter
    achieved_tflops = flops_per_block / elapsed / 1e12

    # A100 peak (approximate, fp16 tensor-core)
    peak_tflops = 312  # A100 fp16 TFLOPS
    mfu = achieved_tflops / peak_tflops * 100

    print(f"  Block: B={B} S={S} D={D} FF={FF}")
    print(f"  FLOPs/block: {flops_per_block / 1e12:.3f} TFLOPs")
    print(f"  Time/block:  {elapsed * 1e3:.3f} ms")
    print(f"  Achieved:    {achieved_tflops:.1f} TFLOPS")
    print(f"  Peak (A100): {peak_tflops} TFLOPS")
    print(f"  MFU:         {mfu:.1f}%")

    print(f"\n  MFU = (FLOPs / time) / peak_FLOPS")
    print(f"  Low MFU (<50%) usually means memory-bound or communication overhead")
    print(f"  High MFU (>70%) means good compute utilization")
    print()


EXPERIMENTS = {
    "dp": exp_dp_ddp,
    "tensor": exp_tensor_parallel,
    "pipeline": exp_pipeline,
    "zero": exp_zero,
    "mfu": exp_mfu,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[exp]()

    print("[parallelism demo] DONE")


if __name__ == "__main__":
    main()
