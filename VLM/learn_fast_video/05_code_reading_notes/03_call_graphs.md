# 调用图集

> 汇总各主要流程的调用图，方便速查。详细版见 `03_core_flows/`。

## 1. 推理主链路

```mermaid
graph TD
    U["用户"] --> FP["from_pretrained"]
    FP --> FC["from_config → FastVideoArgs"]
    FC --> EX["Executor(spawn Worker)"]
    EX --> W["Worker.init_device → build_pipeline"]
    U --> GV["generate_video"]
    GV --> GSV["_generate_single_video"]
    GSV --> EF["executor.execute_forward"]
    EF --> RPC["collective_rpc(Pipe)"]
    RPC --> WEF["Worker.execute_forward"]
    WEF --> PF["pipeline.forward"]
    PF --> STAGES["for stage in stages"]
    STAGES --> SAVE["rearrange+imageio → mp4"]
```

## 2. Pipeline stages

```mermaid
graph LR
    IV["InputValidation"] --> TE["TextEncoding"]
    TE --> CD["Conditioning"]
    CD --> TP["TimestepPrep"]
    TP --> LP["LatentPrep"]
    LP --> DN["Denoising(50步)"]
    DN --> DC["Decoding"]
```

## 3. 去噪循环

```mermaid
graph TD
    T["timestep t"] --> SI["scale_model_input"]
    SI --> DiT["DiT.forward → v_pred"]
    DiT --> CFG["CFG: uncond+scale(cond-uncond)"]
    CFG --> STEP["scheduler.step → 新latents"]
    STEP -->|循环| T
```

## 4. DiT forward（Wan）

```mermaid
graph TD
    H["hidden_states [B,16,T,H,W]"] --> PE["patch_embedding → [B,L,dim]"]
    PE --> SP["SP shard"]
    SP --> CE["condition_embedder(t,text)"]
    CE --> BLK["40× (Self-Attn→Cross-Attn→FFN)"]
    BLK --> OUT["norm_out+proj_out+unpatchify → [B,16,T,H,W]"]
```

## 5. Attention（SP + 后端）

```mermaid
graph TD
    QKV["QKV"] --> A2A1["all_to_all_4D 前"]
    A2A1 --> PRE["preprocess_qkv"]
    PRE --> IMPL["attn_impl.forward(flash/sage/vsa/sdpa)"]
    IMPL --> POST["postprocess_output"]
    POST --> A2A2["all_to_all_4D 后"]
```

## 6. 模型加载

```mermaid
graph TD
    LM["load_modules"] --> CL["ComponentLoader.for_module_type"]
    CL --> TL["TransformerLoader.load"]
    TL --> RC["resolve_model_cls"]
    RC --> FSDP["maybe_load_fsdp_model"]
    FSDP --> SM["shard_model(FSDP2)"]
    FSDP --> LW["load_from_state_dict(distribute_tensor)"]
```

## 7. 训练主循环（新栈）

```mermaid
graph TD
    E["run_training_from_config"] --> BC["build_from_config"]
    BC --> TR["Trainer.run"]
    TR --> STS["method.single_train_step → loss"]
    STS --> BW["method.backward"]
    BW --> OPT["optimizers_schedulers_step"]
    OPT --> CKPT["checkpoint_manager.maybe_save"]
    CKPT --> VAL["validation"]
    VAL -->|循环| STS
```

## 8. DMD2 蒸馏

```mermaid
graph TD
    STS["single_train_step"] --> SR["student_rollout → gen_x0"]
    SR --> DMD["_dmd_loss(teacher+critic) → generator_loss"]
    STS --> CFL["_critic_flow_matching_loss → fake_score_loss"]
    DMD --> BW["backward"]
    CFL --> BW
    BW --> STEP["critic.step + student.step(按interval)"]
```

## 9. Python → CUDA kernel（VSA）

```mermaid
graph TD
    VSA["video_sparse_attn(ops.py)"] --> BSI["block_sparse_attn_from_indices"]
    BSI --> CO["block_sparse_attn_sm90 custom_op"]
    CO --> PB["fastvideo_kernel_ops.block_sparse_fwd(pybind)"]
    PB --> CPP["block_sparse_attention_forward(.cu host)"]
    CPP --> KER["fwd_attend_ker<<<grid>>>(CUDA)"]
```

## 10. 分布式初始化

```mermaid
graph TD
    M["maybe_init_dist_env_and_model_parallel"] --> IDE["init_distributed_environment(_WORLD,_NODE)"]
    IDE --> IMP["initialize_model_parallel(_TP,_SP,_DP)"]
    IMP --> DEV["set torch device"]
```

## 相关
- 各流程详解见 `03_core_flows/`。
