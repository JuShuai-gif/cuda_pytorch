"""Checkpoint Format case study 5: Distributed checkpoint load/save patterns.

Companion script for checkpoint_format/checkpoint_format.md. Covers:
  1. DCP (Distributed Checkpoint) API usage
  2. Resharding with world_size change
  3. Checkpoint consolidation

Run:
    python 05_distributed_checkpoint.py
"""

import sys

import torch


def exp_dcp_basics():
    print("=" * 60)
    print("1. Distributed Checkpoint (DCP) overview")
    print("=" * 60)

    model = torch.nn.Linear(16, 8)
    state_dict = model.state_dict()

    print(f"  DCP API (torch.distributed.checkpoint):")
    print(f"    from torch.distributed.checkpoint import save, load")
    print(f"    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter")
    print(f"")
    print(f"  Save pattern:")
    print(f"    writer = FileSystemWriter('/path/to/ckpt')")
    print(f"    save(state_dict, checkpoint_id='step_1000', storage_writer=writer)")
    print(f"")
    print(f"  Output structure:")
    print(f"    /path/to/ckpt/step_1000/")
    print(f"      .metadata           # global metadata")
    print(f"      __0_0.distcp        # rank 0 shard 0")
    print(f"      __1_0.distcp        # rank 1 shard 0")
    print(f"      ...")

    # Show state_dict key structure
    print(f"\n  State dict keys for Linear(16,8):")
    for k in state_dict:
        print(f"    {k}: shape={list(state_dict[k].shape)}")
    print()


def exp_reshard():
    print("=" * 60)
    print("2. Resharding: world_size changed")
    print("=" * 60)

    print(f"  Saved with world_size=4 (4 ranks, each saves 1/4):")
    print(f"    rank_0: weight[0:4, :]   bias[0:2]")
    print(f"    rank_1: weight[4:8, :]   bias[2:4]")
    print(f"    rank_2: weight[8:12, :]  bias[4:6]")
    print(f"    rank_3: weight[12:16, :] bias[6:8]")
    print(f"")

    print(f"  Load with world_size=2 (2 ranks):")
    print(f"    New rank_0 reads: rank_0 + rank_1 shards -> weight[0:8,:]  bias[0:4]")
    print(f"    New rank_1 reads: rank_2 + rank_3 shards -> weight[8:16,:] bias[4:8]")
    print(f"")

    print(f"  Load with world_size=1 (consolidation):")
    print(f"    Single rank reads ALL shards -> full weight, full bias")
    print(f"")

    print(f"  DCP handles this via planner (DefaultSavePlanner):")
    print(f"    - Writes shard metadata per-rank")
    print(f"    - Reads use plan to merge/redistribute")
    print()


def exp_load_resume():
    print("=" * 60)
    print("3. Training resume with DCP")
    print("=" * 60)

    print(f"  Resuming training:")
    print(f"")

    print(f"  # Save training state")
    print(f"  training_state = {")
    print(f"      'model': model.state_dict(),")
    print(f"      'optimizer': optimizer.state_dict(),")
    print(f"      'lr_scheduler': scheduler.state_dict(),")
    print(f"      'epoch': epoch,")
    print(f"      'global_step': global_step,")
    print(f"  }")
    print(f"  save(training_state, checkpoint_id=f'epoch_{epoch}')")
    print(f"")

    print(f"  # Load and resume")
    print(f"  training_state = load(training_state, checkpoint_id='epoch_10')")
    print(f"  model.load_state_dict(training_state['model'])")
    print(f"  optimizer.load_state_dict(training_state['optimizer'])")
    print(f"")

    print(f"  Note: DCP saves tensors via StorageWriter")
    print(f"  Non-tensor data (epoch, step) goes in metadata")
    print()


EXPERIMENTS = {
    "dcp": exp_dcp_basics,
    "reshard": exp_reshard,
    "resume": exp_load_resume,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint_format case 5] DONE")


if __name__ == "__main__":
    main()
