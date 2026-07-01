"""Intrusive Ptr case study 1: reference counting behavior.

Run: python 01_refcount.py
"""

import sys, torch

def exp_storage_refcount():
    print("=" * 60)
    print("1. Storage intrusive_ptr refcount lifecycle")
    print("=" * 60)
    x = torch.randn(1024)
    s = x.storage()
    print(f"  After x created: use_count={s._use_count()}")
    y = x.view(2, 512)
    print(f"  After view:      use_count={s._use_count()} (shared, no copy)")
    z = x[::2]
    print(f"  After slice:     use_count={s._use_count()} (shared, no copy)")
    del y
    print(f"  After del y:     use_count={s._use_count()}")
    del z
    print(f"  After del z:     use_count={s._use_count()}")
    del x
    print(f"  After del x:     storage shared with s -> still alive")
    print(f"  s still holds ref: use_count={s._use_count()}")

def exp_data_ptr_share():
    print("=" * 60)
    print("2. Data pointer sharing = same intrusive_ptr")
    print("=" * 60)
    a = torch.randn(1000)
    b = a.view(10, 100)
    c = a[100:200]
    print(f"  a.data_ptr() == b.data_ptr(): {a.data_ptr() == b.data_ptr()}")
    print(f"  b.storage().data_ptr() == c.storage().data_ptr(): {b.storage().data_ptr() == c.storage().data_ptr()}")

EXPERIMENTS = {"refcount": exp_storage_refcount, "ptr": exp_data_ptr_share}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[intrusive_ptr] DONE")

if __name__ == "__main__": main()
