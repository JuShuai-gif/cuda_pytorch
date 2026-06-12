# Chapter 15: xFormers Source Analysis

## Scope

This chapter is source-reading oriented. It has no local C++/CUDA binary target. The goal is to understand production attention system design and connect the upstream implementation choices back to the kernels and memory models from earlier chapters.

## Read

```bash
sed -n '1,240p' ../../notes/chapter_15.md
```

## What To Extract

| Item | Why it matters |
|---|---|
| Dispatch path | Explains which backend runs for each shape/dtype/mask. |
| Data layout | Determines kernel compatibility and memory coalescing. |
| Metadata | Sequence lengths, block tables, masks, and cache state control runtime behavior. |
| Failure/fallback path | Production systems need robust behavior when the fastest kernel cannot run. |

## Engineering Notes

Do not read these systems as isolated kernels. Trace the API entrypoint, validation, dispatch, metadata construction, kernel call, and output handling as one end-to-end path.
