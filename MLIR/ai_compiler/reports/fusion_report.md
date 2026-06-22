# Fusion Report

## Before optimization

```
# Edge Graph Statistics Report

- Total operations: 6
- Estimated MACs: 6144  (~0 MMACs)

| Operation | Count |
|-----------|-------|
| edge.matmul | 2 |
| edge.relu | 2 |
| func.return | 1 |
| func.func | 1 |
```

## After (shape-inference + conv-bn-relu fusion)

```
# Edge Graph Statistics Report

- Total operations: 6
- Estimated MACs: 6144  (~0 MMACs)

| Operation | Count |
|-----------|-------|
| edge.matmul | 2 |
| edge.relu | 2 |
| func.return | 1 |
| func.func | 1 |
```

