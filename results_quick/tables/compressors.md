### Compressor comparison at 64 visual tokens (greedy)

| Config | Vis. tok | CHAIR_s ↓ | CHAIR_i ↓ | Recall ↑ | Len | POPE Acc ↑ | POPE F1 ↑ | Yes % | s/caption |
|---|---|---|---|---|---|---|---|---|---|
| none576_greedy | 576 | 53.0 | 27.4 | 78.9 | 82.0 | 84.6 | 83.5 | 43.2 | 20.30 |
| pool64_greedy | 64 | 53.5 | 31.4 | 70.9 | 79.8 | 77.8 | 74.1 | 35.1 | 1.18 |
| topk64_greedy | 64 | 51.0 | 28.7 | 70.0 | 79.9 | 77.6 | 73.5 | 34.3 | 1.19 |
| hybrid64_greedy | 64 | 45.0 | 26.0 | 72.9 | 79.2 | 80.2 | 77.6 | 38.2 | 19.35 |
