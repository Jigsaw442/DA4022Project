### Clean decode latency (1x H100 NVL, 1 job alone, 30 captions, max 128 new tokens, greedy, eager attention)

| Config | Visual tokens | s/caption |
|---|---|---|
| none576 | 576 | 1.33 |
| hybrid144 | 144 | 1.23 |
| hybrid64 | 64 | 1.19 |
| hybrid16 | 16 | 1.17 |

Note: the s/caption column in main_results.md is not comparable across rows (none576/hybrid144/hybrid64 greedy ran during a slow, shared-GPU period). Use this table for speed.
