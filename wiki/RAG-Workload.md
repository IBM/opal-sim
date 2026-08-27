# RAG Workload

This page documents the synthetic **RAG workload**, used for KV-cache tiering experiments. For the workload that replays real captured agent traces, see [[OTel Trace Replay]].

Ships with ready-to-run configs in `configs/`, in three KVC-tier variants:

| Config suffix | `kvc_tiers` | Purpose |
|---|---|---|
| `-HBM` | *(none — APC only)* | GPU APC only, no tiered KV-cache storage. Baseline. |
| `-HBM-DRAM` | `["CPUMemory"]` | Two-tier: GPU APC + CPU DRAM. |
| `-HBM-DRAM-DFS` | `["CPUMemory", "DistributedFS"]` | Three-tier: GPU APC + CPU DRAM + DistributedFS. |

Run any of them with:

```bash
python -m opal.main -c configs/RAG_KVCtiers-HBM-DRAM-DFS.json
```

---

**Type string:** `"RAGWorkload"` (class `RAGWorkload`, `opal/workloads/rag_workload.py`)

Simulates a synthetic retrieval-augmented-generation workload designed to exercise prefix-cache reuse across a fixed document corpus.

## How it works

1. At init, a shared **system prompt** (`system_prompt_size` tokens) is hashed once and reused as the prefix of every request.
2. A corpus (pool) of `num_documents` documents (each `document_size` tokens) is generated.
3. Each request selects `docs_per_request` distinct documents at random, sorts them and concatenates them: `system_prompt + doc_1 + doc_2 + ... `.
4. Requests are submitted at `request_rate` (requests/sec), optionally with `jitter` for inter-arrival variance, up to `max_concurrent_requests` in flight at once.

Because documents are reused across requests but combined in different random subsets, the workload produces partial prefix matches (system prompt always hits; documents hit only when re-selected) — useful for testing tiered KV-cache lookup that has to find non-contiguous chunks across tiers.

## Parameters (`workload_params`)

| Parameter | Default | Description |
|---|---|---|
| `num_documents` | `10` | Size of the document corpus to sample from. |
| `document_size` | `16384` | Tokens per document. |
| `system_prompt_size` | `1024` | Tokens in the shared system prompt (prepended to every request). |
| `docs_per_request` | `4` | Number of distinct documents concatenated per request. Must be ≤ `num_documents`. |
| `output_tokens` | `128` | Fixed decode/output length for every request. |
| `request_rate` | `1.0` | Requests per second (mean, before jitter). |
| `jitter` | `0.0` | Inter-arrival variance; `0` = fixed interval, up to `1.0` = high variance. |
| `total_requests` | `-1` | Stop after this many requests (`-1` = unbounded, run until `simulation_time`/`time_duration_sec`). |
| `time_duration_sec` | `-1` | Stage-local timeout in virtual seconds. |
| `max_concurrent_requests` | `16` | Caps in-flight requests; generation blocks until one completes. |

## Example config (from `configs/RAG_KVCtiers-HBM-DRAM-DFS.json`)

```json
"workload": {
  "stages": [
    {
      "type": "RAGWorkload",
      "workload_params": {
        "num_documents": 10,
        "document_size": 16384,
        "system_prompt_size": 1024,
        "docs_per_request": 4,
        "output_tokens": 128,
        "request_rate": 0.25,
        "jitter": 0.0,
        "total_requests": 1000,
        "time_duration_sec": -1,
        "max_concurrent_requests": 1
      }
    }
  ]
}
```

With these settings, total context per request = `1024 + 4*16384 = 66,560` tokens, with the 1024-token system prompt and any previously-selected documents eligible for cache reuse.

## Unique prefix combinations

Since each request sorts its selected documents before concatenating them, document order doesn't matter — the number of distinct document-subset prefixes the workload can produce is the binomial coefficient `C(num_documents, docs_per_request)`. This is the size of the pool of unique non-system-prompt prefixes that `total_requests` draws from (with repetition): the smaller it is relative to `total_requests`, the more often a given combination — and hence its KV-cache entries — gets reused.

| Pool size (`num_documents`) | `docs_per_request=3` | `docs_per_request=4` | `docs_per_request=5` |
|---|---|---|---|
| 10 | 120 | 210 | 252 |
| 20 | 1,140 | 4,845 | 15,504 |
| 30 | 4,060 | 27,405 | 142,506 |
| 40 | 9,880 | 91,390 | 658,008 |
| 50 | 19,600 | 230,300 | 2,118,760 |
| 60 | 34,220 | 487,635 | 5,461,512 |
| 70 | 54,740 | 916,895 | 12,103,014 |
| 80 | 82,160 | 1,581,580 | 24,040,016 |
| 90 | 117,480 | 2,555,190 | 43,949,268 |
| 100 | 161,700 | 3,921,225 | 75,287,520 |

## Reaching steady state

Each request draws one of the `N = C(num_documents, docs_per_request)` unique prefixes from the table above, uniformly at random with replacement. This is the classic **coupon collector's problem**: assuming an infinite KV-cache pool (so nothing is ever evicted), the question is how many requests, on average, it takes to have seen a given fraction `p` of the `N` distinct prefixes at least once.

Going from having seen `j` distinct prefixes to `j+1` takes, in expectation, `N / (N - j)` requests (the probability the next draw is new is `(N-j)/N`). Summing that from `j=0` to `m-1` (where `m = p*N`) gives the expected number of requests to reach fraction `p`:

```text
E[requests to reach p] = N * (H(N) - H(N-m))   where H(n) = n-th harmonic number, m = p*N
                        ≈ N * ln(1 / (1-p))      for large N
```

The approximation only depends on `p`, not on `N` — it just scales linearly with the pool size:

| Fraction of prefixes seen | Multiplier (`× N`) |
|---|---|
| 50% | `ln(2)` ≈ 0.69 |
| 75% | `ln(4)` ≈ 1.39 |
| 95% | `ln(20)` ≈ 3.00 |

In other words, you need roughly **0.7×N requests** to see half of all unique prefixes, **1.4×N** to see three-quarters, and **3×N** to see 95% — the long tail of rarely-drawn combinations dominates. Concretely, for `docs_per_request=3,4,5`:

**`docs_per_request=3`**

| Pool size | `N` | Requests for 50% | Requests for 75% | Requests for 95% |
| --- | --- | --- | --- | --- |
| 10 | 120 | 83 | 165 | 350 |
| 20 | 1,140 | 790 | 1,579 | 3,406 |
| 30 | 4,060 | 2,814 | 5,627 | 12,153 |
| 40 | 9,880 | 6,848 | 13,695 | 29,588 |
| 50 | 19,600 | 13,585 | 27,170 | 58,707 |
| 60 | 34,220 | 23,719 | 47,437 | 102,504 |
| 70 | 54,740 | 37,942 | 75,884 | 163,977 |
| 80 | 82,160 | 56,948 | 113,896 | 246,120 |
| 90 | 117,480 | 81,430 | 162,860 | 351,929 |
| 100 | 161,700 | 112,081 | 224,162 | 484,400 |

**`docs_per_request=4`**

| Pool size | `N` | Requests for 50% | Requests for 75% | Requests for 95% |
| --- | --- | --- | --- | --- |
| 10 | 210 | 145 | 292 | 630 |
| 20 | 4,845 | 3,359 | 6,716 | 14,510 |
| 30 | 27,405 | 18,996 | 37,991 | 82,094 |
| 40 | 91,390 | 63,346 | 126,694 | 273,780 |
| 50 | 230,300 | 159,631 | 319,262 | 689,908 |
| 60 | 487,635 | 338,003 | 676,007 | 1,460,829 |
| 70 | 916,895 | 635,544 | 1,271,088 | 2,746,777 |
| 80 | 1,581,580 | 1,096,267 | 2,192,534 | 4,737,981 |
| 90 | 2,555,190 | 1,771,122 | 3,542,246 | 7,654,666 |
| 100 | 3,921,225 | 2,717,987 | 5,435,972 | 11,746,936 |

**`docs_per_request=5`**

| Pool size | `N` | Requests for 50% | Requests for 75% | Requests for 95% |
| --- | --- | --- | --- | --- |
| 10 | 252 | 174 | 348 | 757 |
| 20 | 15,504 | 10,746 | 21,492 | 46,440 |
| 30 | 142,506 | 98,777 | 197,556 | 426,906 |
| 40 | 658,008 | 456,096 | 912,191 | 1,971,214 |
| 50 | 2,118,760 | 1,468,612 | 2,937,224 | 6,347,228 |
| 60 | 5,461,512 | 3,785,631 | 7,571,262 | 16,361,230 |
| 70 | 12,103,014 | 8,389,170 | 16,778,341 | 36,257,394 |
| 80 | 24,040,016 | 16,663,269 | 33,326,537 | 72,017,458 |
| 90 | 43,949,268 | 30,463,311 | 60,926,621 | 131,660,239 |
| 100 | 75,287,520 | 52,185,332 | 104,370,663 | 225,541,244 |

Note this is an upper bound on real warm-up time in practice: a finite/evicting cache (the `-HBM-DRAM` / `-HBM-DRAM-DFS` configs) reaches a *different* steady state — recently-evicted prefixes can be re-requested as cache misses even after their first hit, so tiered runs don't strictly converge to "95% of prefixes seen" in the same sense. The numbers above describe the warm-up of the *unbounded* case only, useful as a lower bound on `total_requests` needed before steady-state hit-rate statistics become meaningful.
