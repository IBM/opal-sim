# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple
import numpy as np

from opal.core.request import LLMRequest, LLMRequestStats
from opal.utils.util import sample_series_K


class StageStatistics:
    """Record latency statistics"""

    def __init__(self):
        min_bin = 1000  # 1s
        max_bin = 10 * 1000 * 1000
        self.bins = []
        val = min_bin
        self.total_latency = 0
        self.raw_latency_values = []
        self.raw_queuing_values = []
        self.raw_kvc_io_time = []
        self.raw_kvc_hit_tokens = []
        self.raw_gpu_time = []
        self.raw_ttft_values = []
        self.raw_decode_values: List[List[int]] = []
        # per serving tier (apc/CPUMemory/LocalNVMe/DistributedFS) -> list of
        # prefix-hit tokens, one entry per request that hit that tier.
        self.raw_kvc_hit_tokens_per_tier: dict[str, list] = defaultdict(list)
        # per serving tier (apc/CPUMemory/LocalNVMe/DistributedFS) -> list of
        # prefix-hit tokens, one entry per request that hit that tier.
        self.raw_kvc_hit_tokens_per_tier: dict[str, list] = defaultdict(list)
        self.debug_val = []
        self.input_output_tokens_sz: list[Tuple[int, int]] = []
        self.finished_requests = 0
        self.queued_requests = 0
        self.failed_requests = 0
        self.kvc_tier_tokens = defaultdict(int)
        self.per_unit_req_done = []
        self.per_unit_workers = []
        self.per_unit_gpu_utilization = []
        while val < max_bin:
            self.bins.append(val)
            val *= 2
        self.latencies = np.zeros(len(self.bins))
        self.stage_time_start = -1
        self.stage_time_end = -1

    def _ret_average_max(arr, label):
        pass

    def to_dict(self):
        return {
            "bins": self.bins,
            "total_latency": self.total_latency,
            "raw_latency_values": self.raw_latency_values,
            "raw_queuing_values": self.raw_queuing_values,
            "raw_kvc_io_time": self.raw_kvc_io_time,
            "raw_kvc_hit_tokens": self.raw_kvc_hit_tokens,
            "raw_gpu_time": self.raw_gpu_time,
            "raw_ttft_values": self.raw_ttft_values,
            "raw_decode_values": self.raw_decode_values,
            "debug_val": self.debug_val,
            "finished_requests": self.finished_requests,
            "queued_requests": self.queued_requests,
            "failed_requests": self.failed_requests,
            "per_unit_throughput": self.per_unit_req_done,
            "per_unit_workers": self.per_unit_workers,
            "per_unit_gpu_utilization": self.per_unit_gpu_utilization,
            # numpy array -> list
            "latencies": self.latencies.tolist(),
            "stage_time_start": self.stage_time_start,
            "stage_time_end": self.stage_time_end,
            "input_output_tokens_sz": self.input_output_tokens_sz,
            "kvc_tier_tokens": dict(self.kvc_tier_tokens),
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls()  # calls __init__ to create bins & latencies shape

        # Now overwrite with loaded values
        obj.bins = data["bins"]
        obj.total_latency = data["total_latency"]
        obj.raw_latency_values = data["raw_latency_values"]
        obj.raw_queuing_values = data["raw_queuing_values"]
        obj.raw_kvc_io_time = data["raw_kvc_io_time"]
        obj.raw_kvc_hit_tokens = data.get("raw_kvc_hit_tokens", [])
        obj.raw_gpu_time = data["raw_gpu_time"]
        obj.raw_ttft_values = data["raw_ttft_values"]
        obj.raw_decode_values = data["raw_decode_values"]
        obj.debug_val = data["debug_val"]
        obj.finished_requests = data["finished_requests"]
        obj.per_unit_req_done = data["per_unit_throughput"]
        obj.per_unit_workers = data["per_unit_workers"]
        obj.per_unit_gpu_utilization = data["per_unit_gpu_utilization"]
        obj.stage_time_end = data["stage_time_end"]
        obj.stage_time_start = data["stage_time_start"]
        obj.kvc_tier_tokens = defaultdict(int, data.get("kvc_tier_tokens", {}))

        # convert list -> numpy array
        obj.latencies = np.array(data["latencies"], dtype=float)

        return obj

    def dump_json_compact_lists(obj, indent=4):
        import json

        """
        Convert dict to JSON string where:
        - Each key is on a new line
        - All lists/arrays are kept on one line
        Works recursively for nested dicts
        """

        def _convert_numpy(o):
            """Convert numpy types to native Python types"""
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, tuple):
                return tuple(_convert_numpy(i) for i in o)
            elif isinstance(o, list):
                return [_convert_numpy(i) for i in o]
            elif isinstance(o, dict):
                return {k: _convert_numpy(v) for k, v in o.items()}
            return o

        def _dump(o, level=0):
            space = " " * (indent * level)
            if isinstance(o, dict):
                items = []
                for k, v in o.items():
                    items.append(f'{space}{" " * indent}"{k}": {_dump(v, level + 1)}')
                return "{\n" + ",\n".join(items) + f"\n{space}}}"
            elif isinstance(o, list):
                # keep lists on one line, convert numpy types first
                converted = [_convert_numpy(i) for i in o]
                return "[" + ", ".join(json.dumps(i) for i in converted) + "]"
            else:
                return json.dumps(_convert_numpy(o))

        return _dump(obj)

    def write_to_json(self, json_file: str):
        json_str = StageStatistics.dump_json_compact_lists(self.to_dict())
        with open(json_file, "w") as json_file:
            json_file.write(json_str)

    def read_from_json(cls, filepath):
        import json

        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def print_simend_stats(self):
        raw_ttft_values = np.sort(self.raw_ttft_values)
        raw_queuing_values = np.sort(self.raw_queuing_values)
        raw_latency_values = np.sort(self.raw_latency_values)
        raw_kvc_io_time = np.sort(self.raw_kvc_io_time)
        raw_gpu_time = np.sort(self.raw_gpu_time)
        debug_val = np.sort(self.debug_val)

        # printout various metrics that we have collected
        print("Total entires : ", len(raw_latency_values))
        print("E2E average : ", np.average(raw_latency_values))
        print("W_q         : ", np.average(raw_queuing_values))
        print(
            "TTFT        : ",
            np.average(raw_ttft_values),
            f"( max on TTFT {raw_ttft_values[-1]}, {raw_ttft_values[-2]})",
        )
        print(
            " + avg. kvc time :",
            np.average(raw_kvc_io_time),
            f"( max on kvc time {raw_kvc_io_time[-1]}, {raw_kvc_io_time[-2]})",
        )
        print(
            " + avg GPU time  : ",
            np.average(raw_gpu_time),
            f"( max on GPU time {raw_gpu_time[-1]}, {raw_gpu_time[-2]})",
        )
        print(
            " + avg xxx time  : ",
            np.average(debug_val),
            f"( max on xxx time {debug_val[-1]}, {debug_val[-2]})",
        )
        print(
            "Average request/sec is",
            np.average(self.per_unit_req_done),
            f"(max is : {np.sort(self.per_unit_req_done)[-1]})",
        )

    def add_finished_request(self, request: LLMRequest):
        """This is called to insert individual request statistics into the system at the
        end of the request processing.

        Args:
            stats (LLMRequestStats): LLMRequestStats type, typically found inside the finished
            LLMRequest
        """
        stats: LLMRequestStats = request.stats
        self.add_total_time_in_system({"E2E": stats.get_total_worker_time()})
        self.raw_queuing_values.append(stats.get_queue_time())
        self.raw_ttft_values.append(stats.get_ttft())
        self.raw_kvc_io_time.append(stats.get_kvc_fetch_time())
        self.raw_kvc_hit_tokens.append(stats.get_prefix_hit_tokens())
        self.raw_gpu_time.append(stats.get_gpu_time())
        # self.debug_val.append(stats.start_gpu_time - stats.end_kvc_time)
        self.raw_decode_values.append(stats.get_decode_times_including_ttft())
        self.input_output_tokens_sz.append((request.input_length, request.output_length))
        for tier, n in stats.get_kvc_tier_tokens().items():
            self.kvc_tier_tokens[tier] += n

    def add_total_time_in_system(self, breakdown: dict[str, float]):
        """Add a new request latency -- the total time it has spent in the system = routing + queuing + KVC + GPU"""
        total_lat = sum(breakdown.values())
        self.raw_latency_values.append(total_lat)
        bin_id = 0
        while total_lat > self.bins[bin_id] and bin_id < len(self.bins):
            bin_id += 1
        self.latencies[bin_id] += 1
        self.total_latency += total_lat
        self.finished_requests += 1

    def add_per_unit_gpu_utilization(self, avg_gpu_util: float = 0.0):
        self.per_unit_gpu_utilization.append(avg_gpu_util)

    def add_per_unit_workers(self, worker_count: int):
        self.per_unit_workers.append(worker_count)

    def add_per_unit_workdone(self, done: int):
        self.per_unit_req_done.append(done)

    def sample_workdone_per_K(self, k: int = 1):
        return sample_series_K(self.per_unit_req_done, k)

    def get_average_latency(self):
        return self.total_latency / self.finished_requests

    def get_histogram(self):
        """Generate latency pdf"""
        if self.finished_requests:
            return self.latencies * 100 / self.finished_requests
        else:
            return self.latencies

    def get_histogram_breakdown(self, num_bins: int = 10):
        """Generate latency pdf"""
        if not self.latencies:
            return []

        totals = np.array([sum(b.values()) for b in self.latencies])
        hist, bin_edges = np.histogram(totals, bins=num_bins)
        bin_indices = np.digitize(totals, bin_edges) - 1

        # Collect all unique sources
        all_sources = set()
        for b in self.latencies:
            all_sources.update(b.keys())
        all_sources = sorted(all_sources)  # Sort for consistent order

        # Sum breakdowns per bin
        bin_sums = [defaultdict(float) for _ in range(num_bins)]
        bin_counts = [0] * num_bins
        for idx, bin_idx in enumerate(bin_indices):
            bin_counts[bin_idx] += 1
            for source, value in self.latencies[idx].items():
                bin_sums[bin_idx][source] += value

        # Build results
        results = []
        for i in range(num_bins):
            count = bin_counts[i]
            if count == 0:
                avg_breakdown = {s: 0.0 for s in all_sources}
            else:
                avg_breakdown = {s: bin_sums[i].get(s, 0.0) / count for s in all_sources}
            results.append(
                {
                    "bin_low": bin_edges[i],
                    "bin_high": bin_edges[i + 1],
                    "count": count,
                    "average_breakdown": avg_breakdown,
                }
            )
        return results

    def _calculate_itl_tpot(self, request_arrays):
        """
        Calculates detailed metrics for LLM decoding.

        generated from Gemini.

        Args:
            request_arrays: List of lists, where each inner list contains:
                            [TTFT, decode_1, decode_2, ..., decode_N]
        """
        all_itls = []  # Stores every single inter-token gap for ITL
        request_tpots = []  # Stores the average decoding speed for each request for TPOT

        for timings in request_arrays:
            # Step 1: Isolate decoding steps (remove prefill/TTFT at index 0)
            # if we have more than one entry, then use 1: otherwise, we strip the entry and put -1
            decode_steps = timings[1:] if len(timings) > 1 else [-1]

            if len(decode_steps) > 0:
                # ITL data: Add all individual gaps to the global pool
                all_itls.extend(decode_steps)

                # TPOT data: Calculate the mean for this specific request
                request_tpots.append(np.mean(decode_steps))

        # Helper to calculate statistics
        def get_stats(data):
            if not data:
                return {"mean": 0, "median": 0, "p99": 0}
            return {"mean": np.mean(data), "median": np.median(data), "p99": np.percentile(data, 99)}

        dict_return = {
            "TPOT": get_stats(request_tpots),  # Request-weighted (User UX focus)
            "ITL": get_stats(all_itls),  # Token-weighted (System capacity focus)
        }
        return request_tpots, all_itls

    def calculate_user_metrics_in_ms(self):
        if len(self.raw_ttft_values) == 0:
            # no requests were done here
            return [-1.0] * 9
        np_ttft = np.array(self.raw_ttft_values)
        np_tpot, np_itl = self._calculate_itl_tpot(self.raw_decode_values)
        """
        ravel() → view when possible (faster, less memory)
        flatten() → always copies
        
        in case when I want to skip the first element: 
            result = np.concatenate([np.array(sub[1:]) for sub in example])
        for now I am assuming that this will only contain elements for 2nd token onwards 
        """
        # we skip the first token in each series
        # np_decode = np.concatenate([np.array(sub[1:]) for sub in self.raw_decode_values])

        def return_triplet(x):
            return np.mean(x), np.median(x), np.percentile(x, 99)

        val = return_triplet(np_ttft), return_triplet(np_tpot), return_triplet(np_itl)
        # while flattening, convert from seconds -> mseconds, *1000
        flat = tuple(1000 * float(x) if float(x) > 0 else float(x) for sub in val for x in sub)
        return flat

    def __print(self, dict_label_value, label_width=40, value_width=15, log_file=None):
        """
        Expected input as a format of :
        "label1" : value1
        "label2" : value2
        ...
        """
        for label, value in dict_label_value.items():
            line = f"{label}" if value is None else f"{label:<{label_width}}: {value:>{value_width},.2f}"
            print(line)
            if log_file is not None:
                print(line, file=log_file)

    def print_summary_results(self, log_file=None):
        """The format should be something what vllm serve generates, something like:

        ============ Serving Benchmark Result ============
        Successful requests:                     998
        Failed requests:                         2
        Benchmark duration (s):                  86.00
        Total input tokens:                      214249
        Total generated tokens:                  199536
        Request throughput (req/s):              11.60
        Output token throughput (tok/s):         2320.12
        Peak output token throughput (tok/s):    3200.00
        Peak concurrent requests:                998.00
        Total Token throughput (tok/s):          4811.32
        ---------------Time to First Token----------------
        Mean TTFT (ms):                          33351.37
        Median TTFT (ms):                        33633.66
        P99 TTFT (ms):                           70589.64
        -----Time per Output Token (excl. 1st token)------
        Mean TPOT (ms):                          50.69
        Median TPOT (ms):                        50.04
        P99 TPOT (ms):                           99.75
        ---------------Inter-token Latency----------------
        Mean ITL (ms):                           47.87
        Median ITL (ms):                         37.87
        P99 ITL (ms):                            94.47
        """
        # these are processed
        final_stats = {}
        successful_requests = self.finished_requests
        total_requests = self.queued_requests
        failed_requests = max(self.failed_requests, total_requests - successful_requests)
        benchmark_duration = self.stage_time_end - self.stage_time_start
        if benchmark_duration == 0:
            # for the failed staged, it will be zero
            return

        total_input_tokens = 0
        total_output_tokens = 0
        for i, o in self.input_output_tokens_sz:
            total_input_tokens += i
            total_output_tokens += o
        req_per_sec = total_requests / benchmark_duration
        output_tokens_per_sec = total_output_tokens / benchmark_duration
        peak_output_tokens_sec = -1
        peak_concurrent_requests_finished_per_sec = -1
        total_token_per_sec = (total_input_tokens + total_output_tokens) / benchmark_duration
        res = self.calculate_user_metrics_in_ms()
        # TTFT calculations
        mean_TTFT = res[0]
        median_TTFT = res[1]
        P99_TTFT = res[2]
        # TPOT calculations
        mean_TPOT = res[3]
        median_TPOT = res[4]
        P99_TPOT = res[5]
        # ITL calculations
        mean_ITL = res[6]
        median_ITL = res[7]
        P99_ITL = res[8]

        # KVC per-tier token breakdown. Records raw tokens served per tier and percent of total prefill tokens.
        kvc_tier_breakdown = {}
        total_tier_tokens = sum(self.kvc_tier_tokens.values())
        if total_tier_tokens > 0:
            kvc_tier_breakdown["--------------KVC Tier Token Breakdown-------------"] = None
            ordered_tiers = [t for t in sorted(self.kvc_tier_tokens) if t != "Cache Miss"]
            if "Cache Miss" in self.kvc_tier_tokens:
                ordered_tiers.append("Cache Miss")
            for tier in ordered_tiers:
                toks = self.kvc_tier_tokens[tier]
                pct = toks * 100.0 / total_tier_tokens
                kvc_tier_breakdown[f"{tier} ({pct:.2f}%)"] = toks

        final_stats = {
            "============ Serving Benchmark Result ============": None,
            "Note: negative values means that no sensible values can be calculated or just NYI. ": None,
            "--------------------------------------------------": None,
            "Successful requests": successful_requests,
            "Failed requests": failed_requests,
            "Benchmark duration (s)": benchmark_duration,
            "Total input tokens": total_input_tokens,
            "Total generated tokens": total_output_tokens,
            "Request throughput (req/s)": req_per_sec,
            "Output token throughput (tok/s)": output_tokens_per_sec,
            "Peak output token throughput (tok/s)": peak_output_tokens_sec,
            "Peak concurrent requests": peak_concurrent_requests_finished_per_sec,
            "Total Token throughput (tok/s)": total_token_per_sec,
            " ---------------Time to First Token----------------": None,
            "Mean TTFT (ms)": mean_TTFT,
            "Median TTFT (ms)": median_TTFT,
            "P99 TTFT (ms)": P99_TTFT,
            " -----Time per Output Token (excl. 1st token)------": None,
            "Mean TPOT (ms)": mean_TPOT,
            "Median TPOT (ms)": median_TPOT,
            "P99 TPOT (ms)": P99_TPOT,
            "---------------Inter-token Latency----------------": None,
            "Mean ITL (ms)": mean_ITL,
            "Median ITL (ms)": median_ITL,
            "P99 ITL (ms)": P99_ITL,
            **kvc_tier_breakdown,
            "*--------------------------------------------------": None,
        }
        self.__print(final_stats)
        self.print_kvc_tier_hit_rates()

    def print_kvc_tier_hit_rates(self):
        """Per-tier prefix-cache hit rate, ordered fastest (GPU/APC) -> slowest (DFS).

        The rate is token-weighted: tier_tokens / total_input_tokens. Because
        every matched token is attributed to exactly one tier (see
        OpalKVCacheEngine.lookup), the per-tier rates form a clean split -- they
        sum to the overall prefix-cache hit rate and never exceed 100%.
        """
        if not self.raw_kvc_hit_tokens_per_tier:
            return
        total_input_tokens = sum(i for i, _ in self.input_output_tokens_sz)
        if total_input_tokens == 0:
            return

        # fastest -> slowest; unknown tiers sort last, alphabetically.
        tier_order = ["apc", "CPUMemory", "LocalNVMe", "DistributedFS"]

        def tier_sort_key(item):
            name = item[0]
            return (tier_order.index(name) if name in tier_order else len(tier_order), name)

        tier_stats = {"---KVC Tokens Fetched per Tier (GPU->CPU->NVMe->DFS)---": None}
        for tier_name, values in sorted(self.raw_kvc_hit_tokens_per_tier.items(), key=tier_sort_key):
            arr = np.array(values, dtype=float)
            total_tokens = float(arr.sum())
            mean_tokens = float(arr.mean()) if arr.size else 0.0
            tier_stats[f"  [{tier_name}] hit rate (%)"] = total_tokens / total_input_tokens * 100
            tier_stats[f"  [{tier_name}] total tokens fetched"] = total_tokens
            tier_stats[f"  [{tier_name}] mean tokens/request"] = mean_tokens
        tier_stats["*--------------------------------------------------"] = None
        self.__print(tier_stats)
        self.print_kvc_tier_hit_rates()

    def print_kvc_tier_hit_rates(self):
        """Per-tier prefix-cache hit rate, ordered fastest (GPU/APC) -> slowest (DFS).

        The rate is token-weighted: tier_tokens / total_input_tokens. Because
        every matched token is attributed to exactly one tier (see
        OpalKVCacheEngine.lookup), the per-tier rates form a clean split -- they
        sum to the overall prefix-cache hit rate and never exceed 100%.
        """
        if not self.raw_kvc_hit_tokens_per_tier:
            return
        total_input_tokens = sum(i for i, _ in self.input_output_tokens_sz)
        if total_input_tokens == 0:
            return

        # fastest -> slowest; unknown tiers sort last, alphabetically.
        tier_order = ["apc", "CPUMemory", "LocalNVMe", "DistributedFS"]

        def tier_sort_key(item):
            name = item[0]
            return (tier_order.index(name) if name in tier_order else len(tier_order), name)

        tier_stats = {"---KVC Tokens Fetched per Tier (GPU->CPU->NVMe->DFS)---": None}
        for tier_name, values in sorted(self.raw_kvc_hit_tokens_per_tier.items(), key=tier_sort_key):
            arr = np.array(values, dtype=float)
            total_tokens = float(arr.sum())
            mean_tokens = float(arr.mean()) if arr.size else 0.0
            tier_stats[f"  [{tier_name}] hit rate (%)"] = total_tokens / total_input_tokens * 100
            tier_stats[f"  [{tier_name}] total tokens fetched"] = total_tokens
            tier_stats[f"  [{tier_name}] mean tokens/request"] = mean_tokens
        tier_stats["*--------------------------------------------------"] = None
        self.__print(tier_stats)
