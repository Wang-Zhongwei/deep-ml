def evaluate_preemption(
    num_tokens_list: list,
    kv_cache_bytes_per_token: int,
    swap_bandwidth_gbps: float,
    base_flops_per_token: int,
    attn_flops_per_token_pair: int,
    compute_tflops: float
) -> dict:
    """
    Evaluate preemption strategies (swap vs recompute) for a set of requests.

    Args:
        num_tokens_list: Number of tokens processed so far for each request
        kv_cache_bytes_per_token: Bytes of KV cache stored per token
        swap_bandwidth_gbps: GPU-CPU memory bandwidth in GB/s
        base_flops_per_token: FLOPs for non-attention layers per token during recomputation
        attn_flops_per_token_pair: FLOPs for attention per token pair during recomputation
        compute_tflops: GPU compute throughput in TFLOPS

    Returns:
        Dictionary with strategies, costs, and total cost
    """
    def swap_cost_ms(num_tokens: int) -> float:
        return 2 * num_tokens * kv_cache_bytes_per_token / (swap_bandwidth_gbps * 1e9) * 1000

    def recompute_cost_ms(num_tokens: int) -> float:
        flops = num_tokens * base_flops_per_token + num_tokens ** 2 * attn_flops_per_token_pair
        return flops / (compute_tflops * 1e12) * 1000

    strategies = []
    swap_costs_ms = []
    recompute_costs_ms = []
    total_cost = 0.0
    for num_tokens in num_tokens_list:
        swap_ms = swap_cost_ms(num_tokens)
        recompute_ms = recompute_cost_ms(num_tokens)

        swap_costs_ms.append(round(swap_ms, 4))
        recompute_costs_ms.append(round(recompute_ms, 4))

        # tie -> prefer swap
        if recompute_ms < swap_ms:
            strategies.append('recompute')
            total_cost += recompute_ms
        else:
            strategies.append('swap')
            total_cost += swap_ms

    return {
        'strategies': strategies,
        'swap_costs_ms': swap_costs_ms,
        'recompute_costs_ms': recompute_costs_ms,
        'total_cost_ms': round(total_cost, 4),
    }
