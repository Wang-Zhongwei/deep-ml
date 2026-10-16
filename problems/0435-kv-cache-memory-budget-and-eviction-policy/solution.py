import heapq

def kv_cache_manager(num_layers: int, num_heads: int, head_dim: int, dtype_bytes: int, memory_budget_bytes: int, token_ids: list, token_scores: list, eviction_policy: str, num_protected: int = 0) -> dict:
    """
    Simulate a KV cache with memory budget and eviction policy.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        head_dim: Dimension per attention head
        dtype_bytes: Bytes per element (e.g., 2 for FP16)
        memory_budget_bytes: Total KV cache memory budget in bytes
        token_ids: List of token IDs arriving sequentially
        token_scores: List of importance scores per token
        eviction_policy: 'fifo' or 'score'
        num_protected: Number of initial tokens protected from eviction
    
    Returns:
        Dictionary with bytes_per_token, max_tokens, final_cache,
        evicted_tokens, and num_evictions
    """
    bytes_per_token = num_layers * num_heads * 2 * head_dim * dtype_bytes
    max_tokens = memory_budget_bytes // bytes_per_token

    def enqueue(q: list, i: int, eviction_policy: str):
        if eviction_policy == 'fifo':
            q.append(i)
        else:
            heapq.heappush(q, (token_scores[i], i))
    
    def dequeue(q: list, eviction_policy: str):
        if eviction_policy == 'fifo':
            i = q.pop(0)
        else:
            _, i = heapq.heappop(q)
        return token_ids[i]

    def get_final_cache(q:list, num_protected: int):
        final_cache = token_ids[:num_protected]

        insertion_indices = []
        for i in range(len(q)):
            if eviction_policy == 'fifo':
                insertion_indices.append(q[i])
            else:
                insertion_indices.append(q[i][1])
        
        insertion_indices.sort()

        for i in insertion_indices:
            final_cache.append(token_ids[i])

        return final_cache

    q = []
    evicted_tokens = []
    for i in range(num_protected, len(token_ids)):
        # eviction first to make room for the new token
        if q and len(q) + num_protected >= max_tokens:
            evicted_tokens.append(dequeue(q, eviction_policy))

        # add new token
        enqueue(q, i, eviction_policy)
    
    return {
        'bytes_per_token': bytes_per_token,
        'max_tokens': max_tokens,
        'final_cache': get_final_cache(q, num_protected),
        'evicted_tokens': evicted_tokens,
        'num_evictions': len(evicted_tokens)
    }
