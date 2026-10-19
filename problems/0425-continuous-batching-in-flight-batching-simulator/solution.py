def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """
    Simulate continuous batching for LLM inference.
    
    Args:
        requests: List of dicts with 'arrival_time' (int) and 'tokens_needed' (int)
        max_batch_size: Maximum concurrent sequences in the batch
    
    Returns:
        Dict with 'total_time', 'avg_latency', 'avg_ttft', 'throughput'
    """
    requests.sort(key=lambda r: r['arrival_time'])

    i, t = 0, 0
    n = len(requests)
    total_latencies, total_ttft, total_tokens = 0, 0, 0
    pool: dict = {} # id -> tokens remains
    while i < n or pool:
        while i < n and t >= requests[i]['arrival_time'] and len(pool) < max_batch_size:
            pool[i] = requests[i]['tokens_needed']
            total_ttft += t - requests[i]['arrival_time']
            total_tokens += requests[i]['tokens_needed']
            i += 1

        new_pool = {}
        for j in pool:
            if pool[j] > 1:
                new_pool[j] = pool[j] - 1
            else:
                total_latencies += t - requests[j]['arrival_time']
                
        pool = new_pool

        if len(pool) == 0 and i < n and requests[i]['arrival_time'] > t + 1:
            t = requests[i]['arrival_time']
        else:
            t += 1
    
    avg_latency = round(total_latencies / n, 4)
    avg_ttft = round(total_ttft / n, 4)
    throughput = round(total_tokens / t, 4)

    return {
        'total_time': t,
        'avg_latency': avg_latency,
        'avg_ttft': avg_ttft,
        'throughput': throughput
    }
        

