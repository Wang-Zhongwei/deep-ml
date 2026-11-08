def prefix_cache_hit_rate(prompts: list, cache: list) -> dict:
    """
    Calculate the prefix cache hit rate for a batch of tokenized prompts.
    For each prompt, take the longest cached prefix it starts with.
    """
    total_tokens = 0
    cached_tokens = 0
    for prompt in prompts:
        total_tokens += len(prompt)
        best = 0
        for prefix in cache:
            L = len(prefix)
            if best < L <= len(prompt) and prompt[:L] == prefix:
                best = L
        cached_tokens += best
    hit_rate = round(cached_tokens / total_tokens, 4) if total_tokens else 0.0
    return {
        'hit_rate': hit_rate,
        'cached_tokens': cached_tokens,
        'total_tokens': total_tokens,
    }
