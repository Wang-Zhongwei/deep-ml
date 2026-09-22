def radix_prefix_cache(ops):
    """
    Process insert/lookup ops on a token-id prefix tree.

    Returns a list of longest-prefix match lengths, one per lookup.
    """
    root = {}
    match_lengths = []
    for op, tokens in ops:
        if op == 'insert':
            cur = root
            for t in tokens:
                cur = cur.setdefault(t, {})

        else:
            cur = root
            match_len = 0
            for t in tokens:
                if t in cur:
                    match_len += 1
                    cur = cur[t]
                else:
                    break
            match_lengths.append(match_len)
    return match_lengths

