def kv_cache_offload_sim(tiers: list[dict], entries: list[dict], recompute_latency_us: float = 1000.0) -> dict:
	"""
	Simulate tiered KV cache offloading for LLM inference.

	Args:
		tiers: list of dicts with keys 'name' (str), 'capacity' (int), 'latency_us' (float)
		entries: list of dicts with keys 'id' (int), 'recency' (float)
		recompute_latency_us: latency penalty for evicted entries

	Returns:
		dict with 'placement', 'evicted', 'total_latency_us', 'avg_latency_us'
	"""
	# Your code here
	entries.sort(key=lambda x: -x['recency'])
	evicted = []
	tiers_info = {}
	for t in tiers:
		tiers_info[t['name']] = dict(placement=[], **t)

	tier_names = sorted([t['name'] for t in tiers], key=lambda name: tiers_info[name]['latency_us'])
	
	def tier_has_room(tier_name: str):
		return len(tiers_info[tier_name]['placement']) < tiers_info[tier_name]['capacity']
	
	def offload_to_tier(tier_name: str, id_: int):
		tiers_info[tier_name]['placement'].append(id_)

	for e in entries:

		for tier_name in tier_names:
			if tier_has_room(tier_name):
				offload_to_tier(tier_name, e['id'])
				break
		else:
			evicted.append(e['id'])
	
	total_latency_us = sum(len(v['placement']) * v['latency_us'] for v in tiers_info.values())
	total_latency_us += len(evicted) * recompute_latency_us

	avg_latency_us = round(total_latency_us/len(entries), 2) if len(entries) > 0 else 0.0


	return {
		'placement': {t['name']: sorted(t['placement']) for t in sorted(tiers_info.values(), key=lambda v: v['latency_us'])},
		'evicted': evicted,
		'total_latency_us': total_latency_us,
		'avg_latency_us': avg_latency_us
	}



