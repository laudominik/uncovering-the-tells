def shingles(ids, k=2):
    if len(ids) < k:
        return {tuple(ids)} if ids else set()
    return {tuple(ids[i:i + k]) for i in range(len(ids) - k + 1)}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def build_context(model, sample_text, sample_tokens, pool_texts, num_docs=3,
                   tokens_per_doc=100, shingle=2):
    prefixes = [
        model.encode(text)[:tokens_per_doc]
        for text in pool_texts if text != sample_text
    ]
    if not prefixes or num_docs <= 0:
        return []
    target = shingles(sample_tokens, shingle)
    scored = sorted(
        range(len(prefixes)),
        key=lambda i: (-jaccard(target, shingles(prefixes[i], shingle)), i),
    )
    context = []
    for i in scored[:num_docs]:
        context += prefixes[i]
    return context
