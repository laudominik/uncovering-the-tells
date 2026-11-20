def suffix_match_lengths(tokens, context):
    lengths = [0] * len(tokens)
    alive = list(range(len(tokens)))
    n = 1
    while alive and len(context) >= n:
        grams = {tuple(context[i:i + n]) for i in range(len(context) - n + 1)}
        alive = [i for i in alive if i >= n - 1 and tuple(tokens[i - n + 1:i + 1]) in grams]
        for i in alive:
            lengths[i] = n
        n += 1
    return lengths


def lcs_positions(tokens, context, budget, ngram=2, skip=10):
    lengths = suffix_match_lengths(tokens, context)
    ranked = sorted(
        (j for j in range(1, len(tokens)) if j > skip and lengths[j] >= ngram),
        key=lambda j: (-lengths[j], j),
    )
    return sorted(ranked[:budget])
