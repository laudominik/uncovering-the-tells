import random


def contamination_score(n_helped, n_hurt):
    flips = n_helped + n_hurt
    return 0.0 if not flips else n_hurt / flips


def roc_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    positive_rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def group_auc(positive, negative):
    scores = list(positive) + list(negative)
    labels = [1] * len(positive) + [0] * len(negative)
    return roc_auc(scores, labels)


def tpr_at_fpr(scores, labels, target_fpr=0.05):
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    true_positives = false_positives = 0
    best = 0.0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            if pairs[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        if false_positives / negatives <= target_fpr:
            best = max(best, true_positives / positives)
    return best


def bootstrap_ci(statistic, *groups, n_boot=200, seed=0, level=0.95):
    groups = [list(group) for group in groups]
    if any(not group for group in groups):
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        resampled = [[group[rng.randrange(len(group))] for _ in group] for group in groups]
        value = statistic(*resampled)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    tail = (1 - level) / 2
    lo = values[max(0, min(len(values) - 1, int(round(tail * (len(values) - 1)))))]
    hi = values[max(0, min(len(values) - 1, int(round((1 - tail) * (len(values) - 1)))))]
    return (lo, hi)
