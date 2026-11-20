import argparse

from . import data, metrics
from .context import build_context
from .model import BlackBoxModel
from .policy import lcs_positions


def audit_document(model, tokens, context, budget, ngram, skip):
    positions = lcs_positions(tokens, context, budget, ngram=ngram, skip=skip)
    if not positions:
        return None
    helped = hurt = 0
    for j in positions:
        target = tokens[j]
        hit_with = model.query(context + tokens[:j]) == target
        hit_without = model.query(tokens[:j]) == target
        if hit_with and not hit_without:
            helped += 1
        elif hit_without and not hit_with:
            hurt += 1
    mean_delta = (helped - hurt) / len(positions)
    return helped, hurt, mean_delta


def audit_corpus(model, texts, budget, k, tokens_per_doc, ngram, skip, max_tokens):
    total_helped = total_hurt = 0
    doc_scores = []
    doc_pairs = []
    for text in texts:
        tokens = model.encode(text)[:max_tokens]
        if len(tokens) < 2:
            continue
        context = build_context(model, text, tokens, texts, num_docs=k,
                                 tokens_per_doc=tokens_per_doc)
        result = audit_document(model, tokens, context, budget, ngram, skip)
        if result is None:
            continue
        helped, hurt, mean_delta = result
        total_helped += helped
        total_hurt += hurt
        doc_scores.append(-mean_delta)
        doc_pairs.append((helped, hurt))
    score = metrics.contamination_score(total_helped, total_hurt)
    return score, doc_scores, doc_pairs, total_helped, total_hurt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--seen", default="pile_github")
    parser.add_argument("--unseen", default="humaneval")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("-n", type=int, default=30)
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("-w", type=int, default=16)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--skip", type=int, default=10)
    parser.add_argument("--context-tokens", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=340)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = BlackBoxModel(args.model, device=args.device)

    def fmt(ci):
        return "n/a" if ci is None else f"({ci[0]:.3f}, {ci[1]:.3f})"

    def score_of(pairs):
        return metrics.contamination_score(sum(h for h, _ in pairs), sum(u for _, u in pairs))

    def audit_named(name, tag):
        texts = data.load_named(name, n=args.n, seed=args.seed)
        score, scores, pairs, helped, hurt = audit_corpus(
            model, texts, args.w, args.k, args.context_tokens, args.ngram,
            args.skip, args.max_tokens)
        ci = metrics.bootstrap_ci(score_of, pairs, n_boot=args.n_boot, seed=args.seed)
        print(f"{tag:6s} {name:24s} S(D)={score:.3f} CI={fmt(ci)} "
              f"n_helped={helped} n_hurt={hurt} n_docs={len(scores)}", flush=True)
        return score, scores

    if args.datasets:
        for name in args.datasets.split(","):
            audit_named(name, "audit")
        return

    doc_scores = {1: [], 0: []}
    dataset_scores = {1: [], 0: []}
    for label, names in ((1, args.seen), (0, args.unseen)):
        for name in names.split(","):
            score, scores = audit_named(name, "seen" if label else "unseen")
            doc_scores[label] += scores
            dataset_scores[label].append(score)

    seen_scores, unseen_scores = doc_scores[1], doc_scores[0]
    auc = metrics.group_auc(seen_scores, unseen_scores)
    auc_ci = metrics.bootstrap_ci(metrics.group_auc, seen_scores, unseen_scores,
                                   n_boot=args.n_boot, seed=args.seed)
    labels = [1] * len(seen_scores) + [0] * len(unseen_scores)
    tpr5 = metrics.tpr_at_fpr(seen_scores + unseen_scores, labels)
    print(f"document level: AUC={auc:.3f} CI={fmt(auc_ci)} TPR@5%FPR={tpr5:.3f} "
          f"({len(seen_scores)} seen / {len(unseen_scores)} unseen documents)")

    seen_sets, unseen_sets = dataset_scores[1], dataset_scores[0]
    if len(seen_sets) + len(unseen_sets) > 2:
        labels = [1] * len(seen_sets) + [0] * len(unseen_sets)
        print(f"dataset level:  AUC={metrics.group_auc(seen_sets, unseen_sets):.3f} "
              f"TPR@5%FPR={metrics.tpr_at_fpr(seen_sets + unseen_sets, labels):.3f} "
              f"({len(seen_sets)} seen / {len(unseen_sets)} unseen datasets)")
    else:
        print("dataset level:  pass several comma-separated datasets to --seen/--unseen")


if __name__ == "__main__":
    main()
