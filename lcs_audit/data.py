import json
import random

from datasets import load_dataset

MIN_CHARS = 200
MAX_CHARS = 4000

PILE_SETS = {
    "pile_wikipedia": "Wikipedia (en)",
    "pile_github": "Github",
    "pile_stackexchange": "StackExchange",
    "pile_pubmed_abstracts": "PubMed Abstracts",
    "pile_uspto_backgrounds": "USPTO Backgrounds",
}


def _clean(texts, limit=None):
    out = []
    for text in texts:
        text = text.strip()
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
        if len(text) >= MIN_CHARS:
            out.append(text)
    return out[:limit] if limit else out


def load_pile(pile_set_name, n=30, seed=0):
    rows = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [row["text"] for row in rows if row["meta"]["pile_set_name"] == pile_set_name]
    random.Random(seed).shuffle(texts)
    return _clean(texts, n)


def load_humaneval(n=30):
    rows = load_dataset("openai/openai_humaneval", split="test")
    texts = [row["prompt"] + row["canonical_solution"] for row in rows]
    return _clean(texts, n)


def load_gsm8k(n=30):
    rows = load_dataset("openai/gsm8k", "main", split="test")
    texts = [row["question"] + "\n\n" + row["answer"] for row in rows]
    return _clean(texts, n)


def load_jsonl(path, n=None):
    texts = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts[:n] if n else texts


def load_hf(spec, n=30, seed=0):
    repo, *rest = spec.split(":")
    if not rest:
        rest = ["train", "text"]
    if len(rest) == 2:
        rest = [None] + rest
    if len(rest) != 3:
        raise SystemExit(f"hf:{spec}: use repo, repo:split:field or repo:config:split:field")
    config, split, field = rest
    rows = load_dataset(repo, config, split=split, streaming=True)
    rows = rows.shuffle(seed=seed, buffer_size=10_000)
    texts = []
    for row in rows:
        if field not in row:
            raise SystemExit(f"hf:{spec}: no field '{field}', available: {sorted(row)}")
        texts += _clean([str(row[field])])
        if n and len(texts) >= n:
            break
    return texts


def load_named(name, n=30, seed=0):
    if name.startswith("hf:"):
        return load_hf(name[len("hf:"):], n=n, seed=seed)
    if name in PILE_SETS:
        return load_pile(PILE_SETS[name], n=n, seed=seed)
    if name == "humaneval":
        return load_humaneval(n=n)
    if name == "gsm8k":
        return load_gsm8k(n=n)
    if name.startswith("jsonl:"):
        return load_jsonl(name[len("jsonl:"):], n=n)
    raise KeyError(name)
