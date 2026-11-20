# Uncovering the Tells: Efficient Black-Box Audits for LLM Data Contamination

Dominik B. Lau\*, Filip Pawlicki\*, Jan Dubiński, Sebastian Cygert

NASK National Research Institute

\* Equal contribution

This repository contains the LCS audit from our paper. It checks whether a dataset was
part of a language model's training data.

## How it works

The idea is to measure how much in-context examples help the model. If the model has
never seen a dataset, showing it a few similar documents from that dataset helps it
predict the next token. If it has already memorised the dataset, the context adds little
and more often pulls it away from the continuation it already knows.

For each document in a dataset:

1. **Context.** We take the 3 most similar other documents from the same dataset (by
   overlap of token bigrams) and put their first 100 tokens in front of the document.
2. **Positions.** We pick the 16 positions where the document overlaps most with the
   context, i.e. where the context could actually change the prediction.
3. **Queries.** At each position we ask the model for the next token twice, with and
   without the context. A position counts as *helped* if the model is right only with
   the context, and *hurt* if it is right only without it.

The contamination score of a dataset is

```
S(D) = hurt / (helped + hurt)
```

summed over all its documents. A higher score means the dataset was more likely seen
in training.

## Installation

```bash
git clone https://github.com/NASK-AISafety/uncovering-the-tells.git
cd uncovering-the-tells
pip install -r requirements.txt
```

A GPU is recommended. For small models `--device cpu` also works.

## Usage

### Audit a dataset

Give the model and one or more datasets, and the audit prints a score for each:

```bash
python -m lcs_audit.run --model Qwen/Qwen2.5-0.5B \
    --datasets hf:openai/gsm8k:main:test:question,humaneval -n 20
```

```
audit  hf:openai/gsm8k:main:test:question S(D)=0.343 CI=(0.211, 0.457) n_helped=23 n_hurt=12 n_docs=20
audit  humaneval                S(D)=0.130 CI=(0.026, 0.282) n_helped=40 n_hurt=6 n_docs=20
```
There is no fixed threshold for `S(D)`, since its level depends on the model and the
type of text. Compare it with a dataset the model cannot have seen, ideally from a
similar domain and released after the model's training cutoff. A clearly higher score
is evidence of contamination. A similar score does not prove the opposite: the audit
can only detect exposure that left a trace in the model.

### Compare seen and unseen datasets

If you know which datasets the model was trained on, the audit also reports how well
it separates them. Here Pythia was trained on the Pile, while HumanEval and GSM8K were
released after its training cutoff:

```bash
python -m lcs_audit.run --model EleutherAI/pythia-410m \
    --seen pile_wikipedia,pile_github,pile_stackexchange,pile_pubmed_abstracts,pile_uspto_backgrounds \
    --unseen humaneval,gsm8k -n 30
```

```
seen   pile_wikipedia           S(D)=0.745 CI=(0.636, 0.840) n_helped=12 n_hurt=35 n_docs=30
seen   pile_github              S(D)=0.821 CI=(0.731, 0.929) n_helped=5 n_hurt=23 n_docs=28
seen   pile_stackexchange       S(D)=0.600 CI=(0.478, 0.704) n_helped=24 n_hurt=36 n_docs=30
seen   pile_pubmed_abstracts    S(D)=0.562 CI=(0.400, 0.778) n_helped=14 n_hurt=18 n_docs=30
seen   pile_uspto_backgrounds   S(D)=0.500 CI=(0.370, 0.610) n_helped=20 n_hurt=20 n_docs=30
unseen humaneval                S(D)=0.092 CI=(0.039, 0.158) n_helped=79 n_hurt=8 n_docs=30
unseen gsm8k                    S(D)=0.067 CI=(0.027, 0.107) n_helped=126 n_hurt=9 n_docs=30
document level: AUC=0.929 CI=(0.890, 0.966) TPR@5%FPR=0.399 (148 seen / 60 unseen documents)
dataset level:  AUC=1.000 TPR@5%FPR=1.000 (5 seen / 2 unseen datasets)
```

The document-level line treats every document as one point. The dataset-level line
treats every dataset as one point, scored by `S(D)`, which is the setting the audit is
designed for. It needs several datasets on each side to be meaningful.

### Datasets

A dataset can be given in three ways:

| Form | Example |
|---|---|
| Hugging Face dataset | `hf:repo`, `hf:repo:split:field` or `hf:repo:config:split:field` |
| Local JSONL file | `jsonl:data.jsonl`, with one `{"text": ...}` per line |
| Built-in name | `pile_wikipedia`, `pile_github`, `pile_stackexchange`, `pile_pubmed_abstracts`, `pile_uspto_backgrounds`, `humaneval`, `gsm8k` |

For Hugging Face datasets the split defaults to `train` and the field to `text`. They
are streamed, so only the needed rows are downloaded.

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `EleutherAI/pythia-410m` | Hugging Face model |
| `--datasets` | | datasets to audit without labels |
| `--seen`, `--unseen` | `pile_github`, `humaneval` | labelled datasets to compare |
| `-n` | 30 | documents per dataset |
| `-w` | 16 | positions per document (2 queries each) |
| `-k` | 3 | context documents |
| `--context-tokens` | 100 | tokens taken from each context document |
| `--max-tokens` | 340 | maximum document length in tokens |
| `--ngram` | 2 | minimum overlap with the context for a position to be picked |
| `--skip` | 10 | number of initial tokens never picked |
| `--n-boot` | 200 | bootstrap samples for confidence intervals |
| `--seed` | 0 | random seed |
| `--device` | auto | `cuda` or `cpu` |

## Auditing API models

The model is only used through `BlackBoxModel.query` in `lcs_audit/model.py`, which
returns the greedy next token for a given prefix. To audit a model behind an API,
replace this class with one that sends the prefix as a completion (or assistant
prefill) and returns the first generated token.

## Code structure

```
lcs_audit/
  run.py       command-line interface and evaluation
  policy.py    position selection
  context.py   context construction
  model.py     model wrapper
  data.py      dataset loading
  metrics.py   score, AUC, TPR and confidence intervals
```

## License

This project is released under the [MIT License](LICENSE).
