# LLM Programming — From RNNs to Pretrained LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<GITHUB_ID>/llm-programming/blob/main/labs/lab05_transformer.ipynb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![GPU](https://img.shields.io/badge/Colab-T4%2016GB-F9AB00?logo=googlecolab&logoColor=white)

Study log for the core building blocks of language models in PyTorch —
from recurrent models to attention, Transformers, and pretrained LLMs.

**Coursework:** *LLM Programming*, Dept. of Computer Science, Sangmyung University (Fall 2026)

---

## How I study

Each lab follows the course code line by line, then goes one step further:

- **Annotated code** — what each line does, with tensor shapes (`[batch, seq, dim]`)
- **Experiments** — change one thing, predict the result first, then measure it
- **Notes** in [`notes/`](notes/) — 3-line takeaways + key equations, shapes, and masking logic

## Labs

| # | Topic | What gets implemented | Notebook | Status |
|---|---|---|---|---|
| 02–04 | Prerequisites | N-gram, Word2Vec, RNN, LSTM → consolidated into `common/`, `models/` | — | 🚧 |
| 05 | Transformer | Scaled dot-product attention · causal & padding masks · Transformer block · RNN vs LSTM vs Attention vs Transformer (samples, #params, loss curves) · BeRP dataset | [Colab](https://colab.research.google.com/github/<GITHUB_ID>/llm-programming/blob/main/labs/lab05_transformer.ipynb) | 🚧 |
| 06 | Pretrained LLM | Tokenizer · Qwen2.5-1.5B loading · logits → next-token distribution · LoRA · RAG | [Colab](https://colab.research.google.com/github/<GITHUB_ID>/llm-programming/blob/main/labs/lab06_prellm.ipynb) | ⏳ |

## Structure

```
llm-programming/
├── common/      # shared package: vocab & DataLoader, training loop, sampling (temperature, top-k), device
├── models/      # rnn.py, lstm.py, attention.py, transformer.py
├── labs/        # one notebook per lab (Colab)
├── notes/       # per-lab log: what I changed, what happened, exam points
└── data/        # git-ignored
```

Notebooks import the shared code as `import common as cm`
(`cm.device`, `cm.loader`, `cm.train_model`, `cm.generate_text`, ...).

## Run

1. Click a Colab badge above.
2. Add a `GITHUB_TOKEN` secret in Colab (🔑 sidebar) — only needed for pushing.
3. Run the first cell: it clones/pulls this repo, sets `sys.path`, mounts Drive for the Hugging Face cache, and prints the GPU.

## Notes

- Lab code is based on the course materials of *LLM Programming* (Sangmyung University).
  Annotations, experiments, and notes are my own; parts reconstructed without a reference are marked `# [추측]`.
- Environment: Google Colab (free tier, NVIDIA T4).
