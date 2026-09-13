"""Track B's recurrent baselines: BiLSTM and BiGRU over pretrained fastText.

Answers adviser comment A13. Trained by ``training/finetune_rnn.py``.

Two choices carry the comparison's validity: ``AttentionPool`` is *imported*
from E5's model rather than reimplemented, so the encoder is the only thing
that differs between the tracks; and embeddings are pretrained fastText, since
random init would handicap the RNNs while reusing E5's own matrix would make
the two models non-independent.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from training.model import AttentionPool

#: Row 0 - ``nn.Embedding(padding_idx=0)`` hardcodes this index.
PAD = "<pad>"
UNK = "<unk>"

#: Word-level to match fastText. Keeps "don't" and "system-prompt" whole; split
#: fragments would miss the vector table.
TOKEN = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

FASTTEXT_DIM = 300


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)


def build_vocabulary(texts, *, max_size: int = 40_000, min_freq: int = 2
                     ) -> dict[str, int]:
    """Word -> row index, built from the **training split only**.

    Same leakage rule as the instruction-verb weight table: a vocabulary fitted
    over validation or test rows would not be held out. ``min_freq=2`` drops
    hapaxes, which here are mostly base64 fragments and hashes.
    """
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    vocabulary = {PAD: 0, UNK: 1}
    for word, count in counts.most_common():
        if count < min_freq or len(vocabulary) >= max_size:
            break
        vocabulary[word] = len(vocabulary)
    return vocabulary


def encode(texts, vocabulary: dict[str, int], max_length: int
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Texts -> ``(input_ids, attention_mask)``, padded to the batch's longest.

    Truncates at ``max_length``. Returns a mask, not a sentinel, so
    ``AttentionPool`` gets the same interface a HuggingFace tokenizer gives it.
    """
    unknown = vocabulary[UNK]
    sequences = [
        [vocabulary.get(word, unknown) for word in tokenize(text)[:max_length]]
        or [unknown]  # an empty prompt still needs one position to attend to
        for text in texts
    ]
    width = max(len(sequence) for sequence in sequences)

    input_ids = torch.zeros(len(sequences), width, dtype=torch.long)
    attention_mask = torch.zeros(len(sequences), width, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


def load_fasttext_matrix(vocabulary: dict[str, int], paths, *,
                         dim: int = FASTTEXT_DIM, seed: int = 42
                         ) -> tuple[torch.Tensor, dict]:
    """Build the embedding matrix, reading only the rows this vocabulary needs.

    The ``.vec`` files are gigabytes; they are streamed and all but the needed
    rows discarded, so peak memory is the matrix (~48 MB), not the file.

    ``paths`` are read in order and **earlier files win** - pass English before
    Tagalog. Unmatched words keep their random init rather than being zeroed, so
    they stay distinguishable from one another.

    Check the returned coverage: a low figure means the model is mostly training
    embeddings from scratch after all.
    """
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.normal(0.0, 0.1, (len(vocabulary), dim), generator=generator)
    matrix[vocabulary[PAD]] = 0.0

    # Two lookups mapping to *lists*: "Ignore" and "ignore" are separate rows
    # that want the same vector. A single dict keyed by word would let the first
    # casing claim the key and leave the other row random.
    exact: dict[str, list[int]] = {}
    lowered: dict[str, list[int]] = {}
    for word, index in vocabulary.items():
        exact.setdefault(word, []).append(index)
        lowered.setdefault(word.lower(), []).append(index)

    #: Rows filled by an exact spelling. An exact match may overwrite one a
    #: lowercase fallback filled, since the weaker match can arrive first.
    from_exact: set[int] = set()
    filled: set[int] = set()
    per_file: dict[str, int] = {}

    for path in paths:
        path = Path(path)
        hits = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline().split()
            # A .vec file opens with "<count> <dim>"; a headerless file does not.
            if not (len(first) == 2 and first[0].isdigit()):
                handle.seek(0)
            for line in handle:
                space = line.find(" ")
                if space < 0:
                    continue
                word = line[:space]

                # Both branches run: "ignore" is the exact match for row
                # "ignore" *and* the best vector for row "Ignore".
                exact_targets = [i for i in exact.get(word, ())
                                 if i not in from_exact]
                variant_targets = [i for i in lowered.get(word.lower(), ())
                                   if i not in filled and i not in exact_targets]
                if not exact_targets and not variant_targets:
                    continue

                values = line[space + 1:].split()
                if len(values) != dim:
                    continue
                vector = torch.tensor([float(v) for v in values])
                for index in exact_targets:
                    matrix[index] = vector
                    filled.add(index)
                    from_exact.add(index)
                for index in variant_targets:
                    matrix[index] = vector
                    filled.add(index)
                hits += 1
        per_file[path.name] = hits

    # Stays zero even if a vector file contains the literal "<pad>".
    matrix[vocabulary[PAD]] = 0.0
    filled.discard(vocabulary[PAD])

    return matrix, {
        "vocabulary_size": len(vocabulary),
        "covered": len(filled),
        "coverage": len(filled) / max(len(vocabulary), 1),
        "exact_matches": len(from_exact),
        "per_file": per_file,
    }


class BiRNNClassifier(nn.Module):
    """BiLSTM or BiGRU encoder -> shared AttentionPool -> single logit.

    Mirrors ``InjectionClassifier``'s interface exactly, so the trainer,
    evaluator and threshold tuner need no branch on model type.
    """

    CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, vocabulary_size: int, *, cell: str = "lstm",
                 embedding_dim: int = FASTTEXT_DIM, hidden_size: int = 256,
                 num_layers: int = 1, dropout: float = 0.1,
                 attention_size: int = 128, embedding_matrix=None,
                 freeze_embeddings: bool = False) -> None:
        super().__init__()
        if cell not in self.CELLS:
            raise ValueError(f"cell must be one of {sorted(self.CELLS)}, got {cell!r}")

        self.cell = cell
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        # Fine-tuned by default: 40,556 rows is enough to move "ignore" from
        # what it means on Common Crawl to what it means before a system prompt.
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.rnn = self.CELLS[cell](
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden_size * 2, attention_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                **_ignored) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(logits, attention_weights)``. Logits are raw.

        Packed before the RNN so recurrence stops at each row's real length.
        Without it, a short prompt batched with a long one runs its hidden state
        over hundreds of padding steps - a silent accuracy cost.
        """
        lengths = attention_mask.sum(dim=1)
        embedded = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _state = self.rnn(packed)
        # total_length pins the output width to the mask's, which pad_packed
        # otherwise shortens to the longest row in the batch - a silent shape
        # mismatch against the mask under gradient accumulation.
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=attention_mask.size(1))

        pooled, weights = self.pool(hidden, attention_mask)
        return self.head(self.dropout(pooled)), weights

    def parameter_count(self) -> dict[str, int]:
        """Total and trainable parameters, for the comparison table.

        The embedding table is broken out because it dominates the total (~12M)
        while costing only a lookup at inference.
        """
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embedding.weight.numel()
        return {
            "total": total,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "embedding": embedding,
            "recurrent_and_head": total - embedding,
        }
