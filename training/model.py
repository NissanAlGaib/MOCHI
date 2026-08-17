"""Stage II classifier: E5 encoder plus an attention-pooling head.

Imported by both the trainer and ``mochi.detect.stage2_semantic.E5Scorer``, so
the architecture that produced the weights is the architecture that loads them.
Torch is a hard dependency *of this module only* - the runtime imports it lazily.

Why attention pooling rather than the ``sentence-transformers`` default of mean
pooling:

    A malicious span occupies a measured median of 3.4% of its document in this
    project's corpus, and under 5% in 72% of locatable cases. Mean pooling
    produces ``0.966 * benign + 0.034 * attack``, so the document vector lands
    almost entirely in benign territory and the classifier sees nothing. This is
    the standard multiple-instance-learning problem: the bag carries the label,
    a few instances carry the evidence. Attention pooling learns per-token
    weights, letting a small malicious span dominate the pooled representation.

    The learned weights are also the explanation. They answer "which token
    contributed to the prediction" faithfully - they *are* the mechanism, not a
    post-hoc approximation - at no extra inference cost. Phase 10's SANITIZE
    action needs exactly this to know what to redact.

Reference: Ilse, Tomczak & Welling (2018), "Attention-based Deep Multiple
Instance Learning", ICML.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

#: Encoder to fine-tune. ``-small`` is chosen over ``-base``/``-large`` because
#: the thesis targets <55 ms semantic latency on CPU, which the larger variants
#: do not meet (~200 ms+ observed).
DEFAULT_ENCODER = "intfloat/multilingual-e5-small"

CONFIG_FILENAME = "mochi_head.json"
WEIGHTS_FILENAME = "mochi_head.pt"


class AttentionPool(nn.Module):
    """Gated attention pooling over token embeddings.

    Produces one document vector as a learned weighted sum of token vectors,
    plus the weights themselves for attribution. The gated form (a ``tanh``
    branch multiplied by a ``sigmoid`` branch) is Ilse et al.'s; it lets the
    layer both rank tokens and suppress them, which plain additive attention
    cannot do.
    """

    def __init__(self, hidden_size: int, attention_size: int = 128) -> None:
        super().__init__()
        self.value = nn.Linear(hidden_size, attention_size)
        self.gate = nn.Linear(hidden_size, attention_size)
        self.weight = nn.Linear(attention_size, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: ``(batch, tokens, hidden)`` encoder output.
            mask: ``(batch, tokens)`` attention mask, 1 for real tokens.

        Returns:
            ``(pooled, weights)`` of shapes ``(batch, hidden)`` and
            ``(batch, tokens)``.
        """
        scores = self.weight(torch.tanh(self.value(hidden))
                             * torch.sigmoid(self.gate(hidden))).squeeze(-1)
        # Padding must not receive attention mass; -inf zeroes it after softmax.
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights


class InjectionClassifier(nn.Module):
    """E5 encoder -> attention pool -> single logit."""

    def __init__(self, encoder_name: str = DEFAULT_ENCODER, *,
                 attention_size: int = 128, dropout: float = 0.1,
                 encoder=None) -> None:
        super().__init__()
        if encoder is None:
            encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.attention_size = attention_size
        hidden = self.encoder.config.hidden_size
        self.pool = AttentionPool(hidden, attention_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                **_ignored) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(logits, attention_weights)``.

        Logits are raw - callers apply ``sigmoid``. ``BCEWithLogitsLoss`` wants
        them raw too, and keeping the sigmoid out of the graph avoids the
        double-sigmoid bug that silently flattens gradients.
        """
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        pooled, weights = self.pool(hidden, attention_mask)
        return self.head(self.dropout(pooled)), weights

    # --- persistence ---

    def save(self, directory: str | Path) -> None:
        """Write encoder, head weights, and the architecture config together.

        The head is saved separately from the encoder because it is not part of
        the HuggingFace model; saving the config alongside is what makes
        :meth:`load` able to rebuild the exact architecture rather than guessing.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory)
        torch.save(
            {"pool": self.pool.state_dict(), "head": self.head.state_dict()},
            directory / WEIGHTS_FILENAME,
        )
        (directory / CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "encoder_name": self.encoder_name,
                    "attention_size": self.attention_size,
                    "hidden_size": self.encoder.config.hidden_size,
                    "max_tokens": 512,
                    "pooling": "gated_attention",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu"
             ) -> "InjectionClassifier":
        directory = Path(directory)
        config_path = directory / CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} missing - {directory} was not written by "
                "InjectionClassifier.save(). Re-export from training/finetune_e5.py."
            )
        spec = json.loads(config_path.read_text(encoding="utf-8"))

        encoder = AutoModel.from_pretrained(
            str(directory), config=AutoConfig.from_pretrained(str(directory))
        )
        model = cls(
            encoder_name=spec["encoder_name"],
            attention_size=spec["attention_size"],
            encoder=encoder,
        )
        state = torch.load(directory / WEIGHTS_FILENAME, map_location=device)
        model.pool.load_state_dict(state["pool"])
        model.head.load_state_dict(state["head"])
        return model.to(device)
