"""training/rnn_models.py - Track B's BiLSTM/BiGRU baselines.

Torch is not a default dependency (see ``requirements.txt``), so the whole
module skips without it rather than failing a fresh checkout. Nothing here
trains anything: these pin the plumbing that would otherwise fail silently -
padding that leaks into a document vector, a vocabulary that drifts, an
embedding matrix that quietly stayed random.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Track B requires torch")

from training.rnn_models import (  # noqa: E402
    PAD,
    UNK,
    BiRNNClassifier,
    build_vocabulary,
    encode,
    load_fasttext_matrix,
    tokenize,
)


# --- tokenisation ---------------------------------------------------------


def test_intra_word_punctuation_survives():
    """fastText has vectors for "don't" and "system-prompt" as whole words.
    Splitting them into fragments would miss the table for both halves.
    """
    assert tokenize("don't ignore the system-prompt") == [
        "don't", "ignore", "the", "system-prompt"]


def test_surrounding_punctuation_is_dropped():
    assert tokenize("Ignore, please! (really)") == [
        "Ignore", "please", "really"]


# --- vocabulary -----------------------------------------------------------


def test_reserved_rows_come_first():
    """Row 0 must be PAD because ``nn.Embedding(padding_idx=0)`` is hardcoded
    to it; a vocabulary that put a real word there would train that word's
    gradient to zero and silently degrade it.
    """
    vocabulary = build_vocabulary(["ignore the previous instructions"], min_freq=1)
    assert vocabulary[PAD] == 0
    assert vocabulary[UNK] == 1


def test_rare_words_are_dropped():
    """Hapax legomena in an injection corpus are mostly base64 fragments and
    hashes - an embedding row each, generalising to nothing.
    """
    texts = ["ignore ignore ignore", "aGVsbG93b3JsZGJhc2U2NA"]
    vocabulary = build_vocabulary(texts, min_freq=2)
    assert "ignore" in vocabulary
    assert "aGVsbG93b3JsZGJhc2U2NA" not in vocabulary


def test_max_size_is_respected_and_keeps_the_most_frequent():
    texts = ["a a a a b b b c c d"] * 3
    vocabulary = build_vocabulary(texts, max_size=4, min_freq=1)
    assert len(vocabulary) == 4
    # PAD, UNK, then the two most frequent words.
    assert "a" in vocabulary and "b" in vocabulary
    assert "d" not in vocabulary


def test_vocabulary_is_deterministic():
    texts = ["ignore the previous instructions", "please summarise this"]
    assert build_vocabulary(texts) == build_vocabulary(texts)


# --- encoding -------------------------------------------------------------


def test_padding_is_to_the_batch_longest_not_a_fixed_width():
    """The dynamic-padding claim. Prompts here have a median of ~88 tokens
    against a 512 cap, so a fixed width would do several times the necessary
    work - and attention is quadratic in it.
    """
    vocabulary = build_vocabulary(["a b c d e f"], min_freq=1)
    input_ids, _mask = encode(["a b", "a b c d"], vocabulary, 512)
    assert input_ids.shape[1] == 4


def test_mask_marks_exactly_the_real_tokens():
    vocabulary = build_vocabulary(["a b c"], min_freq=1)
    _ids, mask = encode(["a b", "a b c"], vocabulary, 512)
    assert mask[0].tolist() == [1, 1, 0]
    assert mask[1].tolist() == [1, 1, 1]


def test_truncation_respects_max_length():
    vocabulary = build_vocabulary(["a b c d e"], min_freq=1)
    input_ids, _mask = encode(["a b c d e"], vocabulary, 3)
    assert input_ids.shape[1] == 3


def test_unknown_words_become_unk():
    vocabulary = build_vocabulary(["known known"], min_freq=1)
    input_ids, _mask = encode(["mystery"], vocabulary, 512)
    assert input_ids[0, 0].item() == vocabulary[UNK]


def test_empty_text_still_yields_one_attendable_position():
    """AttentionPool softmaxes over the masked scores. A row with no unmasked
    position produces -inf everywhere and NaN out of the softmax, which then
    poisons the whole batch's loss.
    """
    vocabulary = build_vocabulary(["a b"], min_freq=1)
    _ids, mask = encode(["!!!"], vocabulary, 512)
    assert mask.sum().item() >= 1


# --- fastText matrix ------------------------------------------------------


def _write_vec(tmp_path, name, rows, dim=4, header=True):
    lines = [f"{len(rows)} {dim}"] if header else []
    lines += [f"{word} " + " ".join(str(v) for v in values)
              for word, values in rows.items()]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_matching_words_take_their_pretrained_vector(tmp_path):
    vocabulary = build_vocabulary(["ignore reveal"], min_freq=1)
    path = _write_vec(tmp_path, "wiki.en.vec", {"ignore": [1.0, 2.0, 3.0, 4.0]})

    matrix, report = load_fasttext_matrix(vocabulary, [path], dim=4)
    assert matrix[vocabulary["ignore"]].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert report["per_file"]["wiki.en.vec"] == 1


def test_unmatched_words_stay_random_rather_than_zero(tmp_path):
    """A zero row is a real point every unmatched word would share, teaching
    the model they are the same word. Random keeps them distinguishable.
    """
    vocabulary = build_vocabulary(["ignore mysteryword"], min_freq=1)
    path = _write_vec(tmp_path, "wiki.en.vec", {"ignore": [1.0, 2.0, 3.0, 4.0]})

    matrix, _report = load_fasttext_matrix(vocabulary, [path], dim=4)
    assert matrix[vocabulary["mysteryword"]].abs().sum() > 0


def test_padding_row_is_zero(tmp_path):
    vocabulary = build_vocabulary(["ignore"], min_freq=1)
    path = _write_vec(tmp_path, "wiki.en.vec", {"ignore": [1.0, 2.0, 3.0, 4.0]})
    matrix, _ = load_fasttext_matrix(vocabulary, [path], dim=4)
    assert matrix[vocabulary[PAD]].abs().sum() == 0


def test_earlier_file_wins_on_an_overlapping_word(tmp_path):
    """English is passed first deliberately: the two vocabularies overlap on
    loan words, and for an overwhelmingly English corpus the English vector is
    the better prior.
    """
    vocabulary = build_vocabulary(["ignore"], min_freq=1)
    english = _write_vec(tmp_path, "wiki.en.vec", {"ignore": [1.0, 1.0, 1.0, 1.0]})
    tagalog = _write_vec(tmp_path, "wiki.tl.vec", {"ignore": [9.0, 9.0, 9.0, 9.0]})

    matrix, _ = load_fasttext_matrix(vocabulary, [english, tagalog], dim=4)
    assert matrix[vocabulary["ignore"]].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_headerless_file_is_read_from_its_first_line(tmp_path):
    """A ``.vec`` opens with "<count> <dim>"; some mirrors ship without it. A
    seek that guessed wrong would silently drop the first word.
    """
    vocabulary = build_vocabulary(["ignore"], min_freq=1)
    path = _write_vec(tmp_path, "plain.vec", {"ignore": [5.0, 5.0, 5.0, 5.0]},
                      header=False)
    matrix, _ = load_fasttext_matrix(vocabulary, [path], dim=4)
    assert matrix[vocabulary["ignore"]].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_coverage_is_reported(tmp_path):
    """Low coverage means the model is mostly training embeddings from scratch
    after all - which would make the results answer a different question than
    the one the fastText decision was meant to answer.
    """
    vocabulary = build_vocabulary(["ignore reveal bypass"], min_freq=1)
    path = _write_vec(tmp_path, "wiki.en.vec", {"ignore": [1.0, 2.0, 3.0, 4.0]})
    _matrix, report = load_fasttext_matrix(vocabulary, [path], dim=4)
    assert 0 < report["coverage"] < 1
    assert report["vocabulary_size"] == len(vocabulary)


# --- the model ------------------------------------------------------------


@pytest.fixture(params=["lstm", "gru"])
def model(request):
    torch.manual_seed(0)
    return BiRNNClassifier(vocabulary_size=32, cell=request.param,
                           embedding_dim=8, hidden_size=6).eval()


def test_forward_returns_a_logit_per_row_and_a_weight_per_token(model):
    ids = torch.randint(2, 32, (3, 7))
    mask = torch.ones(3, 7, dtype=torch.long)
    logits, weights = model(input_ids=ids, attention_mask=mask)
    assert logits.shape == (3, 1)
    assert weights.shape == (3, 7)


def test_attention_weights_are_a_distribution_over_real_tokens(model):
    ids = torch.randint(2, 32, (2, 5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    _logits, weights = model(input_ids=ids, attention_mask=mask)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)


def test_padding_receives_no_attention_mass(model):
    ids = torch.randint(2, 32, (1, 6))
    mask = torch.tensor([[1, 1, 0, 0, 0, 0]])
    _logits, weights = model(input_ids=ids, attention_mask=mask)
    assert weights[0, 2:].abs().sum().item() == pytest.approx(0.0, abs=1e-6)


def test_padding_does_not_change_a_rows_prediction(model):
    """The property packing exists to guarantee. Without it a short prompt
    batched beside a long one would run its hidden state forward over hundreds
    of padding steps, and the vector reaching the pool would describe the
    padding. This is the one failure that costs accuracy without ever raising.
    """
    short = torch.tensor([[5, 6, 7]])
    short_mask = torch.tensor([[1, 1, 1]])
    alone, _ = model(input_ids=short, attention_mask=short_mask)

    padded = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0]])
    padded_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    with_padding, _ = model(input_ids=padded, attention_mask=padded_mask)

    assert torch.allclose(alone, with_padding, atol=1e-5)


def test_output_width_matches_the_mask_not_the_batch_longest(model):
    """``pad_packed_sequence`` shortens to the longest row in the batch unless
    ``total_length`` is pinned - a shape mismatch against the mask that would
    surface only on batches where every row is shorter than the mask width.
    """
    ids = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
    _logits, weights = model(input_ids=ids, attention_mask=mask)
    assert weights.shape[1] == 4


def test_pretrained_matrix_is_actually_installed():
    matrix = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    model = BiRNNClassifier(vocabulary_size=32, embedding_dim=8, hidden_size=4,
                            embedding_matrix=matrix)
    assert torch.allclose(model.embedding.weight.data, matrix)


def test_freezing_embeddings_excludes_them_from_trainable_parameters():
    frozen = BiRNNClassifier(vocabulary_size=32, embedding_dim=8, hidden_size=4,
                             freeze_embeddings=True)
    counts = frozen.parameter_count()
    assert counts["trainable"] == counts["total"] - counts["embedding"]


def test_parameter_count_separates_the_embedding_table():
    """The embedding matrix dominates the total but costs only a lookup at
    inference. A single total would make these models look far heavier than
    they run, which is exactly the column the comparison table reports.
    """
    model = BiRNNClassifier(vocabulary_size=1000, embedding_dim=300, hidden_size=4)
    counts = model.parameter_count()
    assert counts["embedding"] == 1000 * 300
    assert counts["total"] == counts["embedding"] + counts["recurrent_and_head"]


def test_an_unknown_cell_is_rejected_at_construction():
    with pytest.raises(ValueError, match="cell must be one of"):
        BiRNNClassifier(vocabulary_size=32, cell="transformer")


def test_lstm_and_gru_are_different_models():
    torch.manual_seed(0)
    lstm = BiRNNClassifier(vocabulary_size=32, cell="lstm", embedding_dim=8,
                           hidden_size=6).parameter_count()
    torch.manual_seed(0)
    gru = BiRNNClassifier(vocabulary_size=32, cell="gru", embedding_dim=8,
                          hidden_size=6).parameter_count()
    # A GRU has three gates to an LSTM's four, so strictly fewer recurrent
    # parameters at identical width - the cheapest available proof that the
    # cell argument is actually reaching the constructor.
    assert gru["recurrent_and_head"] < lstm["recurrent_and_head"]


def test_two_casings_of_one_word_both_receive_a_vector(tmp_path):
    """Regression: a single dict keyed by word let whichever casing was
    inserted first claim the key, and the other vocabulary row silently kept
    its random initialisation. Both rows legitimately want the same vector.
    """
    vocabulary = {PAD: 0, UNK: 1, "Ignore": 2, "ignore": 3}
    path = _write_vec(tmp_path, "wiki.en.vec", {"ignore": [7.0, 7.0, 7.0, 7.0]})

    matrix, report = load_fasttext_matrix(vocabulary, [path], dim=4)
    assert matrix[2].tolist() == [7.0, 7.0, 7.0, 7.0]
    assert matrix[3].tolist() == [7.0, 7.0, 7.0, 7.0]
    assert report["covered"] == 2


def test_an_exact_spelling_overwrites_a_lowercase_fallback(tmp_path):
    """The file is streamed once, so a weaker lowercase match can arrive before
    the exact one. The exact spelling must still win.
    """
    vocabulary = {PAD: 0, UNK: 1, "Apple": 2}
    path = _write_vec(tmp_path, "v.vec", {
        "apple": [1.0, 1.0, 1.0, 1.0],   # lowercase fallback, seen first
        "Apple": [2.0, 2.0, 2.0, 2.0],   # exact, seen second
    })
    matrix, _ = load_fasttext_matrix(vocabulary, [path], dim=4)
    assert matrix[2].tolist() == [2.0, 2.0, 2.0, 2.0]
