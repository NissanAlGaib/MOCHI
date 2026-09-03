"""Word clouds for the corpus, weighted by effect size (register item A10).

    python eval/wordcloud_figures.py --data data/clean
    python eval/wordcloud_figures.py --data data/clean --ngram-max 2

**Sized by |log-odds|, not by frequency.** A frequency-weighted word cloud of any
English corpus is a picture of the stopword list; it says nothing about the
label and belongs in no thesis. Weighting each term by how strongly its presence
shifts the odds of the malicious class makes the figure carry an actual finding.

Three figures are produced:

* ``wordcloud_malicious.png`` / ``wordcloud_benign.png`` - the terms whose
  presence moves the odds toward each class, sized by the strength of that move.
* ``wordcloud_frequency_vs_logodds.png`` - the same corpus weighted both ways,
  side by side. This exists to justify the choice: the frequency panel is
  visibly stopwords, the log-odds panel is visibly attack vocabulary.

The malicious cloud is also the clearest available statement of the corpus's
central problem. Terms like ``pwned`` (2,789 malicious occurrences, 0 benign - a
benchmark success marker), ``kermode``, ``gribbell`` and ``ursus`` (one reused
carrier article) and ``name_1`` (a template placeholder) sit among genuine attack
vocabulary at comparable weight. A classifier can reach a respectable F1 on those
while learning nothing whatsoever about prompt injection. That is an argument for
semantic detection over lexical matching, and it should be made in the open
rather than hidden behind a prettier figure.

Colour is applied per-term by effect size, not at random, so the visual weight
and the statistical weight agree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import DATA_DIR, DatasetError, load_directory  # noqa: E402
from eval.token_association import FDR, MIN_DOC_FREQ, analyse  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

#: Terms per cloud. Beyond roughly 150 the small end is unreadable at any
#: sensible print size, and a thesis figure has to survive being printed.
MAX_TERMS = 150

#: Petrol-teal ramp for benign, warm-red ramp for malicious. Two hues rather
#: than one diverging scale, because the two clouds are separate figures and a
#: reader should not have to remember which end of a scale they are looking at.
MALICIOUS_RAMP = ("#7A1E16", "#A32B21", "#C4442F", "#D9694A", "#E58E70")
BENIGN_RAMP = ("#06484D", "#0B6E75", "#158D91", "#3AA9AB", "#6BC3C2")

#: Terms that are artifacts of corpus construction rather than attack language.
#: Not removed - annotated. Removing them would hide the finding; the figure's
#: whole value is that they are visible.
KNOWN_ARTIFACTS = frozenset({
    "pwned", "pwn", "pwning", "pwnedness", "pawned", "prawned",
    "kermode", "kermodei", "gribbell", "ursus", "americanus",
    "name_1", "few_shot_examples", "sda", "yool",
})


def ramp_colour(ramp, weight: float, peak: float):
    """Pick a ramp stop by relative effect size. Strongest term, darkest stop."""
    if peak <= 0:
        return ramp[len(ramp) // 2]
    index = int((weight / peak) * (len(ramp) - 1))
    return ramp[max(0, min(len(ramp) - 1, len(ramp) - 1 - index))]


def build_cloud(weights: dict[str, float], ramp, *, width: int, height: int):
    """One WordCloud, coloured by effect size rather than at random."""
    from wordcloud import WordCloud

    peak = max(weights.values()) if weights else 1.0

    def colour(word, **_kwargs):
        return ramp_colour(ramp, weights.get(word, 0.0), peak)

    cloud = WordCloud(
        width=width, height=height,
        background_color=None, mode="RGBA",
        prefer_horizontal=0.92,
        relative_scaling=0.55,
        min_font_size=9,
        max_words=MAX_TERMS,
        collocations=False,   # the n-grams are supplied; do not invent more
        margin=4,
    ).generate_from_frequencies(weights)
    return cloud.recolor(color_func=colour)


def save_cloud(weights, ramp, path: Path, title: str, subtitle: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cloud = build_cloud(weights, ramp, width=1600, height=900)

    figure, axes = plt.subplots(figsize=(16, 10.2), dpi=110)
    figure.patch.set_facecolor("white")
    axes.imshow(cloud, interpolation="bilinear")
    axes.axis("off")

    # Title and subtitle are placed in figure coordinates with the axes pushed
    # down to make room. An axes title plus a figure.text at a hand-picked y
    # collide as soon as either string changes length.
    figure.subplots_adjust(top=0.88, bottom=0.01, left=0.01, right=0.99)
    figure.text(0.01, 0.965, title, fontsize=22, fontweight="bold",
                color="#101A1C", va="top")
    figure.text(0.01, 0.912, subtitle, fontsize=12, color="#67777A", va="top")

    figure.savefig(path, facecolor="white")
    plt.close(figure)
    print(f"    wrote {path}")


def save_comparison(frequency: dict, logodds: dict, path: Path) -> None:
    """The panel that justifies the weighting choice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left = build_cloud(frequency, BENIGN_RAMP, width=1100, height=760)
    right = build_cloud(logodds, MALICIOUS_RAMP, width=1100, height=760)

    figure, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(17, 7.4), dpi=110)
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(top=0.80, bottom=0.08, left=0.015, right=0.985, wspace=0.04)

    for axes, cloud, title, note in (
        (ax_left, left, "Weighted by raw frequency",
         "What a default word cloud shows: the stopword list."),
        (ax_right, right, "Weighted by |log-odds|",
         "What relates to the label: attack vocabulary, and the artifacts beside it."),
    ):
        axes.imshow(cloud, interpolation="bilinear")
        axes.axis("off")
        axes.set_title(title, fontsize=16, fontweight="bold", loc="left",
                       pad=10, color="#101A1C")
        axes.text(0, -0.035, note, transform=axes.transAxes, fontsize=10.5,
                  color="#67777A", va="top")

    figure.text(0.015, 0.965, "Why the weighting matters", fontsize=19,
                fontweight="bold", color="#101A1C", va="top")
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    print(f"    wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-freq", type=int, default=MIN_DOC_FREQ)
    parser.add_argument("--ngram-max", type=int, default=1,
                        help="1 for unigrams; 2-3 to let phrases compete")
    args = parser.parse_args()

    try:
        from wordcloud import WordCloud  # noqa: F401
    except ImportError:
        print("ERROR: pip install wordcloud matplotlib")
        return 1

    try:
        samples = load_directory(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\n  Analysing {len(samples):,} samples (n-gram max {args.ngram_max}) ...")
    results, meta = analyse(samples, min_freq=args.min_freq,
                            ngram_max=args.ngram_max)

    significant = [r for r in results if r.q_value < FDR]
    malicious = {r.token: r.log_odds for r in significant if r.log_odds > 0}
    benign = {r.token: -r.log_odds for r in significant if r.log_odds < 0}
    frequency = {r.token: float(r.n_malicious + r.n_benign) for r in results}

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"  {len(significant):,} significant terms at FDR {FDR}\n")

    save_cloud(
        malicious, MALICIOUS_RAMP, args.out / "wordcloud_malicious.png",
        "Terms associated with prompt injection",
        f"Sized by |log-odds| · {meta['n_malicious']:,} malicious vs "
        f"{meta['n_benign']:,} benign · significant at FDR {FDR} after "
        f"Benjamini–Hochberg",
    )
    save_cloud(
        benign, BENIGN_RAMP, args.out / "wordcloud_benign.png",
        "Terms associated with benign prompts",
        f"Sized by |log-odds| · same corpus, same test, opposite direction",
    )
    save_comparison(frequency, malicious,
                    args.out / "wordcloud_frequency_vs_logodds.png")

    # Name the artifacts explicitly. A reader looking at the figure should not
    # have to work out on their own which of these terms is real.
    present = [t for t in sorted(malicious, key=malicious.get, reverse=True)[:MAX_TERMS]
               if t in KNOWN_ARTIFACTS]
    if present:
        print()
        print("  Corpus artifacts visible in the malicious cloud "
              f"({len(present)} of the top {MAX_TERMS}):")
        print(f"    {', '.join(present)}")
        print("  These are benchmark markers, one reused carrier article, and a")
        print("  template placeholder - not attack vocabulary. Their presence at")
        print("  comparable weight to real attack terms is the finding.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
