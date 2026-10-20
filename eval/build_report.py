"""Build the single results workbook from every evaluation on disk.

Four sheets, deliberately:

===============  ==========================================================
Summary          every headline number, plus how to read them
Breakdown        per target, per tier, per layer, per refused intent
Refused queries  every benign query MOCHI turned away, in full
Runs             raw rows from every corpus, one table
===============  ==========================================================

The earlier eleven-sheet version split results by which script produced them,
which is an accident of tooling rather than a fact about the evaluation. What
a reader wants is the headline, then the breakdown behind it, then the
evidence. That is four sheets.

**Attack and benign results are never pooled.** They are different
measurements - one counts attacks that got through, the other counts customers
turned away - and averaging them would be meaningless. They appear on the same
page because they describe one operating point, and a mitigation rate without
a false-positive rate beside it cannot be interpreted: refusing every request
scores 100%.

Reads whatever exists and skips the rest, so it runs before every evaluation
has been done.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Run as a script, so the repo root is not importable by default. Needed since
# this now reads eval.metrics for the enforcement-leakage figures.
sys.path.insert(0, str(REPO))
REPORTS = REPO / "reports"
OUT = REPORTS / "mochi_results.xlsx"

TEAL = "FF0B6E75"
BAND = "FFEAEEEE"

#: ``(label, path, kind)``. ``kind`` picks the reader, because a detection
#: sweep and a two-arm simulation have different columns.
SOURCES = [
    ("generated 100", REPORTS / "simulation_runs.jsonl", "sim"),
    ("images vs llava (visible)", REPORTS / "image_runs.jsonl", "image"),
    ("images vs llava (concealed)", REPORTS / "image_runs_concealed.jsonl", "image"),
    ("gandalf 300", REPORTS / "gandalf_runs.jsonl", "sim"),
    ("gandalf detection (e5-fine-tuned)", REPORTS / "gandalf_detection.jsonl", "detect"),
    ("gandalf detection (e5-banking)", REPORTS / "gandalf_detection_BANKING.jsonl", "detect"),
]
BENIGN = [
    ("e5-fine-tuned", REPORTS / "benign_runs.jsonl"),
    ("e5-banking", REPORTS / "benign_runs_BANKING.jsonl"),
]
INTENTS = {
    "clinc150": REPO / "data" / "external" / "clinc150_intents.json",
    "banking77": REPO / "data" / "external" / "banking77_intents.json",
}

#: Stage ablation arms, in the order they are argued: the cheap layer alone,
#: the learned layer alone, then the cascade. ``sealed`` is the corpus Stage II
#: trained on (held-out rows of it); ``ood`` is Gandalf plus benign traffic
#: that no split contains.
ABLATION = [
    ("Stage I only", "stage1"),
    ("Stage II only", "stage2"),
    ("Stage I + II", "stage12"),
]


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rates(counts: dict) -> dict:
    """Recall, FPR and F1 from a confusion dict, NaN where undefined.

    A corpus with no benign rows has no false-positive rate; reporting 0.0
    there would read as a perfect score rather than an absent measurement.
    """
    tp, fp, tn, fn = (counts.get(k, 0) for k in ("tp", "fp", "tn", "fn"))
    nan = float("nan")
    recall = tp / (tp + fn) if tp + fn else nan
    fpr = fp / (fp + tn) if fp + tn else nan
    precision = tp / (tp + fp) if tp + fp else nan
    f1 = nan
    if precision == precision and recall == recall and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {"recall": recall, "fpr": fpr, "precision": precision, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n": tp + fp + tn + fn}


def BOLD_FONT():
    from openpyxl.styles import Font
    return Font(bold=True)


def blank(value):
    """Excel shows NaN as an error; an absent measurement should be empty."""
    return None if isinstance(value, float) and value != value else value


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalise_image_runs(rows: list[dict]) -> list[dict]:
    """Give image runs the fields the shared summariser expects.

    ``eval/image_simulation.py`` writes its own row shape - there is one
    target, and placement is always the image - so it omits ``target_name``
    and ``placement``. Filling them here keeps one summariser for every
    corpus rather than a second code path that could drift from the first.
    """
    for row in rows:
        row.setdefault("target_name", "LLaVA 7B (vision)")
        row.setdefault("placement", f"image/{row.get('variant', 'visible')}")
    return rows


def intent_name(dataset: str, raw: str) -> str:
    """Map an integer intent label back to its name.

    A sheet reading ``terminate_account`` says what failed; one reading ``55``
    says nothing, and naming the failing category is the entire point of the
    per-intent breakdown.
    """
    path = INTENTS.get(dataset)
    if path is None or not str(raw).isdigit() or not path.exists():
        return str(raw) or "(none)"
    names = json.loads(path.read_text(encoding="utf-8"))
    index = int(raw)
    return names[index] if index < len(names) else str(raw)


def mitigation(rows: list[dict], key) -> list[dict]:
    """ASR and mitigation rate per group.

    ``dangerous`` is counted per (group, target): an attack that works on
    Mistral may be refused by Llama unaided, and crediting MOCHI for the latter
    would inflate the rate. Register item A5 - an attack that fails on a bare
    model was never dangerous.
    """
    seen, undef, defhit = defaultdict(set), defaultdict(set), defaultdict(set)
    counts: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        group = (key(row), row["target_name"])
        seen[group].add(row["attack_id"])
        if row["condition"] == "undefended":
            if row["attack_succeeded"]:
                undef[group].add(row["attack_id"])
        elif row["condition"] == "defended":
            counts[group][row["outcome"]] += 1
            if row.get("stopped_by"):
                counts[group]["by_" + row["stopped_by"]] += 1
            if row["attack_succeeded"]:
                defhit[group].add(row["attack_id"])

    out = []
    for group in sorted(seen, key=lambda g: (str(g[0]), g[1])):
        total = len(seen[group])
        dangerous = undef[group]
        still = dangerous & defhit[group]
        counts_g = counts[group]
        out.append({
            "group": group[0], "target": group[1], "attacks": total,
            "asr_undefended": len(dangerous) / total if total else 0,
            "asr_defended": len(defhit[group]) / total if total else 0,
            "dangerous": len(dangerous), "still": len(still),
            "mitigation": (1 - len(still) / len(dangerous)) if dangerous else None,
            "blocked": counts_g["blocked"], "sanitized": counts_g["sanitized"],
            "refused": counts_g["model_refused"],
            "by_stage1": counts_g["by_stage1"], "by_stage2": counts_g["by_stage2"],
        })
    return out


# --- styling ---------------------------------------------------------------


def head(ws, columns, row=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    ws.append(columns)
    line = ws[row or ws.max_row]
    for cell in line:
        cell.font = Font(bold=True, size=10, color="FFFFFFFF")
        cell.fill = PatternFill("solid", fgColor=TEAL)
        cell.alignment = Alignment(wrap_text=True, vertical="bottom")


def title(ws, text):
    from openpyxl.styles import Font
    ws.append([text])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12)


def autosize(ws, cap=55):
    from openpyxl.utils import get_column_letter
    for i, col in enumerate(ws.columns, start=1):
        widest = max((len(str(c.value)) for c in col if c.value is not None), default=10)
        ws.column_dimensions[get_column_letter(i)].width = min(max(widest + 2, 10), cap)


def pct(ws, col, first=2, fmt="0.0%"):
    for line in ws.iter_rows(min_row=first, min_col=col, max_col=col):
        for cell in line:
            if isinstance(cell.value, (int, float)):
                cell.number_format = fmt


# --- sheets ----------------------------------------------------------------


def sheet_summary(wb, sims, detects, benign, taglish, extra):
    from openpyxl.styles import Alignment, Font
    ws = wb.create_sheet("Summary")
    ws.column_dimensions["A"].width = 34
    for c in "BCDEFG":
        ws.column_dimensions[c].width = 14

    ws.append(["MOCHI — evaluation results"])
    ws.cell(row=1, column=1).font = Font(bold=True, size=15)
    ws.append([])

    for label, rows in sims.items():
        title(ws, f"Attack mitigation — {label}")
        head(ws, ["model", "ASR undefended", "ASR defended", "dangerous",
                  "still succeed", "mitigation"])
        start = ws.max_row + 1
        for r in mitigation(rows, lambda _: label):
            ws.append([r["target"], r["asr_undefended"], r["asr_defended"],
                       r["dangerous"], r["still"], r["mitigation"]])
        for col in (2, 3, 6):
            pct(ws, col, start)
        ws.append([])

    if detects:
        title(ws, "Detection — MOCHI's verdict alone, no model involved")
        head(ws, ["weights", "attacks", "detected", "rate", "by Stage I", "by Stage II"])
        start = ws.max_row + 1
        for label, rows in detects.items():
            got = [r for r in rows if r["outcome"] in ("blocked", "sanitized")]
            s1 = sum(1 for r in rows if r.get("stopped_by") == "stage1")
            s2 = sum(1 for r in rows if r.get("stopped_by") == "stage2")
            ws.append([label, len(rows), len(got), len(got) / len(rows), s1, s2])
        pct(ws, 4, start)
        ws.append([])

    if benign:
        title(ws, "False positives — ordinary customer queries, no attacks")
        head(ws, ["weights", "dataset", "queries", "refused", "FPR"])
        start = ws.max_row + 1
        for label, rows in benign.items():
            per: dict = defaultdict(lambda: [0, 0])
            for r in rows:
                per[r["dataset"]][0] += 1
                per[r["dataset"]][1] += r["false_positive"]
            for ds, (n, fp) in sorted(per.items()):
                ws.append([label, ds, n, fp, fp / n])
        pct(ws, 5, start, "0.00%")
        ws.append([])

    if sims:
        from eval.metrics import ACTED, FORWARDED, enforcement_leakage

        # The raw split comes first. Every derived rate below is a ratio of
        # two of these rows, and a reader who can see the split can check the
        # arithmetic instead of trusting it.
        title(ws, "Where every defended run went")
        head(ws, ["corpus", "verdict", "runs", "reached the model",
                  "attack still worked", "what this row is"])
        start = ws.max_row + 1
        meaning = {
            "BLOCK": ("no", "refused outright - nothing was forwarded"),
            "SANITIZE": ("yes, with the payload cut out",
                         "the denominator for enforcement leakage"),
            "ALLOW": ("yes, unchanged", "MOCHI acted on nothing here"),
        }
        for label, rows in sims.items():
            defended = [r for r in rows if r.get("condition") == "defended"]
            if not defended or all(r.get("mitigation_action") is None
                                   for r in defended):
                continue
            seen = 0
            for verdict in ("BLOCK", "SANITIZE", "ALLOW"):
                group = [r for r in defended
                         if r.get("mitigation_action") == verdict]
                seen += len(group)
                reached, note = meaning[verdict]
                ws.append([label, verdict, len(group), reached,
                           sum(1 for r in group if r.get("attack_succeeded")),
                           note])
            # Rows that carry no verdict at all - a timed-out or errored run.
            # Shown rather than dropped, so the verdict rows add up to the
            # total and nobody has to wonder where a run went.
            if seen < len(defended):
                ws.append([label, "no verdict", len(defended) - seen, "-", 0,
                           "run errored before a verdict was reached"])
            ws.append([label, "TOTAL", len(defended), "",
                       sum(1 for r in defended if r.get("attack_succeeded")),
                       "every attack that beat MOCHI, by either route"])
        for line in ws.iter_rows(min_row=start, max_row=ws.max_row,
                                 min_col=2, max_col=2):
            for cell in line:
                if cell.value == "TOTAL":
                    for col in range(1, 7):
                        ws.cell(row=cell.row, column=col).font = BOLD_FONT()
        ws.append([])

        title(ws, "The four metrics, each with the denominator it is measured over")
        head(ws, ["metric", "corpus", "value", "numerator", "denominator",
                  "what the denominator is"])
        start = ws.max_row + 1
        for label, rows in sims.items():
            defended = [r for r in rows if r.get("condition") == "defended"]
            leak = enforcement_leakage(defended)
            if not leak.acted and not leak.undetected:
                continue  # no verdicts recorded - see the note below
            # An empty denominator is not a zero result. Say so in the row
            # rather than leaving a reader to infer it from two zeroes.
            because = ("attacks MOCHI redacted and forwarded" if leak.forwarded
                       else "nothing was forwarded - every attack here is "
                            "direct, so all were refused outright")
            ws.append(["Enforcement leakage", label,
                       blank(leak.leak_rate), leak.leaked,
                       leak.forwarded or None, because])
            ws.append(["  of which never flagged", label, None,
                       leak.undetected, None,
                       "attacks that worked and were never acted on"])
        pct(ws, 3, start)
        ws.append([])

    sealed = extra.get("sealed", {}).get("stage12")
    if sealed:
        m = sealed["metrics"]
        title(ws, "Detection on the sealed test split — the gateway's own verdict")
        head(ws, ["corpus", "rows", "attacks", "stopped", "recall", "FPR", "F1"])
        start = ws.max_row + 1
        ws.append(["held-out rows of the training corpus", sealed["n_samples"],
                   m["tp"] + m["fn"], m["tp"], m["recall"],
                   m["false_positive_rate"], m["f1"]])
        per = extra.get("per_source_sealed")
        if per:
            no_jaya = [s for s in per["per_source"] if "jayavibhav" not in s]
            agg = rates({k: sum(per["per_source"][s][k] for s in no_jaya)
                         for k in ("tp", "fp", "tn", "fn")})
            ws.append(["excluding jayavibhav", agg["n"], agg["tp"] + agg["fn"],
                       agg["tp"], agg["recall"], agg["fpr"], agg["f1"]])
            tg = [s for s in per["per_source"] if "taglish" in s]
            if tg:
                agg = rates({k: sum(per["per_source"][s][k] for s in tg)
                             for k in ("tp", "fp", "tn", "fn")})
                ws.append(["Taglish rows only", agg["n"], agg["tp"] + agg["fn"],
                           agg["tp"], agg["recall"], agg["fpr"], agg["f1"]])
        for col in (5, 7):
            pct(ws, col, start)
        pct(ws, 6, start, "0.00%")
        ws.append([])

    comparison = extra.get("comparison")
    if comparison:
        title(ws, "Why this model — every candidate on the same sealed rows")
        head(ws, ["model", "track", "family", "F1", "recall", "FPR",
                  "parameters", "recall @ FPR<=1%"])
        start = ws.max_row + 1
        # Sorted by F1 rather than left in run order, so the table argues the
        # choice instead of just listing candidates.
        for row in sorted(comparison,
                          key=lambda r: -(r["results"]["max F1"]["f1"])):
            best = row["results"]["max F1"]
            budget = row["results"].get("FPR<=0.01", {})
            ws.append([row["model"], row["track"], row["family"],
                       best["f1"], best["recall"], best["fpr"],
                       row["parameters"], budget.get("recall")])
        for col in (4, 5, 8):
            pct(ws, col, start)
        pct(ws, 6, start, "0.00%")
        ws.append([])

    contamination = extra.get("taglish_contamination")
    if contamination:
        title(ws, "Taglish held-out set — what the contamination was worth")
        head(ws, ["weights", "subset", "rows", "recall", "FPR", "F1"])
        start = ws.max_row + 1
        for row in contamination:
            ws.append([row["model"], row["subset"], row["n"],
                       blank(row["recall"]), blank(row["fpr"]), blank(row["f1"])])
        for col in (4, 6):
            pct(ws, col, start)
        pct(ws, 5, start, "0.00%")
        ws.append([])

    utility = extra.get("utility")
    if utility:
        title(ws, "Utility — does redaction break the answer the user came for?")
        head(ws, ["document", "payload", "facts kept", "of", "outcome"])
        for row in utility:
            outcome = ("blocked" if row["sanitized_blocked"]
                       else ("lost " + ", ".join(row["lost"]) if row["lost"]
                             else "intact"))
            ws.append([row["document"], row["payload"], row["facts_sanitized"],
                       row["facts_clean"], outcome])
        ws.append([])

    if taglish:
        title(ws, "Taglish ablation — 250 unseen code-switched samples")
        head(ws, ["arm", "recall", "FPR"])
        start = ws.max_row + 1
        for arm, (tp, mal, fp, ben) in taglish.items():
            ws.append([arm, tp / mal, fp / ben])
        pct(ws, 2, start)
        pct(ws, 3, start, "0.00%")
        ws.append([])

    title(ws, "How to read this")
    for line in [
        "The run breakdown above the metrics is there so the rates can be "
        "checked rather than taken on trust. Every verdict row sums to the "
        "total, and each derived rate below is a ratio of two of those rows — "
        "enforcement leakage is the SANITIZE row's failures over its runs, "
        "and the attacks that beat MOCHI are the SANITIZE and ALLOW failures "
        "added together.",
        "Four metrics are reported and they answer different questions. "
        "DETECTION: of all attacks, how many did MOCHI flag? MITIGATION: of "
        "attacks dangerous to the bare model, how many stopped working? "
        "ENFORCEMENT LEAKAGE: of attacks MOCHI acted on but still forwarded, "
        "how many worked anyway? FALSE POSITIVES: of ordinary benign traffic, "
        "how much was refused? Each is quoted with its denominator because "
        "they share no common one.",
        "Enforcement leakage is NOT a false-positive rate. A false positive "
        "is a benign request wrongly refused. Leakage counts malicious "
        "requests that were correctly flagged and whose mitigation then "
        "failed — true positives, not false ones. The two guard different "
        "things and replacing one with the other would leave the mitigation "
        "rate unguarded, since refusing every request scores 100%.",
        "Read leakage next to the row beneath it. On the generated corpus 4 "
        "of 33 redacted requests still succeeded, against 48 attacks never "
        "flagged at all — so redaction mostly holds when it fires, and the "
        "failures are overwhelmingly cases where nothing fired. Effort "
        "belongs in detection rather than in hardening mitigation.",
        "Gandalf has no leakage row because every Gandalf attack sits in the "
        "user's own turn and is refused outright, so nothing is ever "
        "forwarded. An empty denominator is reported as blank rather than 0%, "
        "which would read as a perfect score instead of an absent one.",
        "The image arms have no leakage row either: the harness recorded no "
        "verdict before 2026-10-20. It does now, but those runs predate the "
        "fix and a missing verdict is skipped rather than assumed to be "
        "ALLOW. Re-run the image evaluation to fill this in.",
        "Mitigation is measured against DANGEROUS attacks only — those that "
        "succeeded against the bare model. An attack a model refuses unaided "
        "was never a threat, and counting it would inflate the rate.",
        "ASR and mitigation rate are complements. Quote one in prose, derive "
        "the other.",
        "A mitigation rate means nothing without the false-positive rate "
        "beside it: a gateway that refuses everything scores 100%.",
        "Attack and benign results are never pooled — they are different "
        "measurements with no common metric.",
        "Llama 3.1 refuses most attacks unaided, so its dangerous count is "
        "small and its mitigation rate is noisier than the other two.",
        "MOCHI blocks the same attacks for every model, because enforcement "
        "runs on the prompt before anything is forwarded. Differences between "
        "models come from their own refusals, not from MOCHI.",
        "e5-banking is a retrained checkpoint, not the shipped model. It "
        "trades ~5pp of detection for a large drop in false positives; the "
        "choice is a threat-model decision, not a measurement.",
        "The model comparison is the case for the transformer, and the "
        "false-positive budget column is where it is made. At F1 the classical "
        "models look merely worse; held to a 1% false-positive rate they "
        "collapse — random forest recovers 49.9% of attacks against E5's "
        "98.6%. A gateway has to run at a low false-positive rate, so that "
        "column is the operating point that matters.",
        "The two recurrent baselines are the honest caveat to that. biGRU "
        "reaches F1 0.9703 with a ninth of the parameters and roughly half the "
        "CPU latency. The transformer wins, but not by the margin its size "
        "suggests.",
        "The sealed-split rows are the gateway's verdict on held-out rows of "
        "the corpus Stage II trained on. They are the system-level answer to "
        "the classifier's F1, not a second opinion on unseen data — Gandalf "
        "is that, and it scores 85.3% rather than 99.3%.",
        "The Taglish held-out comparison shows the model is not memorising "
        "that set: recall is identical on the rows e5-fine-tuned trained on "
        "and the rows it did not. It does NOT show the model handles Taglish "
        "as native speakers write it. The set was drawn alongside the "
        "synthetic generated data, so it may sit in the same distribution, "
        "and native-speaker validation remains outstanding.",
        "e5-banking never saw any of the 300 Taglish rows, so its line is the "
        "clean measurement — no subsetting and no memorisation caveat.",
        "In the utility table, 'blocked' is the intended outcome, not a "
        "failure. That document/payload pair would have had 82% of its text "
        "redacted, and the model asked to summarise the remainder invented a "
        "customer, an account number and a balance. Refusing is visible; a "
        "confident fabrication about a banking customer is not.",
    ]:
        ws.append([line])
        cell = ws.cell(row=ws.max_row, column=1)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=ws.max_row, start_column=1,
                       end_row=ws.max_row, end_column=6)
        ws.row_dimensions[ws.max_row].height = 30


def sheet_ablations(wb, extra):
    """Stage ablations and the false-positive/length curve.

    A fifth sheet rather than more rows on Summary, because these answer a
    different question. Summary says how well the gateway works; this says
    which part of it is doing the work, and under what conditions the answer
    changes - which it does, completely, between the two corpora below.
    """
    from openpyxl.styles import Alignment, Font

    sealed = extra.get("sealed", {})
    ood = extra.get("ood", {})
    length = extra.get("length")
    per_source = extra.get("per_source_sealed")
    if not (sealed or ood or length or per_source):
        return

    ws = wb.create_sheet("Ablations")
    ws.column_dimensions["A"].width = 34
    for c in "BCDEFGH":
        ws.column_dimensions[c].width = 13

    ws.append(["Which layer is doing the work?"])
    ws.cell(row=1, column=1).font = Font(bold=True, size=15)
    ws.append([])

    if sealed:
        title(ws, "In distribution — held-out rows of the training corpus")
        head(ws, ["configuration", "TP", "FP", "FN", "recall", "FPR", "F1",
                  "mean ms"])
        start = ws.max_row + 1
        for label, key in ABLATION:
            report = sealed.get(key)
            if not report:
                continue
            m = report["metrics"]
            ws.append([label, m["tp"], m["fp"], m["fn"], m["recall"],
                       m["false_positive_rate"], m["f1"],
                       round(report["latency_ms"]["mean"], 2)])
        for col in (5, 7):
            pct(ws, col, start)
        pct(ws, 6, start, "0.00%")
        ws.append([])

    if ood:
        title(ws, "Out of distribution — Gandalf attacks plus benign traffic")
        head(ws, ["configuration", "TP", "FP", "FN", "recall", "FPR", "F1"])
        start = ws.max_row + 1
        for label, key in ABLATION:
            report = ood.get(key)
            if not report:
                continue
            r = rates(report["total"])
            ws.append([label, r["tp"], r["fp"], r["fn"], blank(r["recall"]),
                       blank(r["fpr"]), blank(r["f1"])])
        for col in (5, 7):
            pct(ws, col, start)
        pct(ws, 6, start, "0.00%")
        ws.append([])

        if ood.get("stage2") and ood.get("stage12"):
            a, b = rates(ood["stage2"]["total"]), rates(ood["stage12"]["total"])
            s_a = rates(sealed["stage2"]["metrics"]) if sealed.get("stage2") else None
            s_b = rates(sealed["stage12"]["metrics"]) if sealed.get("stage12") else None
            title(ws, "What Stage I adds on top of Stage II")
            head(ws, ["corpus", "extra attacks caught", "extra false positives"])
            if s_a and s_b:
                ws.append(["in distribution (sealed split)",
                           s_b["tp"] - s_a["tp"], s_b["fp"] - s_a["fp"]])
            ws.append(["out of distribution (Gandalf)",
                       b["tp"] - a["tp"], b["fp"] - a["fp"]])
            ws.append([])

    if per_source:
        title(ws, "Per source, Stage I + II on the sealed split")
        head(ws, ["source", "rows", "attacks", "recall", "FPR", "F1"])
        start = ws.max_row + 1
        items = sorted(per_source["per_source"].items(),
                       key=lambda kv: -sum(kv[1].values()))
        for name, counts in items:
            r = rates(counts)
            ws.append([name, r["n"], r["tp"] + r["fn"], blank(r["recall"]),
                       blank(r["fpr"]), blank(r["f1"])])
        for col in (4, 6):
            pct(ws, col, start)
        pct(ws, 5, start, "0.00%")
        ws.append([])

    if length:
        rows = length["rows"]
        title(ws, "False positives against document length — benign documents only")
        head(ws, ["characters", "windows", "documents", "flagged", "doc FPR",
                  "per-window FPR", "if independent", "blocked"])
        start = ws.max_row + 1
        for row in rows:
            ws.append([row["target_chars"], row["windows"], row["documents"],
                       row["flagged"], row["fpr"], row["per_window_fpr"],
                       row["predicted_fpr"], row["block_fpr"]])
        for col in (5, 6, 7, 8):
            pct(ws, col, start, "0.00%")
        ws.append([])

    title(ws, "How to read this")
    for line in [
        "The two ablations disagree, and that is the finding. On the corpus "
        "Stage II trained on it sits near its ceiling, so Stage I adds almost "
        "nothing and costs false positives. On Gandalf, which no split "
        "contains, Stage I adds attacks caught at no false-positive cost.",
        "So Stage I is not justified by added recall in distribution. It is "
        "justified by cost (sub-millisecond against ~14 ms), by working when "
        "Stage II is unavailable, and by what it recovers where the learned "
        "layer is weak.",
        "Stage I alone measures 5.8% recall in distribution and 32.0% on "
        "Gandalf. Human 'ignore previous instructions' phrasings are what "
        "regex rules are for.",
        "Gandalf carries no benign rows, so BANKING77 and CLINC150 test are "
        "pooled in to give the arms a false-positive rate. Read recall and "
        "FPR off that sheet, never accuracy — the class balance is 10.4%.",
        "In the length table, window fill matters more than window count. The "
        "two one-window rows are the controlled comparison: 200 characters "
        "flags at 15%, a full 2,048-character window at 0%. Filling a window "
        "with benign context suppresses the score about fiftyfold.",
        "'If independent' is the document rate predicted from the measured "
        "per-window rate. It tracks the observed rate closely at 16 and 32 "
        "windows, so the compounding across windows is arithmetic rather than "
        "mysterious — which is what makes a length-aware threshold derivable.",
        "The sealed split's aggregate FPR misses the 1% target, but that is a "
        "data problem rather than a threshold one: jayavibhav supplies 104 of "
        "167 false positives, and excluding it the rate is 0.75%.",
        "Do not compare the 'mean ms' column between the Stage II and "
        "Stage I + II rows. Those two runs shared a CPU with another "
        "evaluation, so the timings are not measured under equal conditions. "
        "Stage I's sub-millisecond figure ran alone and is sound, and every "
        "confusion-matrix number here is deterministic and unaffected. Re-run "
        "the arms one at a time before quoting latency.",
    ]:
        ws.append([line])
        cell = ws.cell(row=ws.max_row, column=1)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=ws.max_row, start_column=1,
                       end_row=ws.max_row, end_column=8)
        ws.row_dimensions[ws.max_row].height = 30


def sheet_breakdown(wb, sims, benign):
    ws = wb.create_sheet("Breakdown")

    for label, rows in sims.items():
        title(ws, f"{label} — by attack tier")
        head(ws, ["tier", "model", "attacks", "dangerous", "still succeed",
                  "mitigation", "blocked", "sanitized", "model refused"])
        start = ws.max_row + 1
        for r in mitigation(rows, lambda x: x["tier"]):
            ws.append([r["group"], r["target"], r["attacks"], r["dangerous"],
                       r["still"], r["mitigation"], r["blocked"],
                       r["sanitized"], r["refused"]])
        pct(ws, 6, start)
        ws.append([])

        title(ws, f"{label} — which layer stopped it")
        head(ws, ["model", "by Stage I", "by Stage II", "total stopped"])
        for r in mitigation(rows, lambda _: label):
            ws.append([r["target"], r["by_stage1"], r["by_stage2"],
                       r["by_stage1"] + r["by_stage2"]])
        ws.append([])

    for label, rows in benign.items():
        per: dict = defaultdict(lambda: [0, 0])
        for r in rows:
            key = (r["dataset"], intent_name(r["dataset"], r.get("intent", "")))
            per[key][0] += 1
            per[key][1] += r["false_positive"]
        flagged = sorted(((k, v) for k, v in per.items() if v[1]),
                         key=lambda kv: -kv[1][1])
        if not flagged:
            continue
        title(ws, f"Refused benign intents — {label}")
        head(ws, ["dataset", "intent", "queries", "refused", "refusal rate"])
        start = ws.max_row + 1
        for (ds, name), (n, fp) in flagged:
            ws.append([ds, name, n, fp, fp / n])
        pct(ws, 5, start, "0%")
        ws.append([])

    autosize(ws)


def sheet_refused(wb, benign):
    ws = wb.create_sheet("Refused queries")
    head(ws, ["weights", "dataset", "intent", "decision", "stopped by",
              "score", "query"])
    for label, rows in benign.items():
        for r in sorted((x for x in rows if x["false_positive"]),
                        key=lambda x: -(x.get("semantic_score") or 0)):
            ws.append([label, r["dataset"],
                       intent_name(r["dataset"], r.get("intent", "")),
                       r["decision"], r.get("stopped_by"),
                       r.get("semantic_score"), r["text"]])
    autosize(ws, cap=70)


def sheet_runs(wb, sims, detects):
    ws = wb.create_sheet("Runs")
    cols = ["corpus", "attack_id", "tier", "attack_type", "objective",
            "placement", "target_name", "condition", "outcome",
            "attack_succeeded", "reached_model", "stopped_by",
            "mitigation_action", "stage1_outcome", "stage2_outcome",
            "semantic_score", "source_origin", "spans_redacted",
            "latency_ms", "response_text"]
    head(ws, cols)
    for label, rows in list(sims.items()) + list(detects.items()):
        for r in rows:
            ws.append([label] + [r.get(c) for c in cols[1:]])
    autosize(ws, cap=40)


def main() -> int:
    from openpyxl import Workbook

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    sims = {label: read_jsonl(path) for label, path, kind in SOURCES
            if kind == "sim" and read_jsonl(path)}
    sims.update({label: normalise_image_runs(read_jsonl(path))
                 for label, path, kind in SOURCES
                 if kind == "image" and read_jsonl(path)})
    detects = {label: read_jsonl(path) for label, path, kind in SOURCES
               if kind == "detect" and read_jsonl(path)}
    benign = {label: read_jsonl(path) for label, path in BENIGN
              if read_jsonl(path)}
    taglish_path = REPORTS / "taglish_ablation.json"
    taglish = (json.loads(taglish_path.read_text(encoding="utf-8"))
               if taglish_path.exists() else {})

    extra = {
        "sealed": {key: r for _, key in ABLATION
                   if (r := read_json(REPORTS / f"detection_{key}_sealed.json"))},
        "ood": {key: r for _, key in ABLATION
                if (r := read_json(REPORTS / f"ood_{key}.json"))},
        "per_source_sealed": read_json(REPORTS / "detection_stage12_per_source.json"),
        "length": read_json(REPORTS / "long_document_fpr.json"),
        "taglish_contamination": read_json(REPORTS / "taglish_contamination.json"),
        "comparison": read_json(REPORTS / "comparison.json"),
        "utility": read_jsonl(REPORTS / "utility_runs.jsonl"),
    }

    wb = Workbook()
    wb.remove(wb.active)
    sheet_summary(wb, sims, detects, benign, taglish, extra)
    sheet_ablations(wb, extra)
    sheet_breakdown(wb, sims, benign)
    if benign:
        sheet_refused(wb, benign)
    sheet_runs(wb, sims, detects)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)

    total = sum(len(v) for v in list(sims.values()) + list(detects.values())
                + list(benign.values()))
    print(f"\n  {total:,} rows across {len(sims) + len(detects) + len(benign)} "
          f"evaluations -> {args.out}")
    for ws in wb:
        print(f"    {ws.title:<18}{ws.max_row - 1:>6} rows")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
