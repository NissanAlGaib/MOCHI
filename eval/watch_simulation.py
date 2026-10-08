"""Live tally of the attack simulation while it runs.

``reports/simulation.log`` streams one line per run, which is fine for seeing
that something is happening and useless for seeing how it is going. This reads
the results file instead and redraws a running scoreboard.

Safe to run against a job in flight: the harness appends and flushes after
every run, and this only ever reads. Ctrl-C stops the watcher, not the run.

    python -m eval.watch_simulation            # live, refreshes every 2s
    python -m eval.watch_simulation --once     # print once and exit
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO / "reports" / "simulation_runs.jsonl"

#: Fallback when the corpus cannot be found: 100 attacks x 3 targets x 2.
DEFAULT_TOTAL = 600


def expected_total(corpus: Path | None, targets: int = 3) -> int:
    """Total runs for this job.

    Derived from the corpus rather than hardcoded: the generated set is 100
    attacks (600 runs) and the Gandalf sample is 300 (1,800), so a fixed
    constant shows the wrong denominator for whichever one is not running.
    """
    if corpus and corpus.exists():
        with corpus.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        return rows * targets * 2
    return DEFAULT_TOTAL

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
TEAL = "\033[36m"
RESET = "\033[0m"


def read_rows(path: Path) -> list[dict]:
    """Tolerates a half-written final line - the harness may be mid-append."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def render(rows: list[dict], *, started: float, total: int) -> str:
    out: list[str] = []
    done = len(rows)
    pct = done / total if total else 0
    filled = int(38 * min(pct, 1.0))
    elapsed = time.time() - started

    rate = done / elapsed if elapsed > 2 and done else 0
    remaining = (total - done) / rate if rate else 0
    eta = f"{remaining / 60:.0f}m left" if rate and done < total else ""

    out.append(f"{BOLD}MOCHI attack simulation{RESET}")
    out.append(f"  [{'#' * filled}{'.' * (38 - filled)}] "
               f"{done}/{total}  {DIM}{eta}{RESET}")
    out.append("")

    if not rows:
        out.append(f"  {DIM}waiting for the first result...{RESET}")
        return "\n".join(out)

    # --- headline: did the attacks land, with and without the defence ---
    undefended_hits = {r["attack_id"] for r in rows
                       if r["condition"] == "undefended" and r["attack_succeeded"]}
    defended_hits = {r["attack_id"] for r in rows
                     if r["condition"] == "defended" and r["attack_succeeded"]}
    still = undefended_hits & defended_hits

    out.append(f"  {BOLD}Headline{RESET}")
    out.append(f"    attacks that worked on the bare model   "
               f"{GREEN}{len(undefended_hits):>4}{RESET}")
    out.append(f"    of those, still working through MOCHI   "
               f"{RED if still else GREEN}{len(still):>4}{RESET}")
    if undefended_hits:
        rate_pct = 1 - len(still) / len(undefended_hits)
        out.append(f"    mitigation rate so far                 "
                   f"{BOLD}{rate_pct:>7.1%}{RESET}")
    out.append("")

    # --- per target ---
    per: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = (row["target_name"], row["condition"])
        per[key][row["outcome"]] += 1
        per[key]["n"] += 1

    from eval.targets import TARGETS
    for target in TARGETS:
        for condition in ("defended", "undefended"):
            per[(target.name, condition)]  # touch, so pending targets appear

    out.append(f"  {BOLD}By target{RESET}")
    out.append(f"    {'model':<22}{'arm':<12}{'runs':>6}{'success':>9}"
               f"{'blocked':>9}{'sanitzd':>9}{'refused':>9}{'error':>7}")
    for (target, condition) in sorted(per):
        counts = per[(target, condition)]
        hit = counts.get("success", 0)
        out.append(
            f"    {target:<22}{condition:<12}{counts['n']:>6}"
            f"{(RED if hit else DIM)}{hit:>9}{RESET}"
            f"{counts.get('blocked', 0):>9}{counts.get('sanitized', 0):>9}"
            f"{counts.get('model_refused', 0):>9}"
            f"{counts.get('error', 0):>7}"
        )
    out.append("")

    # --- which layer is doing the work ---
    layers: dict = defaultdict(int)
    for row in rows:
        if row["condition"] == "defended" and row.get("stopped_by"):
            layers[row["stopped_by"]] += 1
    if layers:
        parts = "   ".join(f"{name} {count}" for name, count
                           in sorted(layers.items(), key=lambda kv: -kv[1]))
        out.append(f"  {BOLD}Stopped by{RESET}   {TEAL}{parts}{RESET}")
        out.append("")

    # --- per tier, defended only: where the ladder starts to win ---
    tiers: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["condition"] == "defended":
            tiers[row["tier"]]["n"] += 1
            if row["attack_succeeded"]:
                tiers[row["tier"]]["hit"] += 1
    if tiers:
        out.append(f"  {BOLD}Getting through, by tier{RESET} {DIM}(defended){RESET}")
        names = {1: "naive direct", 2: "paraphrase", 3: "obfuscation",
                 4: "structural"}
        for tier in sorted(tiers):
            counts = tiers[tier]
            hit = counts["hit"]
            colour = RED if hit else GREEN
            out.append(f"    tier {tier} {names.get(tier, ''):<16}"
                       f"{colour}{hit:>3}{RESET} / {counts['n']:<4}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RESULTS_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="attack corpus, to size the progress bar")
    parser.add_argument("--total", type=int, default=None,
                        help="override the expected run count")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    started = time.time()
    total = args.total or expected_total(args.corpus)
    if args.once:
        print(render(read_rows(args.runs), started=started, total=total))
        return 0

    try:
        while True:
            rows = read_rows(args.runs)
            print(CLEAR + render(rows, started=started, total=total), flush=True)
            if len(rows) >= total:
                print(f"\n  {GREEN}{BOLD}complete{RESET} - build the workbook:\n"
                      f"    python -m eval.export_simulation\n")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n  {DIM}watcher stopped; the simulation is still running{RESET}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
