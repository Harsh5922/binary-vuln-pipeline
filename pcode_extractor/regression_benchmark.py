"""
regression_benchmark.py  --  Phase 1: 9-Binary Regression

Purpose: verify Architecture v2 Final works across all 9 libraries.
NOT the paper benchmark — bug fixes only, no redesign.

Usage:
    python regression_benchmark.py --provider openrouter
    python regression_benchmark.py --provider openrouter --binary PNG001
    python regression_benchmark.py --dry-run   # parse existing reports, no pipeline run

Outputs:
    regression_results.json   -- per-binary + aggregate metrics
    regression_table.csv      -- paper-ready table
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path("D:/BinaryVulnDataset/results")
LABELS_FILE  = Path("D:/BinaryVulnDataset/labels/function_labels.csv")
PIPELINE     = Path(__file__).parent / "pipeline.py"
SCRIPT_DIR   = Path(__file__).parent

# One binary per library
REGRESSION_TARGETS = [
    # (dir_name,  project_label,   budget)
    ("PNG001",  "libpng",       300),
    ("SQL001",  "sqlite3",      300),
    ("XML001",  "libxml2",      300),
    ("TIF001",  "libtiff",      300),
    ("SND001",  "libsndfile",   300),
    ("SSL001",  "openssl",      300),
    ("LUA001",  "lua",          300),
    ("PDF001",  "poppler",      300),
    ("PHP001",  "php",          300),
]

# ── Ground Truth ───────────────────────────────────────────────────────────────

def load_gt() -> dict[str, set[str]]:
    """Returns {bug_id -> set(func_name)} for all label=1 entries."""
    gt: dict[str, set[str]] = {}
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bid = row.get("bug_id", "").strip()
            fn  = row.get("function_name", "").strip()
            lbl = row.get("label", "0").strip()
            if bid and fn and lbl == "1":
                gt.setdefault(bid, set()).add(fn)
    return gt

# ── Per-binary metric collection ───────────────────────────────────────────────

def _ranked_fns(out_dir: Path) -> set[str]:
    ranked = out_dir / "pcode_ranked.jsonl"
    fns: set[str] = set()
    if not ranked.exists():
        return fns
    with open(ranked, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                name = d.get("name") or d.get("func_name") or d.get("function_name")
                if name:
                    fns.add(name)
            except json.JSONDecodeError:
                pass
    return fns

def _parse_report(out_dir: Path) -> dict:
    rpt_path = out_dir / "vulnerability_report.json"
    if not rpt_path.exists():
        return {}
    with open(rpt_path, encoding="utf-8") as f:
        return json.load(f)

def _parse_cost(out_dir: Path) -> dict:
    cost_path = out_dir / "llm_cost_report.json"
    if not cost_path.exists():
        return {}
    with open(cost_path, encoding="utf-8") as f:
        return json.load(f)

def _parse_snapshot(out_dir: Path) -> dict:
    snap_path = out_dir / "snapshot.json"
    if not snap_path.exists():
        return {}
    with open(snap_path, encoding="utf-8") as f:
        return json.load(f)

def collect_metrics(dir_name: str, gt_fns: set[str]) -> dict:
    out_dir = RESULTS_DIR / dir_name / "vuln_O0"

    report   = _parse_report(out_dir)
    cost     = _parse_cost(out_dir)
    snapshot = _parse_snapshot(out_dir)

    # ── Stage 2: ranked function count + recall ────────────────────────────────
    ranked_fns    = _ranked_fns(out_dir)
    ranked_count  = len(ranked_fns)
    stage2_recall = round(len(gt_fns & ranked_fns) / len(gt_fns), 3) if gt_fns else 0.0

    # ── Stage 3: candidate count ───────────────────────────────────────────────
    summary     = report.get("summary", {})
    stage3_cands = summary.get("total_candidates", 0)

    # ── Stage 4: confirmed by Stage 4 only (before fusion) ────────────────────
    # Approximate via reasoning/stage4_analyst calls in cost report
    stage4_calls = 0
    stage45_calls = 0
    stage46_calls = 0
    total_llm    = 0
    for entry in cost.get("summary", []):
        stage = entry.get("stage", "")
        calls = entry.get("calls", 0)
        if stage == "TOTAL":
            total_llm = calls
        elif stage in ("reasoning", "stage4_analyst", "hypothesis_eval"):
            stage4_calls += calls
        elif stage.startswith("orthogonal_"):
            stage45_calls += calls
        elif stage in ("fusion_judge", "consensus_engine", "stage4_6"):
            stage46_calls += calls

    # ── Stage 4.5 evidence count (unique functions with semantic assessments) ──
    # Orthogonal runs 5 analysis types per function → divide by 5
    stage45_evidence = stage45_calls // 5 if stage45_calls else 0

    # ── BEF routing ───────────────────────────────────────────────────────────
    confirmed_vulns = report.get("confirmed_vulnerabilities", [])
    bef_direct  = sum(1 for v in confirmed_vulns
                      if v.get("calibration", {}).get("stage") == "4.6_bef")
    consensus   = sum(1 for v in confirmed_vulns
                      if v.get("calibration", {}).get("stage") == "4.6_fusion_judge")

    # ── TP / FP / FN / TN ─────────────────────────────────────────────────────
    confirmed_fns = set(v["func_name"] for v in confirmed_vulns)
    tp = len(confirmed_fns & gt_fns)
    fp = len(confirmed_fns - gt_fns)
    fn = len(gt_fns - confirmed_fns)
    # TN = ranked non-GT functions not confirmed
    tn = ranked_count - len(gt_fns & ranked_fns) - fp

    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0

    runtime = snapshot.get("runtime_s", 0)

    return {
        "binary":          dir_name,
        "gt_count":        len(gt_fns),
        "tp":              tp,
        "fp":              fp,
        "fn":              fn,
        "tn":              max(0, tn),
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "runtime_s":       round(runtime, 1),
        "stage2_recall":   stage2_recall,
        "ranked_count":    ranked_count,
        "stage3_cands":    stage3_cands,
        "stage4_calls":    stage4_calls,
        "stage45_evidence": stage45_evidence,
        "bef_direct":      bef_direct,
        "consensus":       consensus,
        "total_llm_calls": total_llm,
    }

# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(dir_name: str, budget: int, provider: str) -> float:
    out_dir = RESULTS_DIR / dir_name / "vuln_O0"
    if not out_dir.exists():
        print(f"  [SKIP] {dir_name}: output dir not found — {out_dir}")
        return 0.0

    ranked = out_dir / "pcode_ranked.jsonl"
    if not ranked.exists():
        print(f"  [SKIP] {dir_name}: pcode_ranked.jsonl missing — run Stage 1+2 first")
        return 0.0

    cmd = [
        sys.executable, str(PIPELINE),
        "--resume-from", "3",
        "--output-dir", str(out_dir),
        "--provider", provider,
        "--budget", str(budget),
        "--json",
        "--force",
    ]

    print(f"\n  Running {dir_name} ...")
    t0 = time.perf_counter()
    env = _build_env()
    result = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        env=env,
        capture_output=False,   # let pipeline print to console
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  [WARN] {dir_name}: pipeline exited with code {result.returncode}")
    else:
        print(f"  [DONE] {dir_name}: {elapsed:.0f}s")

    return elapsed

def _build_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Load API key from .env if not already in environment
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists() and "OPENROUTER_API_KEY" not in env:
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                env["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip()
                break
    return env

# ── Output ─────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    hdr = f"{'Binary':<8} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>5} {'P':>6} {'R':>6} {'F1':>6} " \
          f"{'S2Rec':>6} {'S3Cand':>7} {'S4Calls':>8} {'S45Ev':>6} {'BEFDir':>7} {'Cons':>5} {'LLM':>5} {'Runtime':>8}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['binary']:<8} {r['gt_count']:>4} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>5} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
            f"{r['stage2_recall']:>6.3f} {r['stage3_cands']:>7} {r['stage4_calls']:>8} "
            f"{r['stage45_evidence']:>6} {r['bef_direct']:>7} {r['consensus']:>5} "
            f"{r['total_llm_calls']:>5} {r['runtime_s']:>7.0f}s"
        )

    # Aggregate (sum TP/FP/FN across all binaries)
    agg_tp  = sum(r["tp"]  for r in rows)
    agg_fp  = sum(r["fp"]  for r in rows)
    agg_fn  = sum(r["fn"]  for r in rows)
    agg_gt  = sum(r["gt_count"] for r in rows)
    agg_p   = round(agg_tp / (agg_tp + agg_fp), 3) if (agg_tp + agg_fp) > 0 else 0.0
    agg_r   = round(agg_tp / (agg_tp + agg_fn), 3) if (agg_tp + agg_fn) > 0 else 0.0
    agg_f1  = round(2 * agg_p * agg_r / (agg_p + agg_r), 3) if (agg_p + agg_r) > 0 else 0.0
    agg_llm = sum(r["total_llm_calls"] for r in rows)
    agg_rt  = sum(r["runtime_s"] for r in rows)
    agg_bef = sum(r["bef_direct"] for r in rows)
    agg_con = sum(r["consensus"] for r in rows)
    print("-" * len(hdr))
    print(
        f"{'TOTAL':<8} {agg_gt:>4} {agg_tp:>4} {agg_fp:>4} {agg_fn:>4} {'':>5} "
        f"{agg_p:>6.3f} {agg_r:>6.3f} {agg_f1:>6.3f} "
        f"{'':>6} {'':>7} {'':>8} "
        f"{'':>6} {agg_bef:>7} {agg_con:>5} "
        f"{agg_llm:>5} {agg_rt:>7.0f}s"
    )
    print("=" * len(hdr))

    if (agg_bef + agg_con) > 0:
        bef_savings = round(agg_bef / (agg_bef + agg_con) * 100, 1)
        print(f"\nBEF savings: {bef_savings}% ({agg_bef} direct / {agg_con} consensus)")

def save_results(rows: list[dict], out_dir: Path) -> None:
    json_path = out_dir / "regression_results.json"
    csv_path  = out_dir / "regression_table.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nJSON saved -> {json_path}")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  saved -> {csv_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 regression benchmark")
    parser.add_argument("--provider",  default="openrouter",
                        help="LLM provider (default: openrouter)")
    parser.add_argument("--binary",    default=None,
                        help="Run a single binary only (e.g. PNG001)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Parse existing reports only, do not run pipeline")
    args = parser.parse_args()

    gt_all = load_gt()

    targets = REGRESSION_TARGETS
    if args.binary:
        targets = [t for t in targets if t[0] == args.binary]
        if not targets:
            print(f"Unknown binary: {args.binary}")
            sys.exit(1)

    results: list[dict] = []
    for (dir_name, project, budget) in targets:
        gt_fns = gt_all.get(dir_name, set())
        print(f"\n{'='*60}")
        print(f"  {dir_name} ({project})  GT={len(gt_fns)} functions")
        print(f"{'='*60}")

        if not args.dry_run:
            run_pipeline(dir_name, budget, args.provider)

        metrics = collect_metrics(dir_name, gt_fns)
        results.append(metrics)

        # Print intermediate result immediately
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
              f"BEF_direct={metrics['bef_direct']}  Consensus={metrics['consensus']}")

        # Save partial results after each binary (crash safety)
        save_results(results, SCRIPT_DIR / "test_results")

    print_table(results)
    save_results(results, SCRIPT_DIR / "test_results")


if __name__ == "__main__":
    main()
