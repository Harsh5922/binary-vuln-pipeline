"""
test_integration.py — Full Pipeline Integration Test
=====================================================
Runs the complete Stage 3 -> Stage 4 -> Stage 4.5 -> BEF -> [Consensus] -> Finding
chain on real functions from actual pcode.jsonl files.

Target functions (named + auto-selected from GT):
  png_handle_eXIf, sqlite3Select, xmlStringLenDecodeEntities, multiSelect, ...
  + auto-fill to --budget (default 30) from GT labels across all libraries

Per-function trace output:
  S3:   <n_cands> candidates | fwd=<n> bwd=<n> sem=<n> prior=Y/N
  S4:   <vuln_type> | <support> | eq=<eq> mu=<mu> | confirmed=YES/NO
  S4.5: <type>:<bug>/<conf> [x5 chain]
  BEF:  mem=<m> sem=<s> gap=<g> post=<p> -> <routing>
  FIND: <CONFIRMED/REJECTED>  sev=<sev>  conf=<c>  stage=<stage>

Flags:
  --budget N          Total functions to test (default 30)
  --library LIB       Restrict auto-fill to one library
  --binary BIN        Force a specific binary for named targets
  --skip-stage4       Skip LLM Stage 4 + 4.5 (fast structural-only mode)
  --skip-stage45      Skip Stage 4.5 only
  --no-consensus      Skip ConsensusEngine even for conflicts
  --named-only        Only run the hardcoded named targets (no auto-fill)
  --save              Save full trace to test_results/integration_trace.json

Usage:
    python test_integration.py --budget 30 --save
    python test_integration.py --skip-stage4 --budget 50   # fast, no LLM
    python test_integration.py --binary PNG001 --named-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR      = Path(__file__).parent / "test_results"
OUT_DIR.mkdir(exist_ok=True)

# Auto-load .env if OPENROUTER_API_KEY is not already in environment
_ENV_FILE = Path(__file__).parent / ".env"
if not os.environ.get("OPENROUTER_API_KEY") and _ENV_FILE.exists():
    import os as _os
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        if _line.startswith("OPENROUTER_API_KEY="):
            _os.environ["OPENROUTER_API_KEY"] = _line.split("=", 1)[1].strip()
            break

DATASET_ROOT = Path("D:/BinaryVulnDataset")
LABELS_CSV   = DATASET_ROOT / "labels" / "function_labels.csv"
RESULTS_ROOT = DATASET_ROOT / "results"

LIBRARY_PREFIX = {
    "libpng": "PNG", "sqlite3": "SQL", "libtiff": "TIF",
    "libxml2": "XML", "libsndfile": "SND",
}

# ─── Named targets (user-specified + known CVE functions) ─────────────────────
# (func_name, library_prefix, expected_category)
# ─── Stratified targets (30 hand-curated, 6 per library) ────────────────────
# (func_name, prefix, expected_cat, func_category, fallbacks)
# expected_cat  : "TP" = in GT labels  |  "SAFE" = non-GT, expected benign
# func_category : arithmetic | logic | parser | allocator | simple
STRATIFIED_TARGETS: list[tuple[str, str, str, str, list]] = [
    # ── libpng (6) ─────────────────────────────────────────────────────────
    ("png_handle_eXIf",             "PNG", "TP",   "allocator",  []),  # PNG006 heap OOB
    ("png_check_IHDR",              "PNG", "TP",   "parser",     []),  # PNG005 chunk validation
    ("png_check_chunk_length",      "PNG", "TP",   "arithmetic", []),  # PNG001 int overflow
    ("png_handle_PLTE",             "PNG", "TP",   "parser",     []),  # PNG003 palette
    ("png_get_rowbytes",            "PNG", "SAFE", "simple",     []),  # non-GT
    ("png_get_valid",               "PNG", "SAFE", "simple",     ["png_get_bit_depth", "png_get_image_width"]),  # non-GT clean getter

    # ── sqlite3 (6) ─────────────────────────────────────────────────────────
    ("sqlite3Select",               "SQL", "TP",   "logic",      []),  # SQL006 complex query
    ("multiSelect",                 "SQL", "TP",   "logic",      []),  # SQL014
    ("sqlite3_str_vappendf",        "SQL", "TP",   "arithmetic", []),  # SQL019 format string
    ("flattenSubquery",             "SQL", "TP",   "logic",      []),  # SQL003
    ("selectExpander",              "SQL", "TP",   "logic",      []),  # SQL002
    ("sqlite3_errmsg",              "SQL", "SAFE", "simple",     []),  # non-GT

    # ── libxml2 (6) ─────────────────────────────────────────────────────────
    ("xmlStringLenDecodeEntities",  "XML", "TP",   "arithmetic", []),  # XML010
    ("htmlParseTryOrFinish",        "XML", "TP",   "parser",     []),  # XML007
    ("xmlSnprintfElementContent",   "XML", "TP",   "arithmetic", []),  # XML001/006
    ("xmlStrncat",                  "XML", "TP",   "allocator",  []),  # XML011
    ("xmlParseComment",             "XML", "TP",   "parser",     []),  # XML008
    ("xmlStrlen",                   "XML", "SAFE", "simple",     []),  # non-GT

    # ── libtiff (6) ─────────────────────────────────────────────────────────
    ("LZWDecodeCompat",             "TIF", "TP",   "arithmetic", []),  # TIF010
    ("PixarLogDecode",              "TIF", "TP",   "arithmetic", []),  # TIF002
    ("allocChoppedUpStripArrays",   "TIF", "TP",   "allocator",  []),  # TIF007
    ("TIFFPrintDirectory",          "TIF", "TP",   "parser",     []),  # TIF011
    ("ChopUpSingleUncompressedStrip","TIF","TP",   "logic",      []),  # TIF014
    ("_TIFFMultiply64",             "TIF", "SAFE", "arithmetic", ["_TIFFMultiply32"]),  # non-GT

    # ── libsndfile (6) ──────────────────────────────────────────────────────
    ("wavlike_read_fmt_chunk",      "SND", "TP",   "parser",     []),  # SND017
    ("psf_binheader_writef",        "SND", "TP",   "arithmetic", []),  # SND010
    ("paf_read_header",             "SND", "TP",   "parser",     []),  # SND001
    ("aiff_read_chanmap",           "SND", "TP",   "parser",     []),  # SND005
    ("sf_strerror",                 "SND", "SAFE", "simple",     []),  # non-GT clean getter
    ("sf_get_string",               "SND", "SAFE", "simple",     ["sf_format_check"]),  # non-GT clean getter
]

# Legacy named-only list (kept for --named-only backward compat)
NAMED_TARGETS: list[tuple[str, str, str]] = [
    (n, p, c) for (n, p, c, _, _) in STRATIFIED_TARGETS
]

DANGEROUS_SINKS = {"memcpy","strcpy","strcat","sprintf","gets","malloc","realloc","memmove","strncpy"}


# ─── Ground truth loading ─────────────────────────────────────────────────────

def load_gt() -> dict[str, set[str]]:
    gt: dict[str, set[str]] = defaultdict(set)
    if not LABELS_CSV.exists():
        return gt
    with LABELS_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gt[row["bug_id"].strip()].add(row["function_name"].strip())
    return gt


# ─── pcode loading ────────────────────────────────────────────────────────────

def find_pcode(binary_id: str) -> Optional[Path]:
    base = RESULTS_ROOT / binary_id
    for sub in [base / "vuln_O0", base / "vuln_O2", base]:
        p = sub / "pcode.jsonl"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def load_funcs(pcode_path: Path) -> list[dict]:
    funcs = []
    with pcode_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    funcs.append(json.loads(line))
                except Exception:
                    pass
    return funcs


def search_func(
    name: str,
    prefix: str,
    fallbacks: list[str] | None = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Find function by name across all binaries with given prefix."""
    if not RESULTS_ROOT.exists():
        return None, None
    search = [name] + (fallbacks or [])
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not (d.is_dir() and d.name.startswith(prefix)):
            continue
        pcode = find_pcode(d.name)
        if pcode is None:
            continue
        for fn in load_funcs(pcode):
            if fn["name"] in search:
                return fn, d.name
    return None, None


# ─── Function selection ───────────────────────────────────────────────────────

@dataclass
class FuncEntry:
    func:        dict
    binary_id:   str
    name:        str
    expected_cat: str    # TP / SAFE / UNKNOWN
    is_gt:       bool
    source:      str     # "named" | "gt_auto" | "fp_auto" | "stratified"
    category:    str = "unknown"  # arithmetic | logic | parser | allocator | simple


def collect_functions(
    budget:       int,
    library_filter: Optional[str],
    binary_override: Optional[str],
    named_only:   bool,
    gt:           dict[str, set[str]],
) -> list[FuncEntry]:
    entries: list[FuncEntry] = []
    seen_names: set[str] = set()

    # GT lookup: name -> set of binary_ids containing it
    all_gt_names: set[str] = {fn for fns in gt.values() for fn in fns}
    gt_by_name: dict[str, str] = {}
    for bid, fns in gt.items():
        for fn in fns:
            gt_by_name[fn] = bid

    # 1. Named targets
    for (tgt_name, tgt_prefix, tgt_cat) in NAMED_TARGETS:
        if len(entries) >= budget:
            break
        if library_filter and not tgt_prefix.startswith(
            LIBRARY_PREFIX.get(library_filter, library_filter.upper())
        ):
            continue
        if binary_override:
            pcode = find_pcode(binary_override)
            if pcode:
                by_name = {f["name"]: f for f in load_funcs(pcode)}
                fn = by_name.get(tgt_name)
                if fn:
                    entries.append(FuncEntry(
                        func=fn, binary_id=binary_override, name=tgt_name,
                        expected_cat=tgt_cat, is_gt=tgt_name in all_gt_names,
                        source="named",
                    ))
                    seen_names.add(tgt_name)
        else:
            fn, bid = search_func(tgt_name, tgt_prefix)
            if fn and tgt_name not in seen_names:
                entries.append(FuncEntry(
                    func=fn, binary_id=bid, name=tgt_name,
                    expected_cat=tgt_cat, is_gt=tgt_name in all_gt_names,
                    source="named",
                ))
                seen_names.add(tgt_name)

    if named_only or len(entries) >= budget:
        return entries[:budget]

    # 2. Auto-fill from GT labels
    prefixes = (
        [LIBRARY_PREFIX[library_filter]] if library_filter and library_filter in LIBRARY_PREFIX
        else list(LIBRARY_PREFIX.values())
    )
    for bid, gt_fns in sorted(gt.items()):
        if len(entries) >= budget:
            break
        if not any(bid.startswith(p) for p in prefixes):
            continue
        pcode = find_pcode(bid)
        if pcode is None:
            continue
        by_name = {f["name"]: f for f in load_funcs(pcode)}
        for fn_name in sorted(gt_fns):
            if len(entries) >= budget:
                break
            if fn_name in seen_names or fn_name not in by_name:
                continue
            entries.append(FuncEntry(
                func=by_name[fn_name], binary_id=bid, name=fn_name,
                expected_cat="TP", is_gt=True, source="gt_auto",
            ))
            seen_names.add(fn_name)

    # 3. Auto-fill non-GT dangerous-call functions to round out
    if len(entries) < budget:
        for bid in sorted(RESULTS_ROOT.iterdir(),
                          key=lambda p: p.name) if RESULTS_ROOT.exists() else []:
            if not bid.is_dir():
                continue
            if not any(bid.name.startswith(p) for p in prefixes):
                continue
            pcode = find_pcode(bid.name)
            if pcode is None:
                continue
            for fn in load_funcs(pcode):
                if len(entries) >= budget:
                    break
                if fn["name"] in seen_names or fn["name"] in all_gt_names:
                    continue
                # Has at least one dangerous call → FP candidate
                has_dc = any(
                    op.get("mnem") in ("CALL","CALLIND") and
                    any(s in str(op.get("inputs",[""])[0]).lower() for s in DANGEROUS_SINKS)
                    for op in fn.get("ops", [])
                )
                if has_dc:
                    entries.append(FuncEntry(
                        func=fn, binary_id=bid.name, name=fn["name"],
                        expected_cat="FP", is_gt=False, source="fp_auto",
                    ))
                    seen_names.add(fn["name"])
            if len(entries) >= budget:
                break

    return entries[:budget]


def collect_stratified(gt: dict[str, set[str]]) -> list[FuncEntry]:
    """Collect the 30 hand-curated stratified targets in library/category order."""
    all_gt_names = {fn for fns in gt.values() for fn in fns}
    entries: list[FuncEntry] = []
    for (name, prefix, exp_cat, cat, fallbacks) in STRATIFIED_TARGETS:
        fn, bid = search_func(name, prefix, fallbacks)
        if fn is None:
            print(f"  WARNING: {prefix} {name} not found in pcode files — skipping")
            continue
        entries.append(FuncEntry(
            func=fn, binary_id=bid, name=fn["name"],
            expected_cat=exp_cat, is_gt=(name in all_gt_names),
            source="stratified", category=cat,
        ))
    return entries


# ─── Pipeline stages ─────────────────────────────────────────────────────────

def build_stage3(store_path: str = "pattern_store.db"):
    from pattern_store       import PatternStore
    from pattern_matcher     import PatternMatcher
    from stage3_orchestrator import Stage3Orchestrator
    store   = PatternStore(store_path)
    matcher = PatternMatcher(store)
    return Stage3Orchestrator(matcher), store


def build_stage4(provider="openrouter"):
    from stage4_analyst import HypothesisEvaluationEngine
    try:
        engine = HypothesisEvaluationEngine(provider=provider, llm_mode="require")
    except EnvironmentError:
        # No API key — return disabled engine so caller can check .llm_enabled
        engine = HypothesisEvaluationEngine(provider=provider, llm_mode="skip")
    return engine


def build_stage45():
    from stage4_orthogonal import OrthogonalSemanticAnalyzer
    return OrthogonalSemanticAnalyzer()


def build_bef_and_engine(provider="openrouter", use_consensus=True):
    from stage4_fusion import BayesianEvidenceFusion, CandidateFusion
    bef    = BayesianEvidenceFusion()
    fusion = CandidateFusion(provider=provider, use_consensus=use_consensus)
    return bef, fusion


# ─── Per-function trace ───────────────────────────────────────────────────────

def summarize_stage3(taint_result) -> dict:
    n_vulns = len(taint_result.vulns)
    evs     = taint_result.evidences or {}
    fwd = bwd = sem = prior = 0
    for ev in evs.values():
        # EvidenceVector fields (stage3_evidence.py)
        if getattr(ev, "forward_reached", False):
            fwd  += max(1, len(getattr(ev, "taint_path", []) or []))
        if getattr(ev, "backward_reached", False):
            bwd  += 1
        sem   += len(getattr(ev, "semantic_items", []) or [])
        # Behavioral prior is present when similarity > 0 or TP rate differs from default
        if getattr(ev, "behavioral_similarity", 0.0) > 0.0:
            prior += 1
    return {
        "n_candidates": n_vulns,
        "n_evidence":   len(evs),
        "fwd":  fwd, "bwd": bwd, "sem": sem, "prior": prior,
    }


def summarize_stage4(assessments: list) -> dict:
    if not assessments:
        return {"n_assessments": 0}
    # Use the highest-support assessment
    order = {"Strong": 3, "Moderate": 2, "Weak": 1, "Unsupported": 0}
    best  = max(assessments, key=lambda a: order.get(getattr(a,"hypothesis_support",""), 0))
    unc   = getattr(best, "uncertainty", None)
    ea    = getattr(best, "exploitability_assessment", None)
    agr   = getattr(best, "agreement", None)
    return {
        "n_assessments":        len(assessments),
        "vuln_type":            getattr(best, "vuln_type", "?"),
        "hypothesis_support":   getattr(best, "hypothesis_support", "?"),
        "confirmed":            getattr(best, "confirmed", False),
        "evidence_quality":     getattr(unc, "evidence_quality",  "?") if unc else "?",
        "model_uncertainty":    getattr(unc, "model_uncertainty", "?") if unc else "?",
        "overall_uncertainty":  getattr(unc, "overall",           "?") if unc else "?",
        "reachability":         getattr(ea,  "reachability",      "?") if ea  else "?",
        "exploitability":       getattr(ea,  "exploitability",    "?") if ea  else "?",
        "impact":               getattr(ea,  "impact",            "?") if ea  else "?",
        "agreement_level":      getattr(agr, "level",             "?") if agr else "?",
        "n_active_signals":     getattr(agr, "n_active",          0)   if agr else 0,
        "n_hypotheses":         len(getattr(best, "hypothesis_ranking", []) or []),
        "n_contradictions":     len(getattr(best, "contradictory_evidence", []) or []),
    }


def summarize_stage45(sem_assessments: list) -> dict:
    if not sem_assessments:
        return {"n": 0, "analyses": []}
    analyses = []
    for sa in sem_assessments:
        at   = str(getattr(sa, "analysis_type", "?"))
        bug  = getattr(sa, "potential_bug", "")
        conf = getattr(sa, "confidence", 0.0)
        has_ref = bool(getattr(sa, "prior_analysis_reference", ""))
        analyses.append({
            "type": at, "bug": bug[:60], "confidence": round(conf, 3),
            "has_prior_ref": has_ref,
        })
    confirmed = sum(1 for sa in sem_assessments if getattr(sa, "confirmed", False))
    return {"n": len(sem_assessments), "confirmed": confirmed, "analyses": analyses}


def summarize_bef(signal) -> dict:
    return {
        "memory_score":  round(signal.memory_score,  4),
        "semantic_score": round(signal.semantic_score, 4),
        "gap":           round(abs(signal.memory_score - signal.semantic_score), 4),
        "posterior":     round(signal.posterior,     4),
        "routing":       signal.routing,
    }


def summarize_finding(finding) -> dict:
    if finding is None:
        return {"confirmed": False}
    cal = finding.calibration or {}
    return {
        "confirmed": finding.confirmed,
        "severity":  finding.severity,
        "confidence": round(finding.confidence, 4),
        "model_used": finding.model_used,
        "stage":     cal.get("stage", "?"),
        "routing":   cal.get("routing", "?"),
    }


# ─── Per-function runner ──────────────────────────────────────────────────────

def run_function(
    entry:       FuncEntry,
    orch:        object,
    s4:          object,
    s45:         object,
    bef:         object,
    fusion:      object,
    skip_stage4: bool,
    skip_stage45: bool,
) -> dict:
    trace: dict = {
        "name":         entry.name,
        "binary_id":    entry.binary_id,
        "expected_cat": entry.expected_cat,
        "is_gt":        entry.is_gt,
        "source":       entry.source,
        "category":     entry.category,
    }
    fn = entry.func

    # Stage 3
    t0 = time.perf_counter()
    taint_result = orch.analyze(fn)
    trace["stage3"] = summarize_stage3(taint_result)
    trace["stage3"]["elapsed_s"] = round(time.perf_counter() - t0, 2)

    # Stage 4
    assessments: list = []
    if not skip_stage4 and taint_result.vulns and s4 and s4.llm_enabled:
        t0 = time.perf_counter()
        assessments = s4.evaluate_all([taint_result], {entry.name: fn}, delay_s=1.0)
        trace["stage4"] = summarize_stage4(assessments)
        trace["stage4"]["elapsed_s"] = round(time.perf_counter() - t0, 2)
    elif taint_result.vulns and (skip_stage4 or not (s4 and s4.llm_enabled)):
        trace["stage4"] = {"n_assessments": 0, "skipped": True}
    else:
        trace["stage4"] = {"n_assessments": 0, "no_candidates": True}

    # Stage 4.5
    sem_assessments: list = []
    if not skip_stage4 and not skip_stage45 and s45 and s45.enabled:
        t0 = time.perf_counter()
        sem_assessments = s45.analyze_function(fn, chained=True)
        trace["stage45"] = summarize_stage45(sem_assessments)
        trace["stage45"]["elapsed_s"] = round(time.perf_counter() - t0, 2)
    else:
        trace["stage45"] = {"n": 0, "skipped": True}

    # BEF + Finding
    if assessments:
        best_assessment = max(
            assessments,
            key=lambda a: {"Strong":3,"Moderate":2,"Weak":1,"Unsupported":0}.get(
                getattr(a,"hypothesis_support",""), 0
            )
        )
        t0 = time.perf_counter()
        signal = bef.fuse(best_assessment, sem_assessments)
        trace["bef"] = summarize_bef(signal)
        trace["bef"]["elapsed_s"] = round(time.perf_counter() - t0, 2)

        if signal.routing == "needs_consensus" and fusion._engine and fusion._engine.enabled:
            t0 = time.perf_counter()
            verdict = fusion._engine.resolve(entry.name, best_assessment, sem_assessments, fn)
            finding = verdict.to_finding()
            finding.confidence = signal.posterior  # BEF posterior > uncertainty.confidence * 1.10
            trace["consensus"] = {
                "conflict_type":       verdict.conflict_type,
                "conflict_resolution": verdict.conflict_resolution[:120],
                "elapsed_s":           round(time.perf_counter() - t0, 2),
            }
        else:
            finding = fusion._bef_verdict(best_assessment, sem_assessments, signal, entry.name)
            trace["consensus"] = {"skipped": True, "reason": signal.routing}

        trace["finding"] = summarize_finding(finding)
    else:
        # No Stage 4 assessment — Stage 4.5 produces semantic evidence only, not findings.
        # A Stage 4 anchor is required before BEF can emit a finding.
        trace["bef"]      = {"skipped": True, "reason": "no_stage4_assessment"}
        trace["consensus"] = {"skipped": True, "reason": "no_stage4_assessment"}
        trace["finding"]  = {"confirmed": False, "reason": "no_stage4_assessment"}

    return trace


# ─── Console output ───────────────────────────────────────────────────────────

def print_trace(i: int, total: int, trace: dict) -> None:
    name  = trace["name"]
    bid   = trace["binary_id"]
    gt    = "[GT]" if trace["is_gt"] else "    "
    cat   = trace["expected_cat"]
    src   = trace["source"]

    print(f"\n[{i:02d}/{total}] {name}  ({bid}) {gt} expected={cat} src={src}")

    # Stage 3
    s3 = trace.get("stage3", {})
    print(f"  S3:   {s3.get('n_candidates',0)} cands | "
          f"fwd={s3.get('fwd',0)} bwd={s3.get('bwd',0)} "
          f"sem={s3.get('sem',0)} prior={'Y' if s3.get('prior',0) else 'N'} | "
          f"{s3.get('elapsed_s',0):.1f}s")

    # Stage 4
    s4 = trace.get("stage4", {})
    if s4.get("skipped") or s4.get("no_candidates"):
        reason = "no candidates" if s4.get("no_candidates") else "skipped"
        print(f"  S4:   ({reason})")
    elif s4.get("n_assessments", 0) == 0:
        print(f"  S4:   no assessments")
    else:
        conf = "YES" if s4.get("confirmed") else "NO "
        print(f"  S4:   {s4.get('vuln_type','?')} | {s4.get('hypothesis_support','?')} | "
              f"eq={s4.get('evidence_quality','?')} mu={s4.get('model_uncertainty','?')} | "
              f"confirmed={conf} | "
              f"agree={s4.get('agreement_level','?')} ({s4.get('n_active_signals',0)} sigs) | "
              f"{s4.get('elapsed_s',0):.1f}s")

    # Stage 4.5
    s45 = trace.get("stage45", {})
    if s45.get("skipped"):
        print(f"  S4.5: (skipped)")
    elif s45.get("n", 0) == 0:
        print(f"  S4.5: no analyses")
    else:
        analyses = s45.get("analyses", [])
        short = []
        for a in analyses:
            bug_short = a["bug"].split(":")[0][:20] if a["bug"] else "no_issue"
            ref = "*" if a["has_prior_ref"] else ""
            short.append(f"{a['type'].split('.')[-1][:5]}:{bug_short}/{a['confidence']:.2f}{ref}")
        print(f"  S4.5: [{']  ['.join(short)}] | {s45.get('confirmed',0)} confirmed | {s45.get('elapsed_s',0):.1f}s")

    # BEF
    bef = trace.get("bef", {})
    if bef.get("skipped"):
        print(f"  BEF:  ({bef.get('reason','skipped')})")
    else:
        print(f"  BEF:  mem={bef['memory_score']:.3f} sem={bef['semantic_score']:.3f} "
              f"gap={bef['gap']:.3f} post={bef['posterior']:.3f} -> {bef['routing']}")

    # Consensus
    cons = trace.get("consensus", {})
    if cons and not cons.get("skipped"):
        print(f"  CONS: {cons.get('conflict_type','?')} | {cons.get('conflict_resolution','')[:60]} | {cons.get('elapsed_s',0):.1f}s")

    # Finding
    fnd = trace.get("finding", {})
    status = "CONFIRMED" if fnd.get("confirmed") else "REJECTED "
    if fnd.get("reason"):
        print(f"  FIND: {status}  ({fnd['reason']})")
    else:
        print(f"  FIND: {status}  sev={fnd.get('severity','?')}  "
              f"conf={fnd.get('confidence',0):.3f}  stage={fnd.get('stage','?')}")


def print_summary(traces: list[dict], elapsed: float) -> None:
    total      = len(traces)
    confirmed  = sum(1 for t in traces if t.get("finding", {}).get("confirmed"))
    gt_total   = sum(1 for t in traces if t["is_gt"])
    gt_conf    = sum(1 for t in traces if t["is_gt"] and t.get("finding", {}).get("confirmed"))
    bef_direct = sum(1 for t in traces if t.get("bef", {}).get("routing","").startswith("bayes_direct"))
    consensus  = sum(1 for t in traces if "consensus" in t and not t["consensus"].get("skipped"))
    no_s4      = sum(1 for t in traces if not t.get("stage4", {}).get("n_assessments"))

    print(f"\n{'='*65}")
    print(f"  INTEGRATION TEST SUMMARY  ({total} functions, {elapsed:.1f}s)")
    print(f"{'='*65}")
    print(f"  Confirmed findings:   {confirmed}/{total}  ({100*confirmed//total if total else 0}%)")
    print(f"  GT functions:         {gt_conf}/{gt_total} confirmed  ({100*gt_conf//gt_total if gt_total else 0}% recall)")
    print(f"  BEF direct:           {bef_direct}/{total-no_s4} (no LLM needed)")
    print(f"  Consensus Engine:     {consensus} functions routed to LLM")
    print(f"  No Stage 4 assess.:   {no_s4} (no vulns or skipped)")

    # Per-category breakdown
    by_cat: dict[str, list[bool]] = {}
    for t in traces:
        cat = t["expected_cat"]
        confirmed_t = t.get("finding", {}).get("confirmed", False)
        by_cat.setdefault(cat, []).append(confirmed_t)
    print(f"\n  Per-category confirmed:")
    for cat, results in sorted(by_cat.items()):
        n = sum(results)
        print(f"    {cat:<8} {n}/{len(results)} confirmed")

    # BEF routing distribution
    routing_counts: dict[str, int] = {}
    for t in traces:
        r = t.get("bef", {}).get("routing", "n/a")
        routing_counts[r] = routing_counts.get(r, 0) + 1
    print(f"\n  BEF routing distribution:")
    for r, n in sorted(routing_counts.items()):
        print(f"    {r:<30} {n}")
    print(f"{'='*65}")


# ─── Paper metrics ────────────────────────────────────────────────────────────

def save_csv(traces: list[dict], path: Path) -> None:
    """Save the paper pipeline table as CSV — one row per function."""
    FIELDS = ["Function","Library","GT","Category",
               "Stage3","Stage4","Stage4_5","BEF","Judge","Final",
               "Confidence","Outcome"]
    rows = []
    for t in traces:
        lib = t.get("binary_id", "")[:3]
        gt  = "TP" if t.get("is_gt") else "-"
        cat = t.get("category", "?")

        s3 = t.get("stage3", {})
        stage3_col = (
            "Fwd+Sem"   if (s3.get("fwd", 0) > 0 and s3.get("sem", 0) > 0) else
            "Sem-only"  if s3.get("sem", 0) > 0 else
            "Structural" if s3.get("n_candidates", 0) > 0 else
            "None"
        )

        s4 = t.get("stage4", {})
        if s4.get("skipped") or s4.get("no_candidates") or not s4.get("n_assessments"):
            stage4_col = "Skipped"
        else:
            supp = s4.get("hypothesis_support", "?")
            conf = "YES" if s4.get("confirmed") else "NO"
            stage4_col = f"{supp}/{conf}"

        s45 = t.get("stage45", {})
        stage45_col = (
            "Skipped" if (s45.get("skipped") or not s45.get("n")) else
            f"{s45.get('confirmed',0)}/{s45.get('n',0)} conf"
        )

        bef = t.get("bef", {})
        bef_col = {
            "bayes_direct_confirm": "Direct Confirm",
            "bayes_direct_reject":  "Direct Reject",
            "needs_consensus":      "Consensus",
        }.get(bef.get("routing", ""), "No-Signal" if bef.get("skipped") else bef.get("routing","?"))

        cons = t.get("consensus", {})
        judge_col = (
            "Skipped" if cons.get("skipped") else
            cons.get("conflict_type", "?").replace("_", " ").title()
        )

        fnd = t.get("finding", {})
        confirmed = fnd.get("confirmed", False)
        exp = t.get("expected_cat", "?")
        outcome = (
            "TP" if (exp == "TP"   and confirmed) else
            "FN" if (exp == "TP"   and not confirmed) else
            "FP" if (exp == "SAFE" and confirmed) else
            "TN"
        )
        rows.append({
            "Function":   t.get("name", "?"),
            "Library":    lib,
            "GT":         gt,
            "Category":   cat,
            "Stage3":     stage3_col,
            "Stage4":     stage4_col,
            "Stage4_5":   stage45_col,
            "BEF":        bef_col,
            "Judge":      judge_col,
            "Final":      "Confirmed" if confirmed else "Rejected",
            "Confidence": round(fnd.get("confidence", 0.0), 3),
            "Outcome":    outcome,
        })

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  saved -> {path}")


def compute_paper_metrics(traces: list[dict]) -> dict:
    """Compute paper-quality metrics: precision/recall, BEF savings, Stage4.5 gain, confidence hist."""
    from collections import Counter

    # Outcome counts
    actual_tp = sum(1 for t in traces if t.get("expected_cat")=="TP"   and t.get("finding",{}).get("confirmed"))
    actual_fp = sum(1 for t in traces if t.get("expected_cat")=="SAFE" and t.get("finding",{}).get("confirmed"))
    actual_fn = sum(1 for t in traces if t.get("expected_cat")=="TP"   and not t.get("finding",{}).get("confirmed"))
    actual_tn = sum(1 for t in traces if t.get("expected_cat")=="SAFE" and not t.get("finding",{}).get("confirmed"))
    precision = actual_tp / (actual_tp + actual_fp) if (actual_tp + actual_fp) > 0 else 0.0
    recall    = actual_tp / (actual_tp + actual_fn) if (actual_tp + actual_fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    # BEF routing
    routing_counts: Counter = Counter()
    for t in traces:
        bef = t.get("bef", {})
        routing_counts[bef.get("routing", "no_signal") if not bef.get("skipped") else "no_signal"] += 1
    n_with_signal  = sum(v for k,v in routing_counts.items() if k != "no_signal")
    n_consensus    = routing_counts["needs_consensus"]
    consensus_rate = n_consensus / n_with_signal if n_with_signal > 0 else 0.0
    bef_savings_pct = (1.0 - consensus_rate) * 100

    # Stage 4.5 contribution: TPs where Stage 4 did NOT confirm but final is TP
    s45_exclusive = sum(
        1 for t in traces
        if t.get("expected_cat") == "TP"
        and t.get("finding", {}).get("confirmed")
        and (
            not t.get("stage4", {}).get("confirmed")
            or t.get("consensus", {}).get("conflict_type") == "one_sided_semantic"
        )
    )
    # TPs where sem_score pushed significantly above mem_score
    s45_pushed = sum(
        1 for t in traces
        if t.get("finding", {}).get("confirmed")
        and (t.get("bef", {}).get("semantic_score", 0.5) >
             t.get("bef", {}).get("memory_score", 0.5) + 0.15)
    )
    s4_alone_tps = actual_tp - s45_exclusive

    # Confidence histograms
    def _hist(confs: list) -> dict:
        bins = {"0.0-0.30":0,"0.30-0.55":0,"0.55-0.70":0,"0.70-0.85":0,"0.85-1.00":0}
        for c in confs:
            if   c < 0.30: bins["0.0-0.30"]  += 1
            elif c < 0.55: bins["0.30-0.55"] += 1
            elif c < 0.70: bins["0.55-0.70"] += 1
            elif c < 0.85: bins["0.70-0.85"] += 1
            else:          bins["0.85-1.00"] += 1
        return bins

    tp_confs   = [t["finding"]["confidence"] for t in traces
                  if t.get("expected_cat")=="TP"   and t.get("finding",{}).get("confirmed")]
    fp_confs   = [t["finding"]["confidence"] for t in traces
                  if t.get("expected_cat")=="SAFE" and t.get("finding",{}).get("confirmed")]

    # Disagreement log
    disagreements = [
        {
            "func":           t.get("name"),
            "expected":       t.get("expected_cat"),
            "conflict_type":  t.get("consensus",{}).get("conflict_type",""),
            "resolution":     t.get("consensus",{}).get("conflict_resolution","")[:180],
            "bef_gap":        t.get("bef",{}).get("gap",0),
            "confirmed":      t.get("finding",{}).get("confirmed"),
        }
        for t in traces if not t.get("consensus",{}).get("skipped") and "consensus" in t
    ]

    return {
        "actual_tp": actual_tp, "actual_fp": actual_fp,
        "actual_fn": actual_fn, "actual_tn": actual_tn,
        "precision": round(precision,3), "recall": round(recall,3), "f1": round(f1,3),
        "bef_routing": dict(routing_counts),
        "consensus_rate": round(consensus_rate,3),
        "bef_savings_pct": round(bef_savings_pct,1),
        "s45_exclusive_tps": s45_exclusive,
        "s45_pushed": s45_pushed,
        "s4_alone_tps": s4_alone_tps,
        "tp_conf_hist": _hist(tp_confs),
        "fp_conf_hist": _hist(fp_confs),
        "tp_conf_mean": round(sum(tp_confs)/len(tp_confs),3) if tp_confs else 0.0,
        "fp_conf_mean": round(sum(fp_confs)/len(fp_confs),3) if fp_confs else 0.0,
        "disagreements": disagreements,
    }


def print_paper_metrics(m: dict) -> None:
    line = "=" * 65
    print(f"\n{line}")
    print(f"  PAPER METRICS")
    print(f"{line}")

    print(f"\n  Precision / Recall / F1:")
    print(f"    TP={m['actual_tp']}  FP={m['actual_fp']}  "
          f"FN={m['actual_fn']}  TN={m['actual_tn']}")
    print(f"    Precision = {m['precision']:.3f}")
    print(f"    Recall    = {m['recall']:.3f}")
    print(f"    F1        = {m['f1']:.3f}")

    print(f"\n  BEF Routing:")
    for k, v in sorted(m["bef_routing"].items()):
        print(f"    {k:<35} {v}")
    print(f"    Consensus Rate   = {m['consensus_rate']:.1%}  "
          f"({m['bef_routing'].get('needs_consensus',0)} / "
          f"{sum(v for k,v in m['bef_routing'].items() if k!='no_signal')} with signal)")
    print(f"    BEF Savings      = {m['bef_savings_pct']:.1f}% fewer judge calls "
          f"vs. routing all to consensus")

    print(f"\n  Stage 4.5 Contribution:")
    print(f"    Stage 4 alone TPs       = {m['s4_alone_tps']}")
    print(f"    Stage 4.5 exclusive TPs = {m['s45_exclusive_tps']}  "
          f"(Stage 4 weak/no-confirm, Stage 4.5 confirmed)")
    print(f"    Stage 4.5 signal boost  = {m['s45_pushed']}  "
          f"(sem_score > mem_score + 0.15)")

    print(f"\n  Confidence Separation (TP vs SAFE-expected):")
    print(f"    TP   mean={m['tp_conf_mean']:.3f}  bins={m['tp_conf_hist']}")
    print(f"    Safe mean={m['fp_conf_mean']:.3f}  bins={m['fp_conf_hist']}")

    n_dis = len(m["disagreements"])
    print(f"\n  Disagreements routed to ConsensusEngine: {n_dis}")
    for d in m["disagreements"]:
        fin = "CONF" if d.get("confirmed") else "REJ "
        print(f"    {d['func'][:32]:<32} gap={d.get('bef_gap',0):.3f}  "
              f"{str(d.get('conflict_type','')):<22}  -> {fin}")
    print(f"{line}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Full pipeline integration test")
    ap.add_argument("--budget",      type=int, default=30, help="Total functions (default 30)")
    ap.add_argument("--library",     default=None, choices=list(LIBRARY_PREFIX.keys()))
    ap.add_argument("--binary",      default=None, help="Force specific binary (e.g. PNG001)")
    ap.add_argument("--skip-stage4", action="store_true", help="Skip Stage 4 + 4.5 LLM calls")
    ap.add_argument("--skip-stage45",action="store_true", help="Skip Stage 4.5 only")
    ap.add_argument("--no-consensus",action="store_true", help="Skip ConsensusEngine")
    ap.add_argument("--named-only",  action="store_true", help="Only run named targets")
    ap.add_argument("--stratified",  action="store_true",
                    help="Run the 30-function stratified evaluation (5 libs x 6 funcs)")
    ap.add_argument("--provider",    default="openrouter")
    ap.add_argument("--save",        action="store_true")
    args = ap.parse_args()

    gt = load_gt()
    mode = "stratified" if args.stratified else ("named_only" if args.named_only else "auto")
    print(f"\nIntegration test: mode={mode}  budget={args.budget}  "
          f"library={args.library or 'all'}  skip_s4={args.skip_stage4}")

    # Collect functions
    if args.stratified:
        entries = collect_stratified(gt)
    else:
        entries = collect_functions(
            budget          = args.budget,
            library_filter  = args.library,
            binary_override = args.binary,
            named_only      = args.named_only,
            gt              = gt,
        )
    if not entries:
        print("ERROR: no functions found — check dataset path or --binary flag")
        sys.exit(1)
    n_gt   = sum(1 for e in entries if e.is_gt)
    n_safe = sum(1 for e in entries if e.expected_cat == "SAFE")
    print(f"Selected {len(entries)} functions  ({n_gt} GT-TP, {n_safe} SAFE/non-GT)")

    # Build pipeline components
    print("Building Stage 3 orchestrator...")
    orch, store = build_stage3()

    s4 = None
    if not args.skip_stage4:
        print("Building Stage 4 HypothesisEvaluationEngine...")
        try:
            s4 = build_stage4(args.provider)
            if not s4.llm_enabled:
                print("  Stage 4 LLM disabled (no API key) — structural mode only")
        except Exception as e:
            print(f"  Stage 4 unavailable: {e}")

    s45 = None
    if not args.skip_stage4 and not args.skip_stage45:
        print("Building Stage 4.5 OrthogonalSemanticAnalyzer...")
        try:
            s45 = build_stage45()
            if not s45.enabled:
                print("  Stage 4.5 LLM disabled (no API key)")
        except Exception as e:
            print(f"  Stage 4.5 unavailable: {e}")

    print("Building BEF + ConsensusEngine...")
    bef, fusion = build_bef_and_engine(
        provider=args.provider,
        use_consensus=not args.no_consensus,
    )
    has_engine = fusion._engine is not None and fusion._engine.enabled
    print(f"  ConsensusEngine {'enabled' if has_engine else 'disabled (no key)'}")

    # Run pipeline
    traces: list[dict] = []
    t_total = time.perf_counter()

    print(f"\n{'-'*65}")
    for i, entry in enumerate(entries, 1):
        try:
            trace = run_function(
                entry        = entry,
                orch         = orch,
                s4           = s4,
                s45          = s45,
                bef          = bef,
                fusion       = fusion,
                skip_stage4  = args.skip_stage4,
                skip_stage45 = args.skip_stage45,
            )
            print_trace(i, len(entries), trace)
            traces.append(trace)
        except Exception as e:
            print(f"\n[{i:02d}/{len(entries)}] {entry.name}  ERROR: {e}")
            traces.append({
                "name": entry.name, "binary_id": entry.binary_id,
                "expected_cat": entry.expected_cat, "is_gt": entry.is_gt,
                "source": entry.source, "category": entry.category,
                "error": str(e), "finding": {"confirmed": False},
            })

    store.close()
    elapsed = time.perf_counter() - t_total

    print_summary(traces, elapsed)

    # Paper metrics
    metrics: dict = {}
    if args.stratified or args.save:
        metrics = compute_paper_metrics(traces)
        print_paper_metrics(metrics)

    if args.save:
        tag = "stratified" if args.stratified else "integration"
        json_path = OUT_DIR / f"{tag}_trace.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump({
                "config": {
                    "mode": mode, "budget": args.budget, "library": args.library,
                    "skip_stage4": args.skip_stage4, "skip_stage45": args.skip_stage45,
                },
                "traces":    traces,
                "metrics":   metrics,
                "elapsed_s": round(elapsed, 1),
            }, fh, indent=2)
        print(f"JSON saved -> {json_path}")
        # CSV table
        csv_path = OUT_DIR / f"{tag}_table.csv"
        save_csv(traces, csv_path)


if __name__ == "__main__":
    main()
