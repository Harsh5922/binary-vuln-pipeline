"""
pipeline.py

Unified entry point for the binary vulnerability analysis pipeline.

Usage
-----
    python pipeline.py <binary> [options]

    python pipeline.py ./vuln_O0 --no-llm --budget 50 --html --json
    python pipeline.py ./firmware.exe --provider openrouter --force

    # Resume from Stage 3 using existing pcode_ranked.jsonl (skip Ghidra)
    python pipeline.py --resume-from 3 --output-dir D:/results/SND001/vuln_O0 --provider openrouter --json

Options
-------
    --output-dir PATH     directory for all output files (default: cwd)
    --resume-from N       resume from stage N (3, 4, or 5); binary arg not required
    --no-llm              disable LLM review; auto-confirm LIBRARY_MATCH only
    --budget N            max ranked functions passed to taint analysis (default: 300)
    --min-score FLOAT     filter score floor (default: 0.15)
    --provider PROV       groq | openrouter | gemini | anthropic (default: groq)
    --force               ignore checkpoints, re-run all stages
    --html                also write HTML report
    --json                also write JSON report
    --log-level LEVEL     DEBUG | INFO | WARNING (default: INFO)

Stages
------
    1  extract   binary        → pcode.jsonl
    2  filter    pcode.jsonl   → pcode_ranked.jsonl
    3  taint     ranked funcs  → VulnCandidate list  (in-memory)
    4  reason    candidates    → Finding list         (in-memory)
    5  report    findings      → vulnerability_report.txt [.html] [.json]

Checkpoint / resume
-------------------
Each stage checks whether its output file already exists.  If it does and
--force is not set, the stage loads from the existing file and skips
recomputation.  This means a failed or interrupted run restarts from the
last successful stage.

Use --resume-from 3 when pcode_ranked.jsonl already exists (e.g. Stages 1+2
ran previously). This skips Ghidra extraction entirely and runs only taint
analysis, LLM confirmation, and report generation. Saves hours per binary.
"""

from __future__ import annotations
from load_env import load_env
load_env()

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 encoding on Windows (only when run as script, not when imported)
def _setup_utf8_stdout() -> None:
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure project modules can be imported
project_root = Path(__file__).parent


def _ensure_project_on_path():
    """Re-insert project root into sys.path — pyghidra can drop it."""
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_on_path()

log = logging.getLogger(__name__)

# ── In-scope vulnerability types ──────────────────────────────────────────────
# Maps to CWE-119/120/122/125/190/191/197/476/787 (buffer/heap/integer/null).
# Findings with any other vuln_type are silently dropped before reporting.
IN_SCOPE_VULN_TYPES: frozenset[str] = frozenset({
    "buffer_overflow",        # CWE-119 / 120
    "heap_overflow",          # CWE-122
    "out_of_bounds_read",     # CWE-125
    "out_of_bounds_write",    # CWE-787
    "integer_overflow",       # CWE-190
    "integer_underflow",      # CWE-191
    "integer_truncation",     # CWE-197
    "null_dereference",       # CWE-476
    "write_what_where",       # CWE-119 / 787
    "check_bypass",           # integer-check bypass → leads to CWE-190/119
    "unchecked_arithmetic",   # CWE-190 / 191
    "logic_bug",              # integer/bounds logic errors (Track 2)
})


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — P-code extraction
# ─────────────────────────────────────────────────────────────────────────────

def stage1_extract(binary_path: Path, out: Path, cfg) -> bool:
    """
    Extract SSA P-code from binary → pcode.jsonl.
    Returns True on success, False on failure.
    """
    if out.exists() and out.stat().st_size > 0 and not cfg.force:
        log.info("Stage 1: checkpoint hit (%s exists) — skipping extraction", out.name)
        return True

    log.info("Stage 1: extracting P-code from %s …", binary_path.name)
    t0 = time.perf_counter()

    try:
        from extractor import PcodeExtractor
    except ImportError as e:
        log.error(
            "Stage 1 failed: cannot import extractor — %s\n"
            "  Make sure pyghidra is installed and GHIDRA_INSTALL_DIR is set.",
            e,
        )
        return False

    try:
        ex    = PcodeExtractor(binary_path)
        count = ex.to_jsonl(out)
        log.info(
            "Stage 1 done in %.1fs — %d functions → %s",
            time.perf_counter() - t0, count, out.name,
        )
        return True
    except Exception as e:
        log.error("Stage 1 failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Filter / ranking
# ─────────────────────────────────────────────────────────────────────────────

def stage2_filter(pcode_path: Path, out: Path, cfg) -> bool:
    """
    Rank functions by vulnerability interest → pcode_ranked.jsonl.
    Returns True on success, False on failure.
    """
    if out.exists() and out.stat().st_size > 0 and not cfg.force:
        log.info("Stage 2: checkpoint hit (%s exists) — skipping filter", out.name)
        return True

    log.info("Stage 2: ranking functions from %s …", pcode_path.name)
    t0 = time.perf_counter()

    _ensure_project_on_path()
    try:
        from filter_agent import FunctionFilterAgent
    except ImportError as e:
        log.error("Stage 2 failed: cannot import filter_agent — %s", e)
        return False

    try:
        agent = FunctionFilterAgent.from_jsonl(
            pcode_path,
            budget     = cfg.budget,
            min_score  = cfg.min_score,
            store_path = "",  # Stage 2 must not use pattern_store — keeps ranking stable across runs
        )
        agent.save_ranked(out)
        stats = agent.stats()
        log.info(
            "Stage 2 done in %.1fs — kept %d / %d (reduction %s) → %s",
            time.perf_counter() - t0,
            stats["kept"], stats["total"], stats["reduction_pct"],
            out.name,
        )
        return True
    except Exception as e:
        log.error("Stage 2 failed: %s", e)
        return False



# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.5 — Semantic Recovery (unknown function summarisation)
# ─────────────────────────────────────────────────────────────────────────────

def stage2_5_semantic_recovery(ranked_path: Path, cfg) -> dict:
    """
    For every NO_MATCH function in the ranked list, ask the LLM:
    "what does this function do?" and store the semantic summary.

    This is NOT vulnerability detection — that is Stage 4.
    This is knowledge extraction: role, data flow, argument semantics.

    The learning curve: Binary 1 makes N LLM calls.
    Binary 2 hits the cache for most of them → fewer calls.
    Binary 20: 0 LLM calls. This is Figure 1 of the paper.

    Returns stats dict for logging.
    """
    if cfg.no_llm:
        return {"processed": 0, "cached": 0, "new_summaries": 0, "llm_calls": 0}

    t0 = time.perf_counter()
    log.info("Stage 2.5: semantic recovery on %s …", ranked_path.name)

    _ensure_project_on_path()
    try:
        from semantic_recovery_agent import SemanticRecoveryAgent
        from pattern_store           import PatternStore
        from pattern_matcher         import PatternMatcher
    except ImportError as e:
        log.warning("Stage 2.5 skipped: %s", e)
        return {}

    # Load ranked functions
    funcs: list[dict] = []
    try:
        with ranked_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if not d.get("discarded"):
                        funcs.append(d)
    except Exception as e:
        log.warning("Stage 2.5 failed loading ranked file: %s", e)
        return {}

    if not funcs:
        return {}

    try:
        store_path = Path(cfg.output_dir) / "pattern_store.db"
        if not store_path.exists():
            store_path = Path("pattern_store.db")

        store   = PatternStore(str(store_path))
        matcher = PatternMatcher(store)

        agent = SemanticRecoveryAgent(
            provider = cfg.provider,  # stored for logging; Stage 2.5 always uses OpenRouter
            api_key  = None,          # reads OPENROUTER_API_KEY from environment
            delay_s  = 1.5,
            max_ops  = 40,
        )

        # count_only=True when --no-llm: still reports cache hit rate, no LLM calls
        count_only = cfg.no_llm or not agent.enabled

        # Budget decay: base budget on learned_patterns count (all cached LLM queries,
        # including null sentinels), then sharpen using the actual cache hit rate.
        # Using learned_patterns (not learned_rules) ensures the budget decays even
        # when most LLM responses are "not interesting" — producing Figure 1's curve.
        learned_count   = store.get_stats()["learned_patterns"]
        base_budget     = max(5, min(30, 30 - learned_count // 5))
        prev_hit_rate   = store.get_metadata("prev_stage25_hit_rate", 0.0)
        if prev_hit_rate > 0.8:
            llm_budget = max(3, base_budget // 2)
        elif prev_hit_rate > 0.6:
            llm_budget = max(4, int(base_budget * 0.75))
        else:
            llm_budget = base_budget

        stats = agent.process_binary(
            functions       = funcs,
            pattern_store   = store,
            pattern_matcher = matcher,
            budget          = llm_budget,
            count_only      = count_only,
        )

        # Persist hit rate so next binary can sharpen its budget (P3)
        processed = stats.get("processed", 0)
        cached    = stats.get("cached", 0)
        if processed > 0:
            store.set_metadata("prev_stage25_hit_rate", cached / processed)

        # Log the learning curve data point
        log.info(
            "Stage 2.5 done in %.1fs — processed=%d cached=%d "
            "new_summaries=%d llm_calls=%d budget=%d  ← LEARNING CURVE DATA",
            time.perf_counter() - t0,
            processed,
            cached,
            stats.get("new_summaries", 0),
            stats.get("llm_calls", 0),
            llm_budget,
        )

        store.close()
        return stats

    except Exception as e:
        log.warning("Stage 2.5 failed: %s", e)
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Taint analysis
# ─────────────────────────────────────────────────────────────────────────────

def stage3_taint(ranked_path: Path, cfg) -> Optional[tuple[list, dict]]:
    """
    Run taint analysis on all ranked functions.
    Returns (taint_results, func_map) or None on failure.
    """
    log.info("Stage 3: running taint analysis on %s …", ranked_path.name)
    t0 = time.perf_counter()

    _ensure_project_on_path()
    try:
        from pattern_store      import PatternStore
        from pattern_matcher    import PatternMatcher
        from taint_engine       import TaintEngine
        # Stage 3 redesign: try to load the Hybrid Semantic Data-Flow orchestrator.
        # Falls back to bare TaintEngine if any new module is missing.
        try:
            from stage3_orchestrator import Stage3Orchestrator
            _USE_ORCHESTRATOR = True
        except ImportError:
            _USE_ORCHESTRATOR = False
    except ImportError as e:
        log.error("Stage 3 failed: %s", e)
        return None

    # Load ranked functions
    funcs: list[dict] = []
    try:
        with ranked_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if not d.get("discarded"):
                    funcs.append(d)
    except Exception as e:
        log.error("Stage 3 failed loading %s: %s", ranked_path.name, e)
        return None

    if not funcs:
        log.warning("Stage 3: no functions to analyze in %s", ranked_path.name)
        return [], {}

    log.info("Stage 3: analyzing %d functions …", len(funcs))

    try:
        store_path = Path(cfg.output_dir) / "pattern_store.db"
        # Fall back to local pattern_store.db if output-dir one doesn't exist
        if not store_path.exists():
            store_path = Path("pattern_store.db")

        store   = PatternStore(str(store_path))
        matcher = PatternMatcher(store)

        # Build inter-procedural call summaries (callees before callers)
        summary_db = None
        try:
            from interprocedural import SummaryDatabase
            log.info("Stage 3: building inter-procedural call summaries …")
            summary_db = SummaryDatabase(matcher)
            summary_db.build(funcs)
            log.info("Stage 3: %d summaries built", summary_db.summary_count)
        except ImportError:
            log.warning("interprocedural.py not found — skipping inter-proc summaries")
        except Exception as e:
            log.warning("Inter-proc summary build failed (%s) — continuing without", e)

        if _USE_ORCHESTRATOR:
            engine = Stage3Orchestrator(matcher, summary_db=summary_db)
            log.info("Stage 3: using Hybrid Semantic Data-Flow Analysis (3A+3B+3C+3D)")
        else:
            engine = TaintEngine(matcher, summary_db=summary_db)
            log.info("Stage 3: using TaintEngine (Stage3Orchestrator unavailable)")
        taint_results = engine.analyze_all(funcs)
        store.close()

        total_cands = sum(len(r.vulns) for r in taint_results)
        log.info(
            "Stage 3 done in %.1fs — %d functions → %d candidates",
            time.perf_counter() - t0, len(funcs), total_cands,
        )
        func_map = {f["name"]: f for f in funcs}
        return taint_results, func_map
    except Exception as e:
        log.error("Stage 3 failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — AI Security Analyst (Hypothesis Evaluation Engine)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_reason(taint_results, func_map: dict, cfg) -> Optional[tuple]:
    """
    AI Security Analyst: peer-review hypothesis evaluation for each VulnCandidate.

    Returns (assessments, findings) tuple on success, None on failure.
      assessments — raw Assessment objects (needed by Stage 4.6 Fusion Judge)
      findings    — Assessment.to_finding() list (backward-compat with Stage 5)

    Falls back to legacy ReasoningAgent: returns ([], findings) so Stage 4.6
    fast-paths (agrees_safe / one_sided) still work without Assessment objects.
    """
    total_cands = sum(len(r.vulns) for r in taint_results)
    if total_cands == 0:
        log.info("Stage 4: no candidates to review — skipping")
        return [], []

    log.info("Stage 4 (AI Analyst): evaluating %d candidates …", total_cands)
    t0 = time.perf_counter()

    _ensure_project_on_path()

    # Try new HypothesisEvaluationEngine (peer-review architecture) first
    try:
        from stage4_analyst import HypothesisEvaluationEngine
        _USE_ANALYST = True
    except ImportError:
        _USE_ANALYST = False

    if _USE_ANALYST:
        try:
            llm_mode = "warn" if cfg.no_llm else "require"
            engine   = HypothesisEvaluationEngine(
                provider = cfg.provider,
                llm_mode = llm_mode,
                api_key  = "" if cfg.no_llm else None,
            )
            assessments = engine.evaluate_all(taint_results, func_map, delay_s=2.0)
            findings    = [a.to_finding() for a in assessments]
            confirmed   = sum(1 for f in findings if f.confirmed)
            log.info(
                "Stage 4 done in %.1fs — %d assessments, %d confirmed",
                time.perf_counter() - t0, len(assessments), confirmed,
            )
            return assessments, findings
        except EnvironmentError as e:
            log.error("%s", e)
            return None
        except Exception as e:
            log.error("Stage 4 (AI Analyst) failed: %s — falling back to ReasoningAgent", e)

    # Fallback: legacy ReasoningAgent (no Assessment objects)
    try:
        from reasoning_agent import ReasoningAgent
    except ImportError as e:
        log.error("Stage 4 failed: %s", e)
        return None

    try:
        llm_mode = "warn" if cfg.no_llm else "require"
        agent    = ReasoningAgent(
            provider = cfg.provider,
            llm_mode = llm_mode,
            api_key  = "" if cfg.no_llm else None,
        )
        findings  = agent.review_all(taint_results, func_map, delay_s=2.0)
        confirmed = sum(1 for f in findings if f.confirmed)
        log.info(
            "Stage 4 done in %.1fs — %d findings, %d confirmed (legacy ReasoningAgent)",
            time.perf_counter() - t0, len(findings), confirmed,
        )
        return [], findings   # no Assessment objects available from legacy path
    except EnvironmentError as e:
        log.error("%s", e)
        return None
    except Exception as e:
        log.error("Stage 4 failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4.5 — Orthogonal Semantic Vulnerability Analysis
# ─────────────────────────────────────────────────────────────────────────────

def stage4_5_track2(
    ranked_path:       Path,
    findings:          list,
    cfg,
    taint_results:     object = None,
    stage4_assessments: list  = None,
    func_map:          dict   = None,
) -> list:
    """
    Orthogonal Semantic Analysis + Stage 4.6 Fusion Judge.

    stage4_assessments — raw Assessment objects from Stage 4 (for Fusion Judge).
                         When provided, the Fusion Judge arbitrates between
                         Memory Safety Analyst (Stage 4) and Semantic Analyst (Stage 4.5).
                         When None or [], falls back to simple confidence-boost merge.

    Returns new/additional Finding objects NOT already confirmed in Stage 4.
    When Fusion Judge runs, returns FinalVerdict findings (which supersede Stage 4 ones).
    Falls back to legacy Track2Agent when stage4_orthogonal.py is unavailable.
    """
    if cfg.no_llm:
        return []

    _ensure_project_on_path()

    import os
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        log.info("Stage 4.5: OPENROUTER_API_KEY not set — skipping")
        return []

    # Load ranked functions
    funcs: list[dict] = []
    try:
        with ranked_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if not d.get("discarded"):
                        funcs.append(d)
    except Exception as e:
        log.warning("Stage 4.5: failed loading ranked file: %s", e)
        return []

    t0 = time.perf_counter()

    # Try new OrthogonalSemanticAnalyzer + FusionJudge first
    try:
        from stage4_orthogonal import OrthogonalSemanticAnalyzer
        from stage4_fusion     import CandidateFusion
        _USE_ORTHOGONAL = True
    except ImportError:
        _USE_ORTHOGONAL = False

    if _USE_ORTHOGONAL:
        try:
            confirmed_fns = {f.func_name for f in findings if f.confirmed}
            analyzer = OrthogonalSemanticAnalyzer(api_key=api_key, delay_s=2.0)

            semantic_assessments = analyzer.analyze_all(
                funcs         = funcs,
                budget        = 15,
                skip_fn_names = confirmed_fns,  # informational — doesn't skip
            )

            # Stage 4.6: Fusion Judge arbitrates between Memory Safety + Semantic analysts
            store_path = Path(cfg.output_dir) / "pattern_store.db"
            fusion     = CandidateFusion(
                provider           = cfg.provider,
                pattern_store_path = str(store_path) if store_path.exists() else None,
                use_judge          = bool(stage4_assessments),  # judge only when we have assessments
            )
            all_fusion_findings = fusion.merge(
                assessments          = stage4_assessments or [],
                semantic_assessments = semantic_assessments,
                func_map             = func_map or {},
            )

            # When judge ran: fusion output supersedes Stage 4 findings for judged functions.
            # Collect which functions were handled by the judge (have calibration.stage=4.6).
            judged_fns: set[str] = set()
            new_findings: list = []
            for f in all_fusion_findings:
                cal = f.calibration or {}
                if cal.get("stage") in ("4.6_fusion_judge", "4.6_bef"):
                    judged_fns.add(f.func_name)
                    if f.confirmed:
                        new_findings.append(f)
                else:
                    # Not judged (Stage 4.5 only or legacy merge)
                    existing_keys = {
                        (ex.func_name, ex.vuln_type, ex.sink_fn or "")
                        for ex in findings
                    }
                    if (f.func_name, f.vuln_type, f.sink_fn or "") not in existing_keys and f.confirmed:
                        new_findings.append(f)

            log.info(
                "Stage 4.5+4.6 done in %.1fs — judge ran on %d functions, "
                "%d new/updated confirmed findings",
                time.perf_counter() - t0, len(judged_fns), len(new_findings),
            )
            return new_findings

        except Exception as e:
            log.warning("Stage 4.5 (Orthogonal) failed: %s — falling back to Track2", e)

    # Fallback: legacy Track2Agent
    try:
        from track2_agent            import Track2Agent, track2_result_to_finding
        from semantic_recovery_agent import _KNOWN_ROLES
        from pattern_store           import PatternStore
    except ImportError as e:
        log.warning("Stage 4.5 skipped: %s", e)
        return []

    try:
        store_path = Path(cfg.output_dir) / "pattern_store.db"
        if not store_path.exists():
            store_path = Path("pattern_store.db")
        store        = PatternStore(str(store_path))
        callee_roles = {**_KNOWN_ROLES, **store.get_all_callee_roles()}
        store.close()
    except Exception:
        try:
            from semantic_recovery_agent import _KNOWN_ROLES
            callee_roles = dict(_KNOWN_ROLES)
        except Exception:
            callee_roles = {}

    confirmed_fns  = {f.func_name for f in findings if f.confirmed}
    zero_cand_fns: set[str] = set()
    if taint_results:
        zero_cand_fns = {
            r.func_name for r in taint_results
            if not r.vulns and r.func_name not in confirmed_fns
        }

    agent = Track2Agent(api_key=api_key, delay_s=2.0, max_ops=80)
    track2_results = agent.process_binary(
        functions          = funcs,
        confirmed_fn_names = confirmed_fns,
        callee_roles       = callee_roles,
        budget             = 15,
        priority_fn_names  = zero_cand_fns,
    )
    new_findings = [track2_result_to_finding(r) for r in track2_results]
    log.info(
        "Stage 4.5 done in %.1fs — Track 2 (legacy) found %d new confirmed vulns",
        time.perf_counter() - t0, len(new_findings),
    )
    return new_findings


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Report generation
# ─────────────────────────────────────────────────────────────────────────────

def stage5_report(
    findings,
    taint_results,
    out_dir: Path,
    binary_name: str,
    cfg,
) -> None:
    """Write text / HTML / JSON reports based on cfg flags."""
    _ensure_project_on_path()
    try:
        from reasoning_agent import generate_report, generate_html_report, generate_json_report
    except ImportError as e:
        log.error("Stage 5 failed: %s", e)
        return

    stem = out_dir / "vulnerability_report"

    # Always write text report
    txt_path = Path(str(stem) + ".txt")
    generate_report(findings, taint_results, txt_path)
    log.info("Stage 5: text report → %s", txt_path.name)

    if cfg.html:
        html_path = Path(str(stem) + ".html")
        generate_html_report(findings, taint_results, html_path, binary_name)
        log.info("Stage 5: HTML report → %s", html_path.name)

    if cfg.json:
        json_path = Path(str(stem) + ".json")
        generate_json_report(findings, taint_results, json_path, binary_name)
        log.info("Stage 5: JSON report → %s", json_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg) -> int:
    """
    Orchestrate all 5 stages.
    Returns 0 on success, 1 on failure.
    """
    resume_from = getattr(cfg, "resume_from", None) or 1

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_path = out_dir / "pcode_ranked.jsonl"
    pcode_path  = out_dir / "pcode.jsonl"

    # Determine binary info (not needed when resuming from stage 2+)
    if resume_from <= 1:
        binary_path = Path(cfg.binary)
        if not binary_path.exists():
            log.error("Binary not found: %s", binary_path)
            return 1
        binary_stem = binary_path.stem or binary_path.name
    else:
        binary_path = None
        binary_stem = out_dir.name  # use directory name as identifier

    t_start = time.perf_counter()
    sep     = "─" * 62

    print(f"\n{sep}")
    if binary_path:
        print(f"  Binary     : {binary_path}")
    print(f"  Output dir : {out_dir.resolve()}")
    if resume_from > 1:
        print(f"  Resume     : from Stage {resume_from}")
    print(f"  Budget     : {cfg.budget}  min_score={cfg.min_score}")
    print(f"  LLM        : {'disabled' if cfg.no_llm else cfg.provider}")
    print(f"{sep}\n")

    # ── Stage 1 ───────────────────────────────────────────────────────
    if resume_from <= 1:
        s1_ok = stage1_extract(binary_path, pcode_path, cfg)
        if not s1_ok:
            if ranked_path.exists() and ranked_path.stat().st_size > 0 and not cfg.force:
                log.warning(
                    "Stage 1 failed but valid %s found — resuming from stage 2 checkpoint",
                    ranked_path.name,
                )
                s1_ok = None  # sentinel: skip stage 2 too
            else:
                return 1
    else:
        s1_ok = True
        log.info("Stage 1: skipped (--resume-from %d)", resume_from)

    # ── Stage 2 ───────────────────────────────────────────────────────
    if resume_from <= 2:
        if s1_ok is None:
            log.info("Stage 2: skipped (using existing %s)", ranked_path.name)
        elif not stage2_filter(pcode_path, ranked_path, cfg):
            return 1
    else:
        if not ranked_path.exists() or ranked_path.stat().st_size == 0:
            log.error(
                "Stage 2: --resume-from %d requires %s to exist — not found",
                resume_from, ranked_path,
            )
            return 1
        log.info("Stage 2: skipped (--resume-from %d, using %s)", resume_from, ranked_path.name)

    # ── Stage 2.5 — Semantic Recovery ────────────────────────────────
    if resume_from <= 3:
        pass  # run normally below
    else:
        log.info("Stage 2.5: skipped (--resume-from %d)", resume_from)

    sem_stats = stage2_5_semantic_recovery(ranked_path, cfg) if resume_from <= 3 else {}
    if sem_stats.get("llm_calls", 0) > 0:
        log.info(
            "Semantic recovery: %d new summaries, %d cached",
            sem_stats.get("new_summaries", 0),
            sem_stats.get("cached", 0),
        )

    # ── Stage 3 ───────────────────────────────────────────────────────
    result3 = stage3_taint(ranked_path, cfg)
    if result3 is None:
        return 1
    taint_results, func_map = result3

    # ── Stage 4 ───────────────────────────────────────────────────────
    result4 = stage4_reason(taint_results, func_map, cfg)
    if result4 is None:
        return 1
    s4_assessments, findings = result4

    # ── Stage 4.5 + 4.6 — Orthogonal + Fusion Judge ───────────────────
    # FusionJudge arbitrates between Stage 4 (Memory Safety Analyst)
    # and Stage 4.5 (Semantic Analyst) for each flagged function.
    track2_findings = stage4_5_track2(
        ranked_path,
        findings,
        cfg,
        taint_results      = taint_results,
        stage4_assessments = s4_assessments,
        func_map           = func_map,
    )
    if track2_findings:
        # For judged functions: FinalVerdict supersedes Stage 4 finding.
        # Deduplicate: judge output wins over direct Stage 4 output.
        judged_keys = {
            (f.func_name, f.vuln_type, f.sink_fn or "")
            for f in track2_findings
            if (f.calibration or {}).get("stage") in ("4.6_fusion_judge", "4.6_bef")
        }
        if judged_keys:
            # Remove Stage 4 findings superseded by BEF or Consensus Engine
            findings = [
                f for f in findings
                if (f.func_name, f.vuln_type, f.sink_fn or "") not in judged_keys
            ]
        log.info(
            "Stage 4.5+4.6: %d new/updated findings  (%d BEF-direct, %d via ConsensusEngine)",
            len(track2_findings),
            sum(1 for f in track2_findings
                if (f.calibration or {}).get("stage") == "4.6_bef"),
            sum(1 for f in track2_findings
                if (f.calibration or {}).get("stage") == "4.6_fusion_judge"),
        )
        findings = findings + track2_findings

    # ── Scope filter — drop out-of-scope vuln types before reporting ──
    # Only buffer/heap overflow, integer overflow/truncation, OOB read/write,
    # null dereference are counted (matches paper evaluation CWE filter).
    before_filter = len([f for f in findings if f.confirmed])
    findings = [
        f for f in findings
        if not f.confirmed
        or f.vuln_type.lower() in IN_SCOPE_VULN_TYPES
    ]
    after_filter = len([f for f in findings if f.confirmed])
    if before_filter != after_filter:
        log.info(
            "Scope filter: dropped %d out-of-scope confirmed findings "
            "(kept %d in-scope)",
            before_filter - after_filter, after_filter,
        )

    # ── Stage 5 ───────────────────────────────────────────────────────
    stage5_report(findings, taint_results, out_dir, binary_stem, cfg)

    # ── Item 2: LLM Cost Report ───────────────────────────────────────
    try:
        from llm_cost_tracker import GLOBAL_TRACKER
        if GLOBAL_TRACKER.totals().calls > 0:
            GLOBAL_TRACKER.print_table(label=binary_stem)
            GLOBAL_TRACKER.save_json(out_dir / "llm_cost_report.json")
    except Exception as _ce:
        log.debug("Cost tracker error: %s", _ce)

    # ── Summary ───────────────────────────────────────────────────────
    confirmed  = [f for f in findings if f.confirmed]
    total_time = time.perf_counter() - t_start

    print(f"\n{sep}")
    print(f"  PIPELINE COMPLETE  ({total_time:.1f}s)")
    print(f"{sep}")
    print(f"  Functions analyzed : {len(taint_results)}")
    print(f"  Candidates found   : {sum(len(r.vulns) for r in taint_results)}")
    print(f"  Confirmed vulns    : {len(confirmed)}")
    print(f"  Reports written to : {out_dir.resolve()}")

    if confirmed:
        print(f"\n  TOP CONFIRMED VULNERABILITIES:")
        for f in sorted(confirmed, key=lambda x: {"critical":0,"high":1,"medium":2,"low":3,"info":4}.get(x.severity, 5)):
            print(f"    [{f.severity.upper()}]  {f.vuln_type}  in {f.func_name}  sink={f.sink_fn}")

    print(f"{sep}\n")

    # ── Snapshot (reproducibility record) ────────────────────────────
    try:
        from pipeline_snapshot import save_snapshot
        save_snapshot(out_dir=out_dir, cfg=cfg, runtime_s=total_time)
    except Exception as _se:
        log.debug("Snapshot save failed: %s", _se)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Item 5: Active Learning — FP feedback CLI handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_active_learning_cmd(cfg) -> None:
    """Handle --reject-fp and --list-suppressions sub-commands."""
    _ensure_project_on_path()
    try:
        from pattern_store import PatternStore
    except ImportError:
        log.error("Cannot import PatternStore")
        return

    db_path = Path(cfg.output_dir) / "pattern_store.db"
    if not db_path.exists():
        db_path = Path("pattern_store.db")

    store = PatternStore(str(db_path))

    if cfg.list_suppressions:
        sups = store.get_fp_suppressions()
        if not sups:
            print("No FP suppressions recorded yet.")
        else:
            print(f"\n  FP Suppressions ({len(sups)} total):")
            print("  " + "-" * 64)
            for s in sups:
                print(f"  [{s['suppress_count']:>3}x]  {s['fn_name']:<30}  "
                      f"{s['vuln_type']:<20}  {s['sink_fn']}")
                if s["reason"]:
                    print(f"        reason: {s['reason']}")
        return

    if cfg.reject_fp:
        store.record_fp_rejection(
            fn_name   = cfg.reject_fp,
            vuln_type = getattr(cfg, "reject_vuln", ""),
            sink_fn   = getattr(cfg, "reject_sink", ""),
            reason    = getattr(cfg, "reject_reason", ""),
        )
        print(f"FP rejection stored: {cfg.reject_fp}")
        print(f"  vuln_type  : {getattr(cfg, 'reject_vuln', '') or '(any)'}")
        print(f"  sink_fn    : {getattr(cfg, 'reject_sink', '') or '(any)'}")
        print(f"  reason     : {getattr(cfg, 'reject_reason', '') or '(none provided)'}")
        print(f"Future runs will suppress this finding automatically.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _setup_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Binary vulnerability analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py vuln_O0 --no-llm --budget 50
  python pipeline.py firmware.exe --provider gemini --html --json
  python pipeline.py vuln_O0 --force --output-dir ./results

  # Resume from Stage 3 (Ghidra already ran, pcode_ranked.jsonl exists)
  python pipeline.py --resume-from 3 --output-dir D:/BinaryVulnDataset/results/SND001/vuln_O0 --provider openrouter --json
        """,
    )
    parser.add_argument("binary",         nargs="?", default=None,
                        help="Path to the binary to analyze (not required with --resume-from 2+)")
    parser.add_argument("--output-dir",   default=".",      metavar="PATH",  help="Output directory (default: cwd)")
    parser.add_argument("--resume-from",  type=int, default=None, metavar="N",
                        help="Resume from stage N (3=taint, 4=LLM, 5=report). "
                             "Requires pcode_ranked.jsonl in --output-dir. Skips Ghidra.")
    parser.add_argument("--no-llm",       action="store_true",               help="Disable LLM review")
    parser.add_argument("--budget",       type=int,   default=300,           help="Max ranked functions (default: 300)")
    parser.add_argument("--min-score",    type=float, default=0.15,          help="Min filter score (default: 0.15)")
    parser.add_argument("--provider",     default="groq",                    help="LLM provider: groq|openrouter|gemini|anthropic (default: groq)")
    parser.add_argument("--force",        action="store_true",               help="Re-run all stages ignoring checkpoints")
    parser.add_argument("--html",         action="store_true",               help="Write HTML report")
    parser.add_argument("--json",         action="store_true",               help="Write JSON report")
    parser.add_argument("--log-level",    default="INFO",                    help="Logging level (default: INFO)")
    # Item 5: Active Learning — FP feedback
    parser.add_argument("--reject-fp",    metavar="FN_NAME",  default=None,
                        help="Mark a finding as false positive and store suppression rule")
    parser.add_argument("--reject-vuln",  metavar="VULN_TYPE", default="",
                        help="Vuln type for --reject-fp (optional; narrows suppression scope)")
    parser.add_argument("--reject-sink",  metavar="SINK_FN",   default="",
                        help="Sink function for --reject-fp (optional)")
    parser.add_argument("--reject-reason", metavar="REASON",   default="",
                        help="Human-readable reason for the FP rejection")
    parser.add_argument("--list-suppressions", action="store_true",
                        help="Show all recorded FP suppressions and exit")

    cfg = parser.parse_args()

    # Validate: binary required unless resuming from stage 2+
    if cfg.binary is None and (cfg.resume_from is None or cfg.resume_from < 2):
        if not (cfg.reject_fp or cfg.list_suppressions):
            parser.error("binary argument is required unless --resume-from 2 (or higher) is set")

    # Handle --reject-fp and --list-suppressions before the full pipeline
    if cfg.reject_fp or cfg.list_suppressions:
        _handle_active_learning_cmd(cfg)
        sys.exit(0)

    logging.basicConfig(
        level   = getattr(logging, cfg.log_level.upper(), logging.INFO),
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    sys.exit(run_pipeline(cfg))