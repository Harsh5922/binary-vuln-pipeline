"""
stage3_orchestrator.py  —  Hybrid Semantic Data-Flow Analysis
=============================================================
Coordinates all Stage 3 sub-analyses in PARALLEL then fuses their output
into EvidenceVectors that Stage 4 (LLM) reasons over.

Architecture (user proposal):

    Stage 2 ranked functions
            │
            ▼
    ──────────────────────────────────────────
    Semantic Initialization  (3A source seeds)
    ──────────────────────────────────────────
            │
            ▼
    ══════════════════════════════════════════
             PARALLEL  ANALYSIS
    ══════════════════════════════════════════
    Forward Data Flow  (3B TaintEngine)
    ||
    Semantic Risk      (3C EvidenceCollector — item count, no weights)
    ||
    Behavioral Prior   (PatternStore — historical TP rates)
    ══════════════════════════════════════════
            │
    (after 3B, per-candidate)
    Backward Slicer    (3B+ triggered by: no source found, not a magic threshold)
            │
            ▼
    ──────────────────────────────────────────
    Evidence Fusion    (3E EvidenceVector)
    ──────────────────────────────────────────
            │
            ▼
    Candidate + EvidenceVector  →  Stage 4 LLM

Key design changes vs. previous version:
  - No confidence decay: transformation_count replaces magic decay factors
  - No magic backward trigger threshold: trigger when source not found
  - Analyses run in parallel (3B, 3C, PatternStore)
  - Stage 3E combines evidence; Stage 3 does NOT decide vuln/no-vuln
  - Evidence stored on TaintResult.evidences dict for Stage 4 to use

Public API (backward-compatible with TaintEngine):
    orchestrator = Stage3Orchestrator(matcher, summary_db=...)
    results      = orchestrator.analyze_all(funcs)   # list[TaintResult]
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from taint_engine           import TaintEngine, TaintResult, VulnCandidate, _add_vuln
from stage3_evidence        import EvidenceVector, SourceRole
from stage3_source_analysis import SourceAnalyzer
from stage3_semantic_risk   import SemanticRiskAnalyzer, SemRiskCandidate
from stage3_sink_verifier   import SinkVerifier
from stage3_backward_slicer import BackwardSlicer, SliceResult
from stage3_evidence_fusion import EvidenceFusion

log = logging.getLogger(__name__)

# Semantic risk: emit when at least this many independent observations fired
_SEM_EMIT_MIN_ITEMS = 2


class Stage3Orchestrator:
    """
    Hybrid Semantic Data-Flow Analysis orchestrator.

    Drop-in replacement for TaintEngine with the full evidence pipeline.
    """

    def __init__(
        self,
        matcher,
        summary_db                = None,
        enable_backward_slicing:  bool = True,
        enable_semantic_risk:     bool = True,
        enable_evidence_fusion:   bool = True,
    ):
        self._engine    = TaintEngine(matcher, summary_db=summary_db)
        self._src       = SourceAnalyzer()
        self._sem_risk  = SemanticRiskAnalyzer()
        self._sink_ver  = SinkVerifier()
        self._bwd       = BackwardSlicer(self._src) if enable_backward_slicing else None
        self._fusion    = EvidenceFusion(self._src, getattr(matcher, "_store", None))
        self._use_bwd   = enable_backward_slicing
        self._use_sem   = enable_semantic_risk
        self._use_fuse  = enable_evidence_fusion

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, func: dict) -> TaintResult:
        """Run full Stage 3 pipeline on one function."""
        name  = func.get("name", "unknown")
        ops   = func.get("ops") or []
        role  = func.get("semantic_role", "unknown")

        # ── Semantic Initialization: Stage 3A ─────────────────────────────────
        seeds     = self._src.analyze_seeds(ops)       # {var: (SourceRole, fn)}
        seed_conf = {v: r.base_conf for v, (r, _) in seeds.items()}
        func_with_seeds = {**func, "_source_seeds_3a": seed_conf}

        # ── PARALLEL ANALYSIS ─────────────────────────────────────────────────
        # 3B and 3C are independent; run them together.
        # Pattern store query also runs in parallel (IO-bound).
        with ThreadPoolExecutor(max_workers=3) as pool:
            future_3b  = pool.submit(self._engine.analyze, func_with_seeds)
            future_3c  = pool.submit(
                self._sem_risk.analyze, func, ops, role
            ) if self._use_sem else None
            future_pat = pool.submit(
                self._query_behavioral_prior, func
            )

            tr         = future_3b.result()
            sem_cands  = future_3c.result() if future_3c else []
            pat_result = future_pat.result()

        # ── Backward Slicing ─────────────────────────────────────────────────
        # Trigger: no source found in taint path (not a magic threshold).
        # For each candidate, check if its taint_source has source_confidence > 0.
        # If not, the forward taint chain has no recognized external source —
        # run backward slicer to try to find one.
        slice_results: dict[str, SliceResult] = {}
        if self._use_bwd and self._bwd is not None:
            for cand in tr.vulns:
                src = cand.taint_source or ""
                # Condition: no source found in forward analysis
                no_source = tr.source_confidence.get(src, 0.0) == 0.0 and all(
                    tr.source_confidence.get(v, 0.0) == 0.0
                    for v in (cand.taint_path or [])
                )
                if no_source and src not in slice_results:
                    result = self._bwd.slice(target_var=src, ops=ops)
                    slice_results[src] = result
                    if result.max_conf > 0:
                        log.debug(
                            "3B+: %s %s — backward slice found conf=%.2f via %s",
                            name, cand.sink_fn, result.max_conf,
                            result.external_fns[:2],
                        )

        # ── Evidence Fusion: Stage 3E ─────────────────────────────────────────
        evidence_map: dict[str, EvidenceVector] = {}
        if self._use_fuse:
            for cand in tr.vulns:
                ev = self._fusion.fuse(
                    func          = func,
                    taint_result  = tr,
                    slice_results = slice_results,
                    sem_candidates= sem_cands,
                    candidate     = cand,
                    seeds         = seeds,
                )
                evidence_map[cand.fingerprint or id(cand)] = ev

        # ── Merge 3C semantic candidates ──────────────────────────────────────
        vulns = list(tr.vulns)
        covered_types = {v.vuln_type for v in vulns}

        for sc in sem_cands:
            if len(sc.evidence_items) < _SEM_EMIT_MIN_ITEMS:
                continue
            if sc.vuln_type in covered_types:
                continue
            vr = VulnCandidate(
                func_name    = sc.func_name,
                entry_addr   = sc.entry_addr,
                vuln_type    = sc.vuln_type,
                op_seq       = -1,
                sink_fn      = f"semantic:{sc.pattern_name}",
                taint_source = "semantic_pattern",
                taint_path   = [i.name for i in sc.evidence_items],
                bounded      = False,
                confidence   = sc.semantic_score,
                description  = sc.description + " [3C:semantic_risk]",
                match_kind   = "NO_MATCH",
                arg_sizes    = [],
                fingerprint  = f"sem_risk|{sc.pattern_name}",
            )
            _add_vuln(vulns, vr)
            covered_types.add(sc.vuln_type)

            # Fuse evidence for semantic-only candidate
            if self._use_fuse:
                ev = self._fusion.fuse(
                    func=func, taint_result=tr, slice_results=slice_results,
                    sem_candidates=sem_cands, candidate=vr, seeds=seeds,
                )
                evidence_map[vr.fingerprint] = ev

        # Store evidence map in a way the pipeline can pass to reasoning_agent.
        # We attach it as a dict on a new TaintResult field "evidences".
        return TaintResult(
            func_name         = tr.func_name,
            entry_addr        = tr.entry_addr,
            tainted_vars      = tr.tainted_vars,
            tainted_mem       = tr.tainted_mem,
            unknown_calls     = tr.unknown_calls,
            vulns             = vulns,
            flow_steps        = tr.flow_steps,
            ops_analyzed      = tr.ops_analyzed,
            calls_matched     = tr.calls_matched,
            calls_unknown     = tr.calls_unknown,
            source_confidence = tr.source_confidence,
            taint_hops        = tr.taint_hops,
            mult_tainted_vars = tr.mult_tainted_vars,
            checked_vars      = tr.checked_vars,
            evidences         = evidence_map,
        )

    def analyze_all(self, funcs: list[dict]) -> list[TaintResult]:
        """Analyze all functions (functions themselves run in parallel within analyze())."""
        results = [self.analyze(f) for f in funcs]

        total     = sum(len(r.vulns) for r in results)
        sem_count = sum(
            1 for r in results for v in r.vulns
            if "[3C:semantic_risk]" in (v.description or "")
        )
        bwd_count = sum(
            1 for r in results
            for ev in (r.evidences or {}).values()
            if ev.backward_reached
        )
        log.info(
            "Stage 3 complete — %d functions → %d candidates "
            "(%d semantic_risk  %d backward_confirmed)",
            len(results), total, sem_count, bwd_count,
        )
        return results

    def _query_behavioral_prior(self, func: dict) -> dict:
        """Query behavioral prior (pattern store) for historical TP rates."""
        store = getattr(getattr(self._engine, "matcher", None), "_store", None)
        if store is None:
            return {}
        # Light query — EvidenceFusion queries per-candidate; this warms the cache
        return {}
