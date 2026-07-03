"""
stage3_evidence_fusion.py  —  Stage 3E: Evidence Fusion
=========================================================
Combines output from all four parallel analyses into one EvidenceVector
per candidate.  Stage 3E does NOT make an accept/reject decision — that
is Stage 4's job.

Input:
  - 3A seeds         (SourceRole per seeded variable)
  - 3B taint result  (TaintResult: forward_reached, taint_hops, mult_tainted)
  - 3C semantic risk (list[SemRiskCandidate]: named evidence items)
  - Backward slice   (SliceResult: did we find an external source backward?)
  - Pattern store    (PatternStore: historical TP rate for this sink)
  - func metadata    (reachability_score, semantic_role, interproc_confirmed)

Output:
  - EvidenceVector per candidate

This is the "Evidence Fusion Engine" in the user's proposed architecture:

    Forward Data Flow  ||  Backward Slice  ||  Semantic Risk  ||  Pattern Memory
                              ↓
                     Evidence Fusion Engine (Stage 3E)
                              ↓
                     EvidenceVector → Stage 4 LLM

Usage
-----
    fusion = EvidenceFusion(source_analyzer, pattern_store)
    ev     = fusion.fuse(func, taint_result, slice_results, sem_candidates, candidate)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from stage3_evidence       import EvidenceVector, SourceRole, Uncertainty
from stage3_source_analysis import SourceAnalyzer
from stage3_sink_verifier  import SinkVerifier, SinkClass

if TYPE_CHECKING:
    from taint_engine       import TaintResult, VulnCandidate
    from stage3_semantic_risk  import SemRiskCandidate
    from stage3_backward_slicer import SliceResult

log = logging.getLogger(__name__)


class EvidenceFusion:
    """
    Stage 3E: combine all Stage 3 sub-analyses into a per-candidate EvidenceVector.

    One instance is shared across all functions in analyze_all().
    Thread-safe: fuse() is stateless.
    """

    def __init__(self, source_analyzer: SourceAnalyzer, pattern_store=None):
        self._src      = source_analyzer
        self._store    = pattern_store
        self._verifier = SinkVerifier()

    def fuse(
        self,
        func:          dict,
        taint_result:  "TaintResult",
        slice_results: dict[str, "SliceResult"],   # taint_source → SliceResult
        sem_candidates:"list[SemRiskCandidate]",
        candidate:     "VulnCandidate",
        seeds:         dict[str, tuple[SourceRole, str]] | None = None,
    ) -> EvidenceVector:
        """
        Build an EvidenceVector for one VulnCandidate.

        Parameters
        ----------
        func          : function dict (has semantic_role, reachability_score, …)
        taint_result  : Stage 3B result (source_confidence, taint_hops, …)
        slice_results : backward slicer results keyed by taint_source
        sem_candidates: Stage 3C semantic risk candidates for this function
        candidate     : the VulnCandidate to build evidence for
        seeds         : Stage 3A seed map (var → (SourceRole, fn_name))
        """
        sc    = taint_result.source_confidence
        hops  = taint_result.taint_hops
        seeds = seeds or {}

        # ── Source evidence (3A) ─────────────────────────────────────────────
        src_role, src_fn, src_base_conf = self._resolve_source(
            candidate, sc, seeds
        )

        # ── Forward taint evidence (3B) ─────────────────────────────────────
        forward_reached = bool(candidate.sink_fn) and candidate.op_seq >= 0
        # Transformation count: hops to the taint_source variable
        taint_src = candidate.taint_source or ""
        transform_count = hops.get(taint_src, 0)
        # If taint_src has no hop count but taint_path has entries, use path length
        if transform_count == 0 and candidate.taint_path:
            transform_count = len(candidate.taint_path)

        path_has_mult  = taint_src in taint_result.mult_tainted_vars
        path_checked   = candidate.bounded or (taint_src in taint_result.checked_vars)

        # ── Backward slice evidence (3B+) ────────────────────────────────────
        bwd_result = slice_results.get(taint_src)
        backward_reached = False
        backward_role    = ""
        backward_depth   = 0
        if bwd_result is not None and bwd_result.max_conf > 0.0:
            backward_reached = True
            backward_role    = _conf_to_role_label(bwd_result.max_conf)
            backward_depth   = bwd_result.ops_in_slice

        # ── Semantic evidence (3C) ───────────────────────────────────────────
        sem_items = []
        sem_score = 0.0
        for sc_cand in sem_candidates:
            if sc_cand.vuln_type == candidate.vuln_type:
                sem_items = sc_cand.evidence_items
                sem_score = sc_cand.semantic_score
                break
        if not sem_items:
            # Use any semantic candidate from this function as supporting evidence
            for sc_cand in sem_candidates:
                if sc_cand.semantic_score > sem_score:
                    sem_items = sc_cand.evidence_items
                    sem_score = sc_cand.semantic_score

        # ── Behavioral Prior Analysis ─────────────────────────────────────────
        # "I have seen this behavior before — here is what happened historically."
        behavioral_sim = 0.0
        behavioral_tp  = 0.50   # neutral prior when no historical data
        if self._store is not None:
            try:
                cached = self._store.lookup(candidate.sink_fn,
                                            getattr(candidate, "arg_sizes", []) or [])
                if cached is not None:
                    behavioral_sim = cached.get("confidence", 0.0)
                    behavioral_tp  = 1.0 if cached.get("is_sink", False) else 0.0
            except Exception:
                pass

        # ── Sink class ───────────────────────────────────────────────────────
        sink_class = self._verifier.classify(candidate.sink_fn or "").value.upper()

        # ── Inter-proc & graph metadata ──────────────────────────────────────
        interproc_confirmed = bool(
            any("interproc" in (step.reason or "") for step in (taint_result.flow_steps or []))
        )
        reachability  = func.get("reachability_score", 1.0)
        semantic_role = func.get("semantic_role", "unknown")

        # ── Uncertainty ───────────────────────────────────────────────────────
        uncertainty = Uncertainty(
            unknown_source  = (src_role == SourceRole.UNKNOWN),
            unknown_call    = bool(taint_result.unknown_calls),
            missing_summary = bool(
                any("missing" in (step.reason or "").lower()
                    for step in (taint_result.flow_steps or []))
            ),
            incomplete_cfg  = False,  # would require CFG analysis; conservative default
        )

        # ── Contradiction Detection ───────────────────────────────────────────
        # Rec 8: if analyses disagree, explicitly flag it for Stage 4 to reason about.
        contradictions: list[str] = []

        n_sem_items = len(sem_items)
        if not forward_reached and n_sem_items >= 2:
            contradictions.append(
                f"Forward taint did not reach sink, but semantic analysis found "
                f"{n_sem_items} risk indicator(s) ({', '.join(i.name for i in sem_items[:3])}). "
                f"An unknown function call may be breaking the taint chain."
            )

        if src_role == SourceRole.UNKNOWN and backward_reached:
            contradictions.append(
                f"Forward source identification failed (no recognized I/O function), "
                f"but backward slice confirmed an external source ({backward_role}). "
                f"The forward path likely passes through an unmodeled function."
            )

        if path_checked and path_has_mult:
            contradictions.append(
                f"Bounds check detected in taint path alongside INT_MULT. "
                f"The check may be insufficient for all overflow cases "
                f"(e.g., integer overflow before the comparison)."
            )

        ev = EvidenceVector(
            func_name             = candidate.func_name,
            entry_addr            = candidate.entry_addr,
            vuln_type             = candidate.vuln_type,
            sink_fn               = candidate.sink_fn or "",
            sink_class            = sink_class,
            source_role           = src_role,
            source_fn             = src_fn,
            source_base_conf      = src_base_conf,
            forward_reached       = forward_reached,
            transformation_count  = transform_count,
            path_has_mult         = path_has_mult,
            path_checked          = path_checked,
            taint_path            = list(candidate.taint_path or []),
            backward_reached      = backward_reached,
            backward_role         = backward_role,
            backward_depth        = backward_depth,
            semantic_role         = semantic_role,
            semantic_items        = sem_items,
            semantic_score        = sem_score,
            behavioral_similarity = behavioral_sim,
            behavioral_prior_tp   = behavioral_tp,
            uncertainty           = uncertainty,
            contradictions        = contradictions,
            interproc_confirmed   = interproc_confirmed,
            reachability_score    = reachability,
            description           = candidate.description or "",
        )

        log.debug(
            "3E fused: %s %s src=%s fwd=%s bwd=%s sem=%.2f",
            candidate.func_name, candidate.vuln_type,
            src_role.value, forward_reached, backward_reached, sem_score,
        )
        return ev

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _resolve_source(
        self,
        candidate: "VulnCandidate",
        sc:        dict[str, float],
        seeds:     dict[str, tuple[SourceRole, str]],
    ) -> tuple[SourceRole, str, float]:
        """
        Determine the best source role + function for a candidate.

        Priority:
          1. 3A seed map — most precise (we know exactly which function)
          2. TaintResult.source_confidence — propagated SourceRole base_conf
          3. Legacy "external:" prefix in taint_source
          4. Default UNKNOWN
        """
        taint_src = candidate.taint_source or ""

        # 1. Direct seed lookup
        if taint_src in seeds:
            role, fn = seeds[taint_src]
            return role, fn, role.base_conf

        # 2. Best across taint path
        best_conf = 0.0
        for v in (candidate.taint_path or []):
            if sc.get(v, 0.0) > best_conf:
                best_conf = sc[v]
            if v in seeds:
                r, f = seeds[v]
                if r.base_conf > best_conf:
                    return r, f, r.base_conf

        # 3. Legacy external: prefix
        if taint_src.startswith("external:"):
            fn = taint_src.replace("external:", "")
            role, bare = self._src.classify_call(fn)
            if role != SourceRole.UNKNOWN:
                return role, bare, role.base_conf
            return SourceRole.LIBRARY_READER, fn, SourceRole.LIBRARY_READER.base_conf

        if best_conf > 0.0:
            # Map the propagated confidence back to a role (approximate)
            role = _conf_to_role(best_conf)
            return role, taint_src, best_conf

        return SourceRole.UNKNOWN, taint_src, 0.0


def _conf_to_role(conf: float) -> SourceRole:
    """Map a propagated confidence value back to a SourceRole (approximate)."""
    for role in (
        SourceRole.DIRECT_IO, SourceRole.NETWORK_WRAPPED,
        SourceRole.LIBRARY_READER, SourceRole.PARSER_CALLBACK,
        SourceRole.DB_INPUT, SourceRole.CLI_ARGUMENT, SourceRole.ENVIRONMENT,
    ):
        if abs(conf - role.base_conf) < 0.06:
            return role
    return SourceRole.UNKNOWN


def _conf_to_role_label(conf: float) -> str:
    return _conf_to_role(conf).label
