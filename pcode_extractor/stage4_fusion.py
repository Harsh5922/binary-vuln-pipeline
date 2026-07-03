"""
stage4_fusion.py  —  Stage 4.6: Bayesian Evidence Fusion + Consensus Engine
=============================================================================
Two-layer fusion architecture — BEF handles agreement, ConsensusEngine handles conflict.

Architecture:
  Stage 4 Assessment  +  Stage 4.5 SemanticAssessments
                ↓
    Bayesian Evidence Fusion (BEF)
    ─────────────────────────────
    Compute P(vuln | memory_signal, semantic_signal)
    Compare signal gap = |memory_score - semantic_score|
         ↓                              ↓
    gap ≤ threshold              gap > threshold
    Signals agree                Genuine conflict
    BEF decides directly         ↓
         ↓               Consensus Engine
    FinalVerdict         (LLM builds consensus
    (no LLM)             between analyst positions)
                                 ↓
                          FinalVerdict
                ↓
          list[Finding] → Stage 5

Why BEF first, ConsensusEngine only for conflicts:
  - When both analysts agree (both high or both low), BEF updates the posterior
    and emits a direct verdict. No LLM call needed.
  - Only when one says "Strong" and the other says "nothing" (or contradicts) does
    the ConsensusEngine step in to build consensus.
  - This eliminates the majority of consensus calls — only genuine disagreements
    require the additional LLM round-trip.

BEF scoring:
  memory_score  = f(hypothesis_support, exploitability, uncertainty)
                  Strong+Low-unc=0.90, Moderate=0.75, Weak=0.35, Unsupported=0.10
  semantic_score = f(confirmed_count, max_confidence)
                   No semantic signal → 0.5 (neutral prior)
                   Confirmed findings → 0.5 + max_conf × 0.45
  posterior      = Bayesian update: P(vuln|E) = P(E|vuln)×P(vuln) / P(E)
  consensus_threshold = 0.35 (|memory_score - semantic_score| > this → ConsensusEngine)

Learning Loop:
  Findings with routing=bayes_direct_confirm that are dual-confirmed increment
  PatternStore orthogonal TP counters.

Public API:
    fusion = CandidateFusion(provider="openrouter", pattern_store_path=None)
    findings = fusion.merge(assessments, semantic_assessments, func_map=func_map)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ─── Bayesian Evidence Fusion ────────────────────────────────────────────────

@dataclass
class FusionSignal:
    """
    Output of BayesianEvidenceFusion — the routing decision before ConsensusEngine.

    memory_score    — normalized confidence from Stage 4 Assessment [0, 1]
    semantic_score  — normalized confidence from Stage 4.5 assessments [0, 1]
                      0.5 = neutral (no semantic signal)
    posterior       — Bayesian updated probability of vulnerability
    routing         — what happens next:
                      "bayes_direct_confirm"  — posterior high, no consensus needed
                      "bayes_direct_reject"   — posterior low,  no consensus needed
                      "needs_consensus"       — signal gap > threshold, ConsensusEngine
    reasoning       — one-sentence explanation of the routing decision
    """
    memory_score:   float
    semantic_score: float
    posterior:      float
    routing:        str    # "bayes_direct_confirm" | "bayes_direct_reject" | "needs_consensus"
    reasoning:      str


class BayesianEvidenceFusion:
    """
    Compute a Bayesian posterior from Memory Safety + Semantic signals.

    Routes to ConsensusEngine ONLY when signals genuinely conflict.
    All agreement cases are handled here without any LLM call.

    Parameters
    ----------
    consensus_threshold : |memory_score - semantic_score| > this triggers ConsensusEngine
    confirm_threshold   : posterior must exceed this for direct_confirm
    """

    def __init__(
        self,
        consensus_threshold: float = 0.35,
        confirm_threshold:   float = 0.55,
    ):
        self.consensus_threshold = consensus_threshold
        self.confirm_threshold   = confirm_threshold

    def fuse(
        self,
        assessment:           object,   # Stage 4 Assessment
        semantic_assessments: list,     # Stage 4.5 SemanticAssessment list
    ) -> FusionSignal:
        """Compute posterior and routing decision."""
        mem_score    = self._memory_score(assessment)
        has_semantic = bool(semantic_assessments)

        if not has_semantic:
            # No semantic signal — treat as neutral prior (0.5)
            # BEF decides directly from memory alone
            sem_score = 0.5
            posterior = mem_score
            routing   = ("bayes_direct_confirm" if posterior >= self.confirm_threshold
                         else "bayes_direct_reject")
            reasoning = (
                f"No semantic signal — direct from memory_score={mem_score:.2f}. "
                f"Posterior={posterior:.2f}."
            )
            return FusionSignal(mem_score, sem_score, posterior, routing, reasoning)

        sem_score  = self._semantic_score(semantic_assessments)
        posterior  = self._bayes_update(mem_score, sem_score)
        gap        = abs(mem_score - sem_score)
        routing    = self._route(gap, posterior)
        reasoning  = (
            f"memory_score={mem_score:.2f} semantic_score={sem_score:.2f} "
            f"posterior={posterior:.2f} gap={gap:.2f} "
            f"threshold={self.consensus_threshold:.2f} → {routing}"
        )
        return FusionSignal(mem_score, sem_score, posterior, routing, reasoning)

    def _memory_score(self, assessment) -> float:
        """
        Map Assessment.hypothesis_support → [0, 1] score.
        Modulated by exploitability and uncertainty.
        """
        base = {
            "Strong":      0.90,
            "Moderate":    0.75,  # raised from 0.65: Moderate support is meaningful evidence
            "Weak":        0.35,
            "Unsupported": 0.10,
        }.get(getattr(assessment, "hypothesis_support", "Weak"), 0.35)

        # Modulate by uncertainty
        unc_obj = getattr(assessment, "uncertainty", None)
        unc_lvl = getattr(unc_obj, "overall", "Moderate") if unc_obj else "Moderate"
        unc_factor = {"Low": 1.00, "Moderate": 0.88, "High": 0.70}.get(unc_lvl, 0.88)

        # Modulate by exploitability
        ea  = getattr(assessment, "exploitability_assessment", None)
        expl = getattr(ea, "exploitability", "Low") if ea else "Low"
        expl_factor = {"High": 1.00, "Medium": 0.92, "Low": 0.80}.get(expl, 0.90)

        return min(1.0, base * unc_factor * expl_factor)

    def _semantic_score(self, semantic_assessments: list) -> float:
        """
        Map SemanticAssessment list → [0, 1] score.
        0.5 = neutral.  Confirmed findings push above 0.5.  No confirmed → below 0.5.
        """
        if not semantic_assessments:
            return 0.5

        confirmed = [sa for sa in semantic_assessments if getattr(sa, "confirmed", False)]

        if not confirmed:
            # Semantic ran but found nothing confident enough
            avg_conf = (
                sum(getattr(sa, "confidence", 0.0) for sa in semantic_assessments)
                / len(semantic_assessments)
            )
            # Partial negative signal: between 0.15 (very low) and 0.49 (borderline)
            return max(0.15, avg_conf * 0.5)

        # Has confirmed findings
        max_conf = max(getattr(sa, "confidence", 0.0) for sa in confirmed)
        # Maps conf [0.60..1.0] → score [0.77..0.95]
        return 0.50 + max_conf * 0.45

    def _bayes_update(self, mem_score: float, sem_score: float) -> float:
        """
        Bayesian update:
          Prior   P(vuln)           = mem_score
          P(E | vuln)               = sem_score
          P(E | ~vuln)              = 1 - sem_score
          P(E)                      = P(E|v)×P(v) + P(E|~v)×P(~v)
          Posterior P(vuln | E)     = P(E|v)×P(v) / P(E)
        """
        p_vuln = mem_score
        p_e_v  = sem_score
        p_e_nv = 1.0 - sem_score
        p_e    = p_e_v * p_vuln + p_e_nv * (1.0 - p_vuln)
        if p_e < 1e-9:
            return p_vuln
        return min(1.0, max(0.0, (p_e_v * p_vuln) / p_e))

    def _route(self, gap: float, posterior: float) -> str:
        if gap > self.consensus_threshold:
            return "needs_consensus"
        return ("bayes_direct_confirm" if posterior >= self.confirm_threshold
                else "bayes_direct_reject")


# ─── CandidateFusion ─────────────────────────────────────────────────────────

class CandidateFusion:
    """
    Orchestrate BEF + ConsensusEngine into a final confirmed finding list.

    Parameters
    ----------
    provider            : LLM provider for ConsensusEngine
    api_key             : reads from env if None
    consensus_model     : override LLM model for ConsensusEngine
    pattern_store_path  : path to pattern_store.db for Learning Loop
    min_semantic_conf   : minimum SemanticAssessment.confidence to pass to BEF
    use_consensus       : False = skip ConsensusEngine entirely (BEF only)
    bef_threshold       : |memory - semantic| > this routes to ConsensusEngine
    """

    def __init__(
        self,
        provider:           str           = "openrouter",
        api_key:            Optional[str] = None,
        consensus_model:    Optional[str] = None,
        pattern_store_path: Optional[str] = None,
        min_semantic_conf:  float         = 0.40,
        use_consensus:      bool          = True,
        bef_threshold:      float         = 0.35,
    ):
        self.min_semantic_conf = min_semantic_conf
        self._bef = BayesianEvidenceFusion(consensus_threshold=bef_threshold)

        # Pattern store for Learning Loop
        self._store = None
        path = pattern_store_path or os.environ.get("PATTERN_STORE_PATH", "pattern_store.db")
        try:
            from pattern_store import PatternStore
            self._store = PatternStore(path)
        except Exception as e:
            log.debug("CandidateFusion: pattern store unavailable: %s", e)

        # Consensus Engine (Stage 4.6 — only called for genuine conflicts)
        self._engine = None
        if use_consensus:
            try:
                from stage4_judge import ConsensusEngine
                self._engine = ConsensusEngine(
                    provider = provider,
                    api_key  = api_key,
                    model    = consensus_model,
                )
            except Exception as e:
                log.debug(
                    "CandidateFusion: ConsensusEngine unavailable (%s) — BEF-only mode", e
                )

    def merge(
        self,
        assessments:          list,         # list[Assessment] from Stage 4
        semantic_assessments: list,         # list[SemanticAssessment] from Stage 4.5
        func_map:             dict = None,  # func_name → func dict
    ) -> list:
        """
        Merge both tracks via BEF + ConsensusEngine.
        Returns list[Finding] compatible with Stage 5.

        Flow per function:
          1. BEF computes posterior from Memory + Semantic signals
          2. If routing = direct → emit FinalVerdict from BEF (no LLM)
          3. If routing = needs_consensus → ConsensusEngine builds consensus
        """
        func_map = func_map or {}

        # Group Stage 4.5 assessments by function
        sem_by_func: dict[str, list] = {}
        for sa in semantic_assessments:
            if getattr(sa, "confidence", 0.0) < self.min_semantic_conf:
                continue
            fn = getattr(sa, "func_name", "")
            sem_by_func.setdefault(fn, []).append(sa)

        finding_by_key: dict[tuple, object] = {}
        judged_fn_types: set[tuple] = set()

        bef_direct_n    = 0
        consensus_n     = 0
        consensus_llm_n = 0

        for assessment in assessments:
            fn      = getattr(assessment, "func_name", "")
            vtype   = getattr(assessment, "vuln_type", "")
            fn_sems = sem_by_func.get(fn, [])

            # ── BEF: compute posterior + routing ─────────────────────────────
            signal = self._bef.fuse(assessment, fn_sems)

            if signal.routing == "needs_consensus" and self._engine is not None:
                # ── Genuine conflict → ConsensusEngine ────────────────────────
                verdict  = self._engine.resolve(
                    func_name            = fn,
                    assessment           = assessment,
                    semantic_assessments = fn_sems,
                    func                 = func_map.get(fn, {}),
                )
                finding  = verdict.to_finding()
                # BEF posterior is the authoritative Bayesian confidence (uses both signals);
                # ConsensusEngine's default is just uncertainty.confidence * 1.10, too conservative
                finding.confidence = signal.posterior
                finding.calibration["bef_posterior"] = round(signal.posterior, 3)
                finding.calibration["bef_gap"]       = round(abs(signal.memory_score - signal.semantic_score), 3)
                consensus_n += 1
                if self._engine.enabled:
                    consensus_llm_n += 1
                log.debug(
                    "BEF→Consensus [%s]: %s  gap=%.2f  confirmed=%s",
                    verdict.conflict_type, fn, abs(signal.memory_score - signal.semantic_score),
                    finding.confirmed,
                )
            else:
                # ── Direct BEF verdict (no LLM) ───────────────────────────────
                finding = self._bef_verdict(assessment, fn_sems, signal, fn)
                bef_direct_n += 1

            if not finding.confirmed:
                continue

            key = (fn, vtype, getattr(assessment, "sink_fn", "") or "")
            if key not in finding_by_key or finding.confidence > finding_by_key[key].confidence:
                finding_by_key[key] = finding

            # Mark semantic assessments consumed
            for sa in fn_sems:
                judged_fn_types.add((fn, getattr(sa, "vuln_type", "")))

        # ── Stage 4.5-only findings not covered by Stage 4 ───────────────────
        for fn, sems in sem_by_func.items():
            for sa in sems:
                vtype = getattr(sa, "vuln_type", "")
                if (fn, vtype) in judged_fn_types:
                    continue
                if not getattr(sa, "confirmed", False):
                    continue
                f   = sa.to_finding()
                key = (fn, vtype, f.sink_fn or "")
                if key not in finding_by_key or f.confidence > finding_by_key[key].confidence:
                    finding_by_key[key] = f

        # ── Learning Loop ─────────────────────────────────────────────────────
        # Dual-confirmed (BEF direct confirm with semantic evidence) = highest TP signal
        for f in finding_by_key.values():
            cal = f.calibration or {}
            if cal.get("routing") == "bayes_direct_confirm" and cal.get("has_semantic"):
                self._record_learning(
                    sem_by_func.get(f.func_name, []),
                    memory_assessment=None,
                )

        # ── Final deduplication + sort ────────────────────────────────────────
        confirmed = sorted(
            finding_by_key.values(),
            key=lambda f: f.confidence,
            reverse=True,
        )

        log.info(
            "CandidateFusion: %d confirmed — %d BEF-direct, %d consensus (%d LLM)",
            len(confirmed), bef_direct_n, consensus_n, consensus_llm_n,
        )
        return confirmed

    def _bef_verdict(
        self,
        assessment:   object,
        fn_sems:      list,
        signal:       FusionSignal,
        func_name:    str,
    ) -> object:
        """Create a Finding directly from BEF signal — no LLM needed."""
        from reasoning_agent import Finding

        confirmed   = signal.routing == "bayes_direct_confirm"
        ea          = getattr(assessment, "exploitability_assessment", None)
        unc         = getattr(assessment, "uncertainty", None)
        mem_support = getattr(assessment, "hypothesis_support", "?")
        sem_bugs    = [sa.potential_bug for sa in fn_sems if getattr(sa, "confirmed", False)]

        if confirmed:
            severity = getattr(assessment, "severity", "medium")
            if sem_bugs:
                severity = _boost_severity(severity)
        else:
            severity = "low"

        reasoning = (
            f"[BEF/{signal.routing}] {signal.reasoning}. "
            f"Memory: {mem_support}. "
            + (f"Semantic confirms: {'; '.join(sem_bugs[:2])}." if sem_bugs else "No semantic signal.")
        )
        exploit = getattr(ea, "description", "") if ea else ""

        return Finding(
            func_name    = func_name,
            entry_addr   = getattr(assessment, "entry_addr", ""),
            vuln_type    = getattr(assessment, "vuln_type", "?"),
            sink_fn      = getattr(assessment, "sink_fn", "") or "",
            op_seq       = getattr(assessment, "op_seq", -1),
            taint_source = getattr(assessment, "taint_source", "") or "",
            taint_path   = list(getattr(assessment, "taint_path", []) or []),
            confirmed    = confirmed,
            severity     = severity,
            reasoning    = reasoning,
            exploit_condition     = exploit,
            false_positive_reason = "; ".join(
                getattr(assessment, "contradictory_evidence", [])[:2] or []
            ),
            confidence   = signal.posterior,
            model_used   = f"bef/{signal.routing}",
            analysis_time_s = 0.0,
            calibration  = {
                "stage":        "4.6_bef",
                "routing":      signal.routing,
                "memory_score": round(signal.memory_score, 3),
                "semantic_score": round(signal.semantic_score, 3),
                "posterior":    round(signal.posterior, 3),
                "has_semantic": bool(fn_sems),
                "mem_support":  mem_support,
                "uncertainty":  getattr(unc, "overall", "?") if unc else "?",
            },
        )

    def _record_learning(self, sem_findings: list, memory_assessment) -> None:
        """Learning Loop: BEF direct-confirm with semantic agreement = highest TP signal."""
        if self._store is None:
            return
        for sa in sem_findings:
            if not getattr(sa, "confirmed", False):
                continue
            try:
                at      = getattr(sa, "analysis_type", "unknown")
                key     = f"orthogonal_tp_{at}"
                current = float(self._store.get_metadata(key, 0.0))
                self._store.set_metadata(key, str(current + 1.0))
                log.debug(
                    "Learning: BEF dual-confirm %s → orthogonal/%s TP=%.0f",
                    getattr(sa, "func_name", "?"), at, current + 1.0,
                )
            except Exception as e:
                log.debug("Learning loop failed: %s", e)


def _boost_severity(severity: str) -> str:
    """Upgrade severity one level for dual-confirmed findings."""
    order = ["info", "low", "medium", "high", "critical"]
    idx = order.index(severity) if severity in order else 2
    return order[min(idx + 1, len(order) - 1)]
