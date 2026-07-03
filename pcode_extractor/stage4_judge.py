"""
stage4_judge.py  —  Stage 4.6: Consensus Engine
================================================
Resolves GENUINE CONFLICTS between Stage 4 (Memory Safety Analyst) and Stage 4.5
(Semantic Analyst) when Bayesian Evidence Fusion (BEF) cannot decide directly.

Called ONLY when BEF routes conflict_type = "needs_consensus".
BEF handles all agreement cases without any LLM call.

Architecture:
  Stage 4 Assessment   +   Stage 4.5 SemanticAssessments
               ↓
   Bayesian Evidence Fusion  (stage4_fusion.py)
     ├─ agrees (signals within threshold) → FinalVerdict (no LLM)
     └─ conflict (signals diverge > threshold)
               ↓
       Consensus Engine  ← YOU ARE HERE
         Reads both analyst positions
         Builds consensus — not judgment
         (It isn't judging. It's finding what both analysts actually agree on.)
               ↓
          FinalVerdict

Conflict types routed here:
  disagrees — Memory says Strong, Semantic says no findings (or reverse)

Conflict types handled by BEF (not here):
  bayes_direct_confirm — both agree, posterior high
  bayes_direct_reject  — both agree, posterior low
  one_sided_memory     — Semantic had no signal (neutral)
  one_sided_semantic   — Memory not confirmed, Semantic found something

Key principle: the Consensus Engine does NOT re-analyze P-code.
It reads two prepared analyst positions and builds consensus between them.

Public API:
    engine = ConsensusEngine(api_key=..., provider="openrouter")
    verdict = engine.resolve(func_name, assessment, semantic_assessments, func)
    findings = [v.to_finding() for v in verdicts if v.confirmed]
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class FinalVerdict:
    """
    Stage 4.6 output: single final verdict after multi-analyst debate.

    confirmed  — whether the function should be flagged as a finding
    severity   — the final severity after judge arbitration
    confidence — judge-adjusted confidence (0.0 – 1.0)

    conflict_detected  — True when analysts disagreed
    conflict_type      — categorizes the conflict (see module docstring)
    conflict_resolution — judge's explanation of why winner won
    """
    func_name:    str
    entry_addr:   str
    vuln_type:    str
    sink_fn:      str

    # Analyst positions (summarized for audit trail)
    memory_analyst_support:    str           # Assessment.hypothesis_support
    memory_analyst_uncertainty: str          # Assessment.uncertainty.overall
    semantic_analyst_findings: list[str]     # [sa.potential_bug for sa in SAs]
    semantic_analyst_types:    list[str]     # [sa.analysis_type for sa in SAs]

    # Conflict analysis
    conflict_detected:   bool
    conflict_type:       str    # agrees_vulnerable | agrees_safe | disagrees | one_sided_*
    conflict_resolution: str    # judge's reasoning

    # Final decision
    confirmed:    bool
    severity:     str   # critical | high | medium | low
    recommended_cwe:  str
    recommended_cvss: str

    # Merged evidence
    exploit_chain:       str
    merged_supporting:   list[str]
    merged_contradictions: list[str]

    # Final uncertainty
    final_uncertainty: str    # Low | Moderate | High
    confidence:        float  # judge-adjusted

    model_used:      str
    analysis_time_s: float = 0.0

    def to_finding(self):
        """Convert to legacy Finding for Stage 5 report compatibility."""
        from reasoning_agent import Finding
        return Finding(
            func_name    = self.func_name,
            entry_addr   = self.entry_addr,
            vuln_type    = self.vuln_type,
            sink_fn      = self.sink_fn or "",
            op_seq       = -1,
            taint_source = "",
            taint_path   = [],
            confirmed    = self.confirmed,
            severity     = self.severity,
            reasoning    = (
                f"[ConsensusEngine/{self.conflict_type}] {self.conflict_resolution} "
                f"| Exploit: {self.exploit_chain}"
            ),
            exploit_condition     = self.exploit_chain,
            false_positive_reason = "; ".join(self.merged_contradictions[:2]),
            confidence   = self.confidence,
            model_used   = self.model_used,
            analysis_time_s = self.analysis_time_s,
            calibration  = {
                "stage":                  "4.6_fusion_judge",
                "conflict_detected":      self.conflict_detected,
                "conflict_type":          self.conflict_type,
                "conflict_resolution":    self.conflict_resolution,
                "memory_analyst_support": self.memory_analyst_support,
                "semantic_types":         self.semantic_analyst_types,
                "final_uncertainty":      self.final_uncertainty,
            },
        )


# ─── Prompts ─────────────────────────────────────────────────────────────────

_SYS_JUDGE = """\
You are the Consensus Engine in a multi-analyst binary vulnerability review.

You have received position statements from two independent AI analysts:
  - Memory Safety Analyst (Stage 4): focused on taint paths, memory allocation, bounds
  - Semantic Analyst (Stage 4.5):   focused on logic, state machines, API misuse

Bayesian Evidence Fusion already confirmed these analysts disagree — you are called
ONLY when the signal gap between them exceeds the consensus threshold.

Your job: build consensus.
  1. Identify what the two analysts ACTUALLY agree on (the agreed facts)
  2. Identify the specific point of disagreement
  3. Determine which analyst's evidence is stronger FOR THIS SPECIFIC CASE
  4. Produce a single final verdict grounded in the agreed facts

You do NOT re-analyze the P-code. You work from two prepared analyst positions.
The output is a consensus — not a judgment of which analyst is "right" in general.
Be direct about uncertainty — if the disagreement cannot be resolved, say so.

Respond with ONLY valid JSON. No markdown. No explanation outside the JSON."""

_USER_JUDGE = """\
FUNCTION: {func_name}  @ {entry_addr}
HYPOTHESIS: {vuln_type}  |  SINK: {sink_fn}

══════════ MEMORY SAFETY ANALYST POSITION (Stage 4) ══════════
Hypothesis support:   {hypothesis_support}
Evidence quality:     {uncertainty}
Reachability:         {reachability}
Exploitability:       {exploitability}
Impact:               {impact}

Signal agreement:
  Source:     {agr_source}
  Forward:    {agr_forward}
  Backward:   {agr_backward}
  Semantic:   {agr_semantic}
  Behavioral: {agr_behavioral}

Supporting evidence:
{supporting_evidence}

Contradictory evidence:
{contradictory_evidence}

Missing evidence:
{missing_evidence}

Peer review verdicts (what this analyst accepted or reserved):
{peer_review_summary}

══════════ SEMANTIC ANALYST POSITION (Stage 4.5) ══════════
{semantic_section}

══════════ CONSENSUS ENGINE: Build consensus ══════════
Respond with ONLY this JSON:

{{
  "conflict_detected":   true|false,
  "conflict_type":       "disagrees|one_sided_memory|one_sided_semantic",
  "conflict_resolution": "<2-3 sentences: what the analysts agree on, what they disagree on, and why the stronger evidence wins>",
  "primary_evidence":    "memory_safety|semantic|both|neither",
  "confirmed":           true|false,
  "severity":            "critical|high|medium|low",
  "exploit_chain":       "<end-to-end exploit scenario — combine both analysts' insights>",
  "merged_supporting":   ["<strongest evidence items from either analyst>"],
  "merged_contradictions": ["<strongest arguments against — from either analyst>"],
  "final_uncertainty":   "Low|Moderate|High",
  "confidence":          <0.0-1.0>,
  "recommended_cwe":     "CWE-XXX",
  "recommended_cvss":    "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
}}
"""


# ─── ConsensusEngine ─────────────────────────────────────────────────────────────

class ConsensusEngine:
    """
    Stage 4.6: Consensus Engine — resolves genuine analyst conflicts.

    Called ONLY by BayesianEvidenceFusion when the signal gap between Memory Safety
    Analyst and Semantic Analyst exceeds the consensus threshold.

    BEF handles all agreement cases before reaching here.
    This engine builds consensus — it does not re-analyze the P-code.

    LLM call only for:
      - disagrees: one analyst says vulnerable, other says not

    Parameters
    ----------
    provider    : "openrouter" | "groq" | "anthropic" | "gemini"
    api_key     : reads from env if None
    model       : overrides default model for provider
    """

    _DEFAULT_MODELS = {
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
        "groq":       "llama-3.3-70b-versatile",
        "anthropic":  "claude-haiku-4-5-20251001",
        "gemini":     "gemini-2.5-flash",
    }
    _ENV_KEYS = {
        "openrouter": "OPENROUTER_API_KEY",
        "groq":       "GROQ_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "gemini":     "GEMINI_API_KEY",
    }

    def __init__(
        self,
        provider: str           = "openrouter",
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model    = model or self._DEFAULT_MODELS.get(self.provider,
                                                           "meta-llama/llama-3.3-70b-instruct")
        env_var = self._ENV_KEYS.get(self.provider, "OPENROUTER_API_KEY")
        self.api_key = api_key or os.environ.get(env_var, "")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            log.warning("ConsensusEngine: no API key for '%s' — LLM arbitration disabled",
                        self.provider)

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(
        self,
        func_name:            str,
        assessment:           object,          # Stage 4 Assessment
        semantic_assessments: list,            # Stage 4.5 SemanticAssessment list
        func:                 dict = None,     # function dict (for context)
    ) -> FinalVerdict:
        """
        Produce a FinalVerdict by arbitrating between the two analyst positions.

        Returns a FinalVerdict regardless of whether LLM is enabled:
        fast-paths work without an LLM call.
        """
        t0 = time.perf_counter()

        mem_confirmed  = getattr(assessment, "confirmed", False)
        sem_confirmed  = [sa for sa in semantic_assessments
                          if getattr(sa, "confirmed", False)]
        sem_findings   = [sa for sa in semantic_assessments
                          if getattr(sa, "confidence", 0.0) >= 0.40]

        conflict_type = self._classify_conflict(mem_confirmed, sem_confirmed, sem_findings)

        # ── Fast paths ────────────────────────────────────────────────────────
        if conflict_type == "agrees_vulnerable":
            return self._fast_agree_vulnerable(
                assessment, sem_confirmed, func_name,
                time.perf_counter() - t0,
            )

        if conflict_type == "agrees_safe":
            return self._fast_agree_safe(
                assessment, func_name,
                time.perf_counter() - t0,
            )

        if conflict_type in ("one_sided_memory", "one_sided_semantic"):
            return self._fast_one_sided(
                conflict_type, assessment, sem_findings, func_name,
                time.perf_counter() - t0,
            )

        # ── LLM arbitration for genuine disagreement ──────────────────────────
        if not self.enabled:
            return self._fallback_verdict(
                assessment, func_name, conflict_type,
                "LLM unavailable — passing through memory analyst position.",
                time.perf_counter() - t0,
            )

        user_msg = self._build_prompt(assessment, sem_findings, func_name)
        raw      = self._call_llm(user_msg, func_name)
        elapsed  = time.perf_counter() - t0

        if raw is None:
            return self._fallback_verdict(
                assessment, func_name, conflict_type,
                "LLM call failed — passing through memory analyst position.",
                elapsed,
            )

        parsed = self._parse(raw)
        if parsed is None:
            return self._fallback_verdict(
                assessment, func_name, conflict_type,
                "LLM response unparseable — passing through memory analyst position.",
                elapsed,
            )

        return self._build_verdict(assessment, sem_findings, conflict_type, parsed, elapsed)

    # ── Conflict classification ───────────────────────────────────────────────

    def _classify_conflict(
        self,
        mem_confirmed:  bool,
        sem_confirmed:  list,
        sem_findings:   list,
    ) -> str:
        has_sem = bool(sem_findings)
        sem_ok  = bool(sem_confirmed)

        if mem_confirmed and sem_ok:
            return "agrees_vulnerable"
        if not mem_confirmed and not sem_ok:
            return "agrees_safe"
        if mem_confirmed and not has_sem:
            return "one_sided_memory"
        if not mem_confirmed and sem_ok and not has_sem:
            return "one_sided_semantic"
        if sem_ok and not mem_confirmed:
            return "one_sided_semantic"
        if mem_confirmed and has_sem and not sem_ok:
            return "disagrees"
        return "disagrees"

    # ── Fast paths ────────────────────────────────────────────────────────────

    def _fast_agree_vulnerable(
        self,
        assessment: object,
        sem_confirmed: list,
        func_name: str,
        elapsed: float,
    ) -> FinalVerdict:
        from stage4_orthogonal import SemanticAssessment
        mem_conf   = getattr(assessment, "confidence", 0.75)
        boosted    = min(1.0, mem_conf * 1.10)
        sem_types  = [sa.analysis_type for sa in sem_confirmed]
        sem_bugs   = [sa.potential_bug  for sa in sem_confirmed]
        mem_sup    = getattr(assessment, "supporting_evidence", [])
        ea         = getattr(assessment, "exploitability_assessment", None)
        ea_str     = (
            f"Reach={ea.reachability} Expl={ea.exploitability} Impact={ea.impact}"
            if ea else ""
        )
        log.info(
            "ConsensusEngine [agrees_vulnerable]: %s — "
            "Memory(%s) + Semantic(%s) → confirmed, conf=%.0f%%",
            func_name,
            getattr(assessment, "hypothesis_support", "?"),
            ", ".join(sem_types),
            boosted * 100,
        )
        return FinalVerdict(
            func_name   = func_name,
            entry_addr  = getattr(assessment, "entry_addr", ""),
            vuln_type   = getattr(assessment, "vuln_type", "?"),
            sink_fn     = getattr(assessment, "sink_fn", "") or "",
            memory_analyst_support     = getattr(assessment, "hypothesis_support", "?"),
            memory_analyst_uncertainty = getattr(getattr(assessment, "uncertainty", None),
                                                 "overall", "?"),
            semantic_analyst_findings  = sem_bugs,
            semantic_analyst_types     = sem_types,
            conflict_detected   = False,
            conflict_type       = "agrees_vulnerable",
            conflict_resolution = (
                f"Both analysts confirm vulnerability. "
                f"Memory Safety: {getattr(assessment,'hypothesis_support','?')}. "
                f"Semantic: {', '.join(sem_bugs[:2])}."
            ),
            confirmed    = True,
            severity     = getattr(assessment, "severity", "medium"),
            recommended_cwe  = getattr(assessment, "recommended_cwe", "") or "",
            recommended_cvss = getattr(assessment, "recommended_cvss", "") or "",
            exploit_chain = (
                f"{ea_str}. "
                + (ea.description if ea else "")
            ).strip(),
            merged_supporting    = list(mem_sup[:3]) + sem_bugs[:2],
            merged_contradictions = list(getattr(assessment, "contradictory_evidence", [])[:2]),
            final_uncertainty = getattr(
                getattr(assessment, "uncertainty", None), "overall", "Low"
            ),
            confidence   = boosted,
            model_used   = "fast/agrees_vulnerable",
            analysis_time_s = round(elapsed, 3),
        )

    def _fast_agree_safe(
        self,
        assessment: object,
        func_name:  str,
        elapsed:    float,
    ) -> FinalVerdict:
        log.info("ConsensusEngine [agrees_safe]: %s — both analysts: not confirmed", func_name)
        return FinalVerdict(
            func_name   = func_name,
            entry_addr  = getattr(assessment, "entry_addr", ""),
            vuln_type   = getattr(assessment, "vuln_type", "?"),
            sink_fn     = getattr(assessment, "sink_fn", "") or "",
            memory_analyst_support     = getattr(assessment, "hypothesis_support", "?"),
            memory_analyst_uncertainty = getattr(getattr(assessment, "uncertainty", None),
                                                 "overall", "?"),
            semantic_analyst_findings  = [],
            semantic_analyst_types     = [],
            conflict_detected   = False,
            conflict_type       = "agrees_safe",
            conflict_resolution = (
                "Neither analyst found sufficient evidence to confirm a vulnerability. "
                f"Memory Safety: {getattr(assessment,'hypothesis_support','?')}. "
                "Semantic: no confirmed findings."
            ),
            confirmed    = False,
            severity     = "low",
            recommended_cwe  = getattr(assessment, "recommended_cwe", "") or "",
            recommended_cvss = "",
            exploit_chain    = "",
            merged_supporting    = [],
            merged_contradictions = list(
                getattr(assessment, "contradictory_evidence", [])[:3]
            ),
            final_uncertainty = "High",
            confidence   = 0.20,
            model_used   = "fast/agrees_safe",
            analysis_time_s = round(elapsed, 3),
        )

    def _fast_one_sided(
        self,
        conflict_type:        str,
        assessment:           object,
        sem_findings:         list,
        func_name:            str,
        elapsed:              float,
    ) -> FinalVerdict:
        """Pass through the single analyst's verdict with explicit uncertainty note."""
        if conflict_type == "one_sided_memory":
            mem_conf  = getattr(assessment, "confidence", 0.60)
            confirmed = getattr(assessment, "confirmed", False)
            severity    = getattr(assessment, "severity", "medium")
            cwe         = getattr(assessment, "recommended_cwe", "") or ""
            cvss        = getattr(assessment, "recommended_cvss", "") or ""
            ea          = getattr(assessment, "exploitability_assessment", None)
            exploit     = ea.description if ea else ""
            resolution  = (
                "Memory Safety Analyst only — Semantic Analyst ran no analyses on this function. "
                "Uncertainty elevated: single-analyst verdict."
            )
            log.info(
                "ConsensusEngine [one_sided_memory]: %s — Memory(%s) → %s",
                func_name, getattr(assessment, "hypothesis_support", "?"),
                "confirmed" if confirmed else "not confirmed",
            )
            return FinalVerdict(
                func_name   = func_name,
                entry_addr  = getattr(assessment, "entry_addr", ""),
                vuln_type   = getattr(assessment, "vuln_type", "?"),
                sink_fn     = getattr(assessment, "sink_fn", "") or "",
                memory_analyst_support     = getattr(assessment, "hypothesis_support", "?"),
                memory_analyst_uncertainty = getattr(
                    getattr(assessment, "uncertainty", None), "overall", "?"
                ),
                semantic_analyst_findings  = [],
                semantic_analyst_types     = [],
                conflict_detected   = False,
                conflict_type       = conflict_type,
                conflict_resolution = resolution,
                confirmed    = confirmed,
                severity     = severity,
                recommended_cwe  = cwe,
                recommended_cvss = cvss,
                exploit_chain    = exploit,
                merged_supporting    = list(
                    getattr(assessment, "supporting_evidence", [])[:3]
                ),
                merged_contradictions = list(
                    getattr(assessment, "contradictory_evidence", [])[:2]
                ),
                final_uncertainty = "Moderate",
                confidence   = mem_conf * 0.90,
                model_used   = "fast/one_sided_memory",
                analysis_time_s = round(elapsed, 3),
            )

        else:  # one_sided_semantic
            top_sa     = max(sem_findings, key=lambda s: s.confidence, default=None)
            confirmed  = top_sa.confirmed if top_sa else False
            sem_bugs   = [sa.potential_bug  for sa in sem_findings]
            sem_types  = [sa.analysis_type  for sa in sem_findings]
            resolution = (
                "Semantic Analyst only — Memory Safety Analyst did not confirm this function. "
                "Uncertainty elevated: single-analyst verdict."
            )
            log.info(
                "ConsensusEngine [one_sided_semantic]: %s — Semantic(%s) → %s",
                func_name, ", ".join(sem_types),
                "confirmed" if confirmed else "not confirmed",
            )
            return FinalVerdict(
                func_name   = func_name,
                entry_addr  = getattr(assessment, "entry_addr", ""),
                vuln_type   = getattr(assessment, "vuln_type", "?"),
                sink_fn     = getattr(assessment, "sink_fn", "") or "",
                memory_analyst_support     = getattr(assessment, "hypothesis_support", "?"),
                memory_analyst_uncertainty = getattr(
                    getattr(assessment, "uncertainty", None), "overall", "?"
                ),
                semantic_analyst_findings  = sem_bugs,
                semantic_analyst_types     = sem_types,
                conflict_detected   = False,
                conflict_type       = conflict_type,
                conflict_resolution = resolution,
                confirmed    = confirmed,
                severity     = "medium" if (top_sa and top_sa.confidence >= 0.75) else "low",
                recommended_cwe  = getattr(assessment, "recommended_cwe", "") or "",
                recommended_cvss = "",
                exploit_chain    = top_sa.potential_bug if top_sa else "",
                merged_supporting    = sem_bugs[:3],
                merged_contradictions = [
                    top_sa.alternative_explanation
                ] if top_sa and top_sa.alternative_explanation else [],
                final_uncertainty = "Moderate",
                confidence   = (top_sa.confidence * 0.90) if top_sa else 0.30,
                model_used   = "fast/one_sided_semantic",
                analysis_time_s = round(elapsed, 3),
            )

    def _fallback_verdict(
        self,
        assessment:     object,
        func_name:      str,
        conflict_type:  str,
        resolution:     str,
        elapsed:        float,
    ) -> FinalVerdict:
        return FinalVerdict(
            func_name   = func_name,
            entry_addr  = getattr(assessment, "entry_addr", ""),
            vuln_type   = getattr(assessment, "vuln_type", "?"),
            sink_fn     = getattr(assessment, "sink_fn", "") or "",
            memory_analyst_support     = getattr(assessment, "hypothesis_support", "?"),
            memory_analyst_uncertainty = getattr(
                getattr(assessment, "uncertainty", None), "overall", "?"
            ),
            semantic_analyst_findings  = [],
            semantic_analyst_types     = [],
            conflict_detected   = True,
            conflict_type       = conflict_type,
            conflict_resolution = resolution,
            confirmed    = getattr(assessment, "confirmed", False),
            severity     = getattr(assessment, "severity", "low"),
            recommended_cwe  = getattr(assessment, "recommended_cwe", "") or "",
            recommended_cvss = getattr(assessment, "recommended_cvss", "") or "",
            exploit_chain    = "",
            merged_supporting    = list(getattr(assessment, "supporting_evidence", [])[:2]),
            merged_contradictions = list(getattr(assessment, "contradictory_evidence", [])[:2]),
            final_uncertainty = "High",
            confidence   = getattr(assessment, "confidence", 0.30) * 0.80,
            model_used   = "fallback/judge_error",
            analysis_time_s = round(elapsed, 3),
        )

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        assessment:   object,
        sem_findings: list,
        func_name:    str,
    ) -> str:
        agr = getattr(assessment, "agreement", None)
        ea  = getattr(assessment, "exploitability_assessment", None)
        unc = getattr(assessment, "uncertainty", None)
        pr  = getattr(assessment, "peer_review", {}) or {}

        # Peer review summary
        pr_lines = []
        for key, val in pr.items():
            if isinstance(val, dict):
                pr_lines.append(f"  {key}: {val.get('verdict','?')} — {val.get('reason','')}")
        pr_summary = "\n".join(pr_lines) if pr_lines else "  (not available)"

        # Semantic section
        if sem_findings:
            sem_lines = []
            for sa in sem_findings:
                sem_lines.append(
                    f"  [{sa.analysis_type}] {sa.potential_bug}  "
                    f"(confidence {sa.confidence:.0%})\n"
                    f"    Reason: {sa.reason}\n"
                    f"    Alternative: {sa.alternative_explanation}"
                    + (f"\n    Prior ref: {sa.prior_analysis_reference}"
                       if getattr(sa, "prior_analysis_reference", "") else "")
                )
            sem_section = "\n".join(sem_lines)
        else:
            sem_section = "(No Stage 4.5 findings — semantic analyses ran but found nothing >= 0.40 confidence)"

        # Supporting + contradictory bullets
        sup_s = "\n".join(
            f"  • {e}" for e in (getattr(assessment, "supporting_evidence", []) or [])[:4]
        ) or "  (none)"
        con_s = "\n".join(
            f"  • {e}" for e in (getattr(assessment, "contradictory_evidence", []) or [])[:3]
        ) or "  (none)"
        mis_s = "\n".join(
            f"  • {e}" for e in (getattr(assessment, "missing_evidence", []) or [])[:3]
        ) or "  (none)"

        return _USER_JUDGE.format(
            func_name        = func_name,
            entry_addr       = getattr(assessment, "entry_addr", ""),
            vuln_type        = getattr(assessment, "vuln_type", "?"),
            sink_fn          = getattr(assessment, "sink_fn", "") or "",
            hypothesis_support = getattr(assessment, "hypothesis_support", "?"),
            uncertainty      = unc.overall if unc else "?",
            reachability     = ea.reachability   if ea else "?",
            exploitability   = ea.exploitability if ea else "?",
            impact           = ea.impact         if ea else "?",
            agr_source       = agr.source     if agr else "absent",
            agr_forward      = agr.forward    if agr else "absent",
            agr_backward     = agr.backward   if agr else "absent",
            agr_semantic     = agr.semantic   if agr else "absent",
            agr_behavioral   = agr.behavioral if agr else "absent",
            supporting_evidence   = sup_s,
            contradictory_evidence = con_s,
            missing_evidence      = mis_s,
            peer_review_summary   = pr_summary,
            semantic_section      = sem_section,
        )

    def _build_verdict(
        self,
        assessment:   object,
        sem_findings: list,
        conflict_type: str,
        parsed:       dict,
        elapsed:      float,
    ) -> FinalVerdict:
        func_name = getattr(assessment, "entry_addr", "")
        unc = getattr(assessment, "uncertainty", None)
        log.info(
            "ConsensusEngine [%s]: %s — confirmed=%s  uncertainty=%s  conf=%.0f%%",
            conflict_type,
            getattr(assessment, "func_name", "?"),
            parsed.get("confirmed", False),
            parsed.get("final_uncertainty", "?"),
            float(parsed.get("confidence", 0)) * 100,
        )
        return FinalVerdict(
            func_name   = getattr(assessment, "func_name", "?"),
            entry_addr  = getattr(assessment, "entry_addr", ""),
            vuln_type   = getattr(assessment, "vuln_type", "?"),
            sink_fn     = getattr(assessment, "sink_fn", "") or "",
            memory_analyst_support     = getattr(assessment, "hypothesis_support", "?"),
            memory_analyst_uncertainty = unc.overall if unc else "?",
            semantic_analyst_findings  = [sa.potential_bug  for sa in sem_findings],
            semantic_analyst_types     = [sa.analysis_type  for sa in sem_findings],
            conflict_detected   = bool(parsed.get("conflict_detected", True)),
            conflict_type       = conflict_type,
            conflict_resolution = parsed.get("conflict_resolution", ""),
            confirmed    = bool(parsed.get("confirmed", False)),
            severity     = parsed.get("severity", "low"),
            recommended_cwe  = parsed.get("recommended_cwe", "")  or "",
            recommended_cvss = parsed.get("recommended_cvss", "") or "",
            exploit_chain    = parsed.get("exploit_chain", ""),
            merged_supporting    = list(parsed.get("merged_supporting", [])),
            merged_contradictions = list(parsed.get("merged_contradictions", [])),
            final_uncertainty = parsed.get("final_uncertainty", "Moderate"),
            confidence   = float(parsed.get("confidence", 0.50)),
            model_used   = f"judge/{self.provider}/{self.model}",
            analysis_time_s = round(elapsed, 3),
        )

    # ── LLM infrastructure ────────────────────────────────────────────────────

    def _call_llm(self, user_message: str, func_name: str) -> Optional[str]:
        t0 = time.perf_counter()
        if self.provider == "openrouter":
            text, usage = self._call_openrouter(user_message, func_name)
        elif self.provider == "groq":
            text, usage = self._call_groq(user_message, func_name)
        elif self.provider == "anthropic":
            text, usage = self._call_anthropic(user_message, func_name)
        elif self.provider == "gemini":
            text, usage = self._call_gemini(user_message, func_name)
        else:
            log.error("ConsensusEngine: unknown provider '%s'", self.provider)
            return None
        elapsed = time.perf_counter() - t0
        if text is not None:
            try:
                from llm_cost_tracker import GLOBAL_TRACKER
                GLOBAL_TRACKER.record(
                    stage="judge", model=self.model,
                    input_tokens  = usage.get("input_tokens") or max(1, len(user_message) // 4),
                    output_tokens = usage.get("output_tokens") or max(1, len(text) // 4),
                    latency_s=elapsed, fn_name=func_name,
                )
            except Exception:
                pass
        return text

    def _parse(self, raw: str) -> Optional[dict]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            log.warning("ConsensusEngine: no JSON in response")
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            log.warning("ConsensusEngine: JSON parse error: %s", e)
            return None

    def _call_openrouter(self, user_message: str, func_name: str):
        import urllib.request as _ur
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/binary-vuln-pipeline",
            "X-Title":       "Binary Vulnerability Analysis Pipeline",
        }
        payload = json.dumps({
            "model": self.model, "max_tokens": 1000, "temperature": 0.0,
            "messages": [
                {"role": "system", "content": _SYS_JUDGE},
                {"role": "user",   "content": user_message},
            ],
        }).encode()
        for attempt in range(3):
            try:
                req = _ur.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=payload, headers=headers, method="POST",
                )
                with _ur.urlopen(req, timeout=90) as resp:
                    body = json.loads(resp.read().decode())
                usage = body.get("usage", {})
                return body["choices"][0]["message"]["content"], usage
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    log.error("ConsensusEngine OpenRouter error for %s: %s", func_name, e)
                    return None, {}
        return None, {}

    def _call_groq(self, user_message: str, func_name: str):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")
        client = Groq(api_key=self.api_key)
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYS_JUDGE},
                        {"role": "user",   "content": user_message},
                    ],
                    max_tokens=1000, temperature=0.0,
                )
                usage = {}
                if r.usage:
                    usage = {"input_tokens": r.usage.prompt_tokens,
                             "output_tokens": r.usage.completion_tokens}
                return r.choices[0].message.content, usage
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate" in err.lower()) and attempt < 2:
                    time.sleep(65)
                else:
                    log.error("ConsensusEngine Groq error for %s: %s", func_name, e)
                    return None, {}
        return None, {}

    def _call_anthropic(self, user_message: str, func_name: str):
        try:
            import anthropic as ant
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        client = ant.Anthropic(api_key=self.api_key)
        for attempt in range(3):
            try:
                r = client.messages.create(
                    model=self.model, max_tokens=1000, system=_SYS_JUDGE,
                    messages=[{"role": "user", "content": user_message}],
                )
                usage = {"input_tokens": r.usage.input_tokens,
                         "output_tokens": r.usage.output_tokens}
                return r.content[0].text, usage
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    log.error("ConsensusEngine Anthropic error for %s: %s", func_name, e)
                    return None, {}
        return None, {}

    def _call_gemini(self, user_message: str, func_name: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model, system_instruction=_SYS_JUDGE)
        for attempt in range(3):
            try:
                r = model.generate_content(user_message)
                return r.text, {}
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    log.error("ConsensusEngine Gemini error for %s: %s", func_name, e)
                    return None, {}
        return None, {}
