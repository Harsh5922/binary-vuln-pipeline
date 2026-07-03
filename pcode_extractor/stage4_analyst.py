"""
stage4_analyst.py  —  Stage 4: AI Memory-Safety Security Analyst
================================================================
Structured peer-review hypothesis evaluation — not "LLM verification."

Architecture (5 reasoning steps — peer review before decision):
  1. Peer Review          — explicit verdict on EACH evidence item
                            (Accepted / Accepted with reservation / Rejected / Insufficient / Missing)
  2. Hypothesis Evaluation — assess using ONLY the accepted/reserved evidence
  3. Competing Hypotheses — rank top-3 with "why winner wins" + "why others fail"
  4. Exploitability        — three independent dimensions (Reachability / Exploitability / Impact)
  5. Uncertainty Estimate  — two-dimensional: Evidence Quality + Model Uncertainty

Key design principles:
  - Stage 4 produces an Assessment, not a verdict.
    Stage 4.6 Bayesian Evidence Fusion (BEF) decides — ConsensusEngine only for conflicts.
  - `confirmed` is derived from hypothesis_support + exploitability + uncertainty.
  - Confidence = derived from combined uncertainty (no magic float).
  - Agreement vocabulary: strong_support / weak_support / contradicts / insufficient / absent.
    Gradated support — Strong vs Weak matters for BEF posterior computation.
  - Uncertainty split: Evidence Quality (how good is the package?) vs
    Model Uncertainty (how certain is the LLM about its own reasoning?).
    These are INDEPENDENT — rich evidence + high model uncertainty is a real case.

Public API:
    engine = HypothesisEvaluationEngine(provider="openrouter")
    assessments = engine.evaluate_all(taint_results, func_map)
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

def _est(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class HypothesisRank:
    """
    One hypothesis in the ranked competitor list.

    The winner must explain why competing hypotheses FAIL — not just its own support.
    This is the "competing hypotheses" pattern from Heuer's ACH methodology.
    """
    vuln_type: str
    verdict:   str   # "SUPPORTED" | "REJECTED" | "UNCERTAIN"
    reason:    str   # for SUPPORTED: why this evidence fits; for REJECTED: what is missing


@dataclass
class ExploitabilityAssessment:
    """
    Split exploitability into three independently reviewable dimensions (CVSS-inspired).

    reachability   — can an attacker reach this code path from an external entry point?
    exploitability — can the attacker control the relevant inputs at this location?
    impact         — what is the damage if the vulnerability is triggered?

    Reviewers can challenge each dimension independently.
    """
    reachability:   str   # "High|Medium|Low"
    exploitability: str   # "High|Medium|Low"
    impact:         str   # "Critical|High|Medium|Low"
    description:    str   # end-to-end exploit scenario


@dataclass
class UncertaintyEstimate:
    """
    Two-dimensional uncertainty — evidence quality and model certainty are separate concepts.

    evidence_quality  — how complete and trustworthy is the evidence PACKAGE?
                        "Excellent|Good|Poor"
                        Excellent: all 5 items present, no contradictions, no unknown calls
                        Good:      most items present, minor gaps or one unknown
                        Poor:      major gaps, many unknowns, or contradictory signals

    model_uncertainty — how certain is the LLM about its OWN reasoning over this evidence?
                        "Low|Moderate|High"
                        Low:      clear, unambiguous evidence — one conclusion is obvious
                        Moderate: some ambiguity in interpreting the signals
                        High:     LLM notes significant ambiguity (complex state machine,
                                  indirect call chain, unusual API pattern)

    These are INDEPENDENT.  Example — "Evidence Excellent, Model Uncertain":
    A state-machine use-after-free with complete forward+backward evidence can still
    require complex multi-step reasoning → high model uncertainty despite rich evidence.

    Structural flags (from Stage 3 EvidenceVector — NOT filled by the LLM):
      evidence_complete, unknown_functions, missing_summaries, incomplete_backward
    """
    # LLM self-assessed (from Stage 4 JSON response)
    evidence_quality:  str = "Good"     # "Excellent|Good|Poor"
    model_uncertainty: str = "Moderate" # "Low|Moderate|High"

    # Structural flags (populated from Stage 3 EvidenceVector, not from LLM)
    evidence_complete:   bool = True
    unknown_functions:   int  = 0
    missing_summaries:   bool = False
    incomplete_backward: bool = False

    @property
    def overall(self) -> str:
        """
        Combined uncertainty level for backward compatibility.
          Poor evidence OR High model uncertainty  → overall High
          Excellent evidence AND Low model uncert  → overall Low
          Otherwise                                → Moderate
        """
        if self.evidence_quality == "Poor" or self.model_uncertainty == "High":
            return "High"
        if self.evidence_quality == "Excellent" and self.model_uncertainty == "Low":
            return "Low"
        return "Moderate"

    @property
    def confidence(self) -> float:
        """Derive confidence from combined uncertainty — no magic weights."""
        return {"Low": 0.85, "Moderate": 0.60, "High": 0.35}.get(self.overall, 0.50)


@dataclass
class EvidenceAgreement:
    """
    Per-source agreement across 5 evidence signals — gradated support vocabulary.

    Vocabulary (no numbers — qualitative labels only):
      strong_support  — analysis provides clear, specific evidence confirming the hypothesis
                        e.g. forward taint reached sink AND INT_MULT present AND no bounds check
      weak_support    — analysis is broadly consistent but doesn't confirm specifically
                        e.g. source classified as external reader, but no taint path confirmed
      contradicts     — analysis argues against the primary hypothesis
                        e.g. backward slice found constant source, not external input
      insufficient    — analysis ran but evidence is too ambiguous/thin to classify
      absent          — analysis was not triggered (excluded from scoring denominator)

    Gradation matters: "3 strong_support vs 1 weak_support vs 1 contradicts" is a very
    different signal than "3 weak_support vs 1 contradicts". BEF posterior uses this.

    Scoring:
      strong_support = +1.0  |  weak_support = +0.5  |  contradicts = -0.5
      insufficient/absent = 0  (absent excluded from denominator)
    """
    source:     str = "absent"       # 3A source analysis
    forward:    str = "absent"       # 3B forward taint
    backward:   str = "absent"       # 3B+ backward slice
    semantic:   str = "insufficient" # 3C semantic risk
    behavioral: str = "absent"       # behavioral prior

    @property
    def _signals(self) -> list[str]:
        return [self.source, self.forward, self.backward, self.semantic, self.behavioral]

    @property
    def n_strong(self) -> int:
        return sum(1 for v in self._signals if v == "strong_support")

    @property
    def n_weak(self) -> int:
        return sum(1 for v in self._signals if v == "weak_support")

    @property
    def n_support(self) -> int:
        return self.n_strong + self.n_weak

    @property
    def n_contradict(self) -> int:
        return sum(1 for v in self._signals if v == "contradicts")

    @property
    def n_active(self) -> int:
        return sum(1 for v in self._signals if v != "absent")

    @property
    def score(self) -> float:
        """
        Weighted score normalized by active signals.
        strong_support=+1.0, weak_support=+0.5, contradicts=-0.5, insufficient=0.
        """
        if self.n_active == 0:
            return 0.0
        total = self.n_strong * 1.0 + self.n_weak * 0.5 - self.n_contradict * 0.5
        return total / self.n_active

    @property
    def level(self) -> str:
        s = self.score
        return "High" if s >= 0.60 else "Medium" if s >= 0.30 else "Low"

    def summary(self) -> str:
        return (
            f"{self.n_strong} strong + {self.n_weak} weak support, "
            f"{self.n_contradict} contradict of {self.n_active} active signals  "
            f"[src={self.source} fwd={self.forward} bwd={self.backward} "
            f"sem={self.semantic} beh={self.behavioral}]"
        )


@dataclass
class Assessment:
    """
    Stage 4 output: structured peer-review assessment.

    NOT a binary verdict — a scientific document that Stage 4.6 (Fusion Judge) uses
    alongside Stage 4.5 (Semantic Analysis) to produce the final Finding.

    `confirmed` is a derived property — Stage 4 never sets it directly.
    `confidence` is derived from UncertaintyEstimate.overall — no magic float.
    """
    func_name:    str
    entry_addr:   str
    vuln_type:    str
    sink_fn:      str
    op_seq:       int
    taint_source: str
    taint_path:   list[str]

    # Step 1: Per-item peer review verdicts
    peer_review: dict = field(default_factory=dict)
    # {"source": {"verdict": "Accepted|...", "reason": "..."}, "forward_taint": {...}, ...}

    # Step 2: Hypothesis support
    hypothesis_support:   str = "Weak"   # "Strong|Moderate|Weak|Unsupported"
    hypothesis_reasoning: str = ""

    # Step 3: Competing hypotheses (ranked)
    hypothesis_ranking: list[HypothesisRank] = field(default_factory=list)

    # Step 4: Exploitability (3 independent dimensions)
    exploitability_assessment: ExploitabilityAssessment = field(
        default_factory=lambda: ExploitabilityAssessment("Low", "Low", "Low", "")
    )

    # Step 5: Agreement (5 signals, richer vocabulary)
    agreement: EvidenceAgreement = field(default_factory=EvidenceAgreement)

    # Evidence
    supporting_evidence:    list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    missing_evidence:       list[str] = field(default_factory=list)

    # Uncertainty (Rec 5 — replaces single confidence float)
    uncertainty: UncertaintyEstimate = field(default_factory=UncertaintyEstimate)

    recommended_cwe:  str = ""
    recommended_cvss: str = ""

    model_used:      str   = ""
    analysis_time_s: float = 0.0
    match_kind:      str   = "NO_MATCH"
    description:     str   = ""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def confidence(self) -> float:
        """Derived from uncertainty level — no magic weights."""
        return self.uncertainty.confidence

    @property
    def exploitability(self) -> str:
        """Compatibility shim for code that reads Assessment.exploitability."""
        return self.exploitability_assessment.exploitability

    @property
    def confirmed(self) -> bool:
        """
        Confirmed when hypothesis is Moderate/Strong AND exploitability is Medium/High
        AND uncertainty is not High (High uncertainty = insufficient information to confirm).
        """
        return (
            self.hypothesis_support in ("Strong", "Moderate")
            and self.exploitability_assessment.exploitability in ("High", "Medium")
            and self.uncertainty.overall != "High"
        )

    @property
    def severity(self) -> str:
        ea = self.exploitability_assessment
        if self.hypothesis_support == "Strong" and ea.impact in ("Critical",):
            return "critical"
        if self.hypothesis_support == "Strong" and ea.exploitability == "High":
            return "high"
        if self.hypothesis_support == "Moderate" and ea.exploitability == "High":
            return "high"
        if self.hypothesis_support == "Moderate":
            return "medium"
        return "low"

    def to_finding(self):
        """Convert to legacy Finding for Stage 5 compatibility."""
        from reasoning_agent import Finding
        exploit = self.exploitability_assessment.description
        return Finding(
            func_name    = self.func_name,
            entry_addr   = self.entry_addr,
            vuln_type    = self.vuln_type,
            sink_fn      = self.sink_fn,
            op_seq       = self.op_seq,
            taint_source = self.taint_source,
            taint_path   = self.taint_path,
            confirmed    = self.confirmed,
            severity     = self.severity,
            reasoning    = self._format_reasoning(),
            exploit_condition     = exploit,
            false_positive_reason = "; ".join(self.contradictory_evidence[:2]),
            confidence   = self.confidence,
            model_used   = self.model_used,
            analysis_time_s = self.analysis_time_s,
            calibration  = {
                "hypothesis_support":  self.hypothesis_support,
                "reachability":        self.exploitability_assessment.reachability,
                "exploitability":      self.exploitability_assessment.exploitability,
                "impact":              self.exploitability_assessment.impact,
                "agreement_level":     self.agreement.level,
                "n_support":           self.agreement.n_support,
                "n_contradict":        self.agreement.n_contradict,
                "n_active":            self.agreement.n_active,
                "uncertainty":         self.uncertainty.overall,
                "unknown_functions":   self.uncertainty.unknown_functions,
            },
        )

    def _format_reasoning(self) -> str:
        lines = [f"[{self.hypothesis_support}] {self.hypothesis_reasoning}"]
        if self.hypothesis_ranking:
            winner = next((h for h in self.hypothesis_ranking if h.verdict == "SUPPORTED"), None)
            rejects = [h for h in self.hypothesis_ranking if h.verdict == "REJECTED"]
            if winner:
                lines.append(f"Primary: {winner.vuln_type} — {winner.reason}")
            for r in rejects[:2]:
                lines.append(f"Rejected: {r.vuln_type} — {r.reason}")
        ea = self.exploitability_assessment
        lines.append(
            f"Reachability={ea.reachability} Exploitability={ea.exploitability} Impact={ea.impact}. "
            f"Agreement: {self.agreement.summary()}. Uncertainty: {self.uncertainty.overall}."
        )
        return " ".join(lines)


# ─── Prompts ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an AI Memory-Safety Security Analyst performing structured peer review
of binary vulnerability candidates found by Hybrid Semantic Data-Flow Analysis.

You reason through 5 explicit steps — peer review BEFORE decision:
  1. Review each evidence item — use academic peer-review verdicts:
       Accepted                — evidence is reliable and supports the hypothesis
       Accepted with reservation — evidence is present but has a specific limitation
                                  (state the reservation in 'reason')
       Rejected                — evidence is unreliable or does not support the claim
       Insufficient            — evidence ran but is too thin/ambiguous to use
       Missing                 — this analysis was not triggered
  2. Evaluate the primary hypothesis using Accepted/Reserved items only
  3. Rank competing hypotheses — explain exactly why winner wins, why others fail
  4. Assess exploitability on 3 independent dimensions: Reachability, Exploitability, Impact
  5. Report TWO-DIMENSIONAL uncertainty:
       evidence_quality  = Excellent|Good|Poor  (how complete is the evidence package?)
       model_uncertainty = Low|Moderate|High    (how certain are YOU about your reasoning?)
       These are INDEPENDENT — rich evidence with complex logic = Good + High is valid.

Agreement vocabulary (no numbers — qualitative labels):
  strong_support  — clear, specific evidence that directly confirms the hypothesis
  weak_support    — broadly consistent but does not specifically confirm
  contradicts     — argues against the hypothesis
  insufficient    — triggered but too ambiguous
  absent          — not triggered

You do NOT produce a confirmed/rejected verdict.
You produce a structured Assessment. Bayesian Evidence Fusion will combine this with
Stage 4.5 (Semantic Analysis) — only genuine conflicts go to the Consensus Engine.

P-code context:
  CALL|fn|[args] — call  |  INT_MULT — integer multiply  |  STORE — memory write
  CBRANCH — conditional branch (bounds check)  |  INT_ZEXT/SEXT — extension/truncation risk
  VAR_N — SSA variable  |  const(0xN) — compile-time constant

Respond with ONLY valid JSON. No markdown. No explanation outside the JSON."""

_USER_TEMPLATE = """\
FUNCTION: {func_name}  @ {entry_addr}
PRIMARY HYPOTHESIS: {vuln_type}
SINK: {sink_fn}  seq={op_seq}
SOURCE: {taint_source}

══════════ STEP 1: PEER REVIEW ══════════
Review EACH evidence item before forming a judgment.

ITEM 1 — Source Analysis (3A)
{source_evidence}

ITEM 2 — Forward Taint (3B)
{forward_evidence}

ITEM 3 — Backward Slice (3B+)
{backward_evidence}

ITEM 4 — Semantic Risk (3C)
{semantic_evidence}

ITEM 5 — Behavioral Prior
{behavioral_evidence}

ITEM 6 — Uncertainty Flags (from Stage 3)
{uncertainty_flags}

Stage 3 detected contradictions:
{contradictions}

══════════ RELEVANT P-CODE ══════════
{pcode_block}

══════════ CALL GRAPH CONTEXT ══════════
{callee_context}

─────────────────────────────────────────────────────────────────
Respond with ONLY this JSON (fill every field):

{{
  "peer_review": {{
    "source":           {{"verdict": "Accepted|Accepted with reservation|Rejected|Insufficient|Missing", "reason": "<1 sentence — for 'with reservation' state what the reservation is>"}},
    "forward_taint":    {{"verdict": "Accepted|Accepted with reservation|Rejected|Insufficient|Missing", "reason": "<1 sentence>"}},
    "backward_slice":   {{"verdict": "Accepted|Accepted with reservation|Rejected|Insufficient|Missing", "reason": "<1 sentence>"}},
    "semantic_risk":    {{"verdict": "Accepted|Accepted with reservation|Rejected|Insufficient|Missing", "reason": "<1 sentence>"}},
    "behavioral_prior": {{"verdict": "Accepted|Accepted with reservation|Rejected|Insufficient|Missing", "reason": "<1 sentence>"}}
  }},
  "hypothesis_support":   "Strong|Moderate|Weak|Unsupported",
  "hypothesis_reasoning": "<2-3 sentences citing ONLY fully Accepted or Accepted-with-reservation items>",
  "hypothesis_ranking": [
    {{"vuln_type": "{vuln_type}",         "verdict": "SUPPORTED|REJECTED|UNCERTAIN", "reason": "<why winner wins — which specific evidence clinches it>"}},
    {{"vuln_type": "<alternative type>",  "verdict": "SUPPORTED|REJECTED|UNCERTAIN", "reason": "<exactly why this specific evidence does NOT support this alternative>"}},
    {{"vuln_type": "<another type>",      "verdict": "SUPPORTED|REJECTED|UNCERTAIN", "reason": "<why rejected>"}}
  ],
  "reachability":        "High|Medium|Low",
  "exploitability":      "High|Medium|Low",
  "impact":              "Critical|High|Medium|Low",
  "exploit_description": "<end-to-end: attacker input to code path to damage>",
  "evidence_agreement": {{
    "source":           "strong_support|weak_support|contradicts|insufficient|absent",
    "forward_taint":    "strong_support|weak_support|contradicts|insufficient|absent",
    "backward_slice":   "strong_support|weak_support|contradicts|insufficient|absent",
    "semantic_risk":    "strong_support|weak_support|contradicts|insufficient|absent",
    "behavioral_prior": "strong_support|weak_support|contradicts|insufficient|absent"
  }},
  "supporting_evidence":    ["<specific accepted observation>"],
  "contradictory_evidence": ["<specific observation that argues against the hypothesis>"],
  "missing_evidence":       ["<what information would increase certainty>"],
  "evidence_quality":    "Excellent|Good|Poor",
  "model_uncertainty":   "Low|Moderate|High",
  "recommended_cwe":   "CWE-XXX",
  "recommended_cvss":  "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
}}
"""


# ─── HypothesisEvaluationEngine ──────────────────────────────────────────────

class HypothesisEvaluationEngine:
    """
    Stage 4: AI Memory-Safety Analyst — structured peer review hypothesis evaluation.

    Drop-in replacement for ReasoningAgent with richer, more defensible output.
    """

    _DEFAULT_MODELS = {
        "groq":       "llama-3.3-70b-versatile",
        "gemini":     "gemini-2.5-flash",
        "anthropic":  "claude-haiku-4-5-20251001",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
    }
    _ENV_KEYS = {
        "groq":       "GROQ_API_KEY",
        "gemini":     "GEMINI_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    def __init__(
        self,
        provider:    str           = "groq",
        model:       Optional[str] = None,
        api_key:     Optional[str] = None,
        context_ops: int           = 20,
        llm_mode:    str           = "require",
    ):
        self.provider    = provider.lower()
        self.context_ops = context_ops
        self.model       = model or self._DEFAULT_MODELS.get(self.provider,
                                                              "llama-3.3-70b-versatile")
        env_var = self._ENV_KEYS.get(self.provider, "GROQ_API_KEY")
        if api_key is None:
            self.api_key = os.environ.get(env_var, "")
        else:
            self.api_key = api_key

        self.llm_enabled = bool(self.api_key) and llm_mode not in ("warn", "skip")

        if not self.llm_enabled:
            if llm_mode == "require":
                raise EnvironmentError(
                    f"[stage4_analyst] No API key for provider '{self.provider}'. "
                    f"Set {env_var} or use llm_mode='warn'."
                )
            log.warning("HypothesisEvaluationEngine: LLM disabled — no API key for %s",
                        self.provider)

        self._pattern_store = None
        try:
            from pattern_store import PatternStore
            db_path = os.environ.get("PATTERN_STORE_PATH", "pattern_store.db")
            self._pattern_store = PatternStore(db_path)
        except Exception as e:
            log.debug("Pattern store unavailable: %s", e)

        self._n_assessed  = 0
        self._n_confirmed = 0
        self._n_errors    = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        candidate:    object,
        func:         dict,
        taint_result: object = None,
    ) -> Optional[Assessment]:
        t0 = time.perf_counter()

        if self._is_fp_suppressed(candidate):
            return None

        ev_items     = self._peer_review_items(candidate, taint_result)
        pcode_block  = self._pcode_context(func, getattr(candidate, "op_seq", 0))
        callee_ctx   = self._callee_context(func)

        user_msg = _USER_TEMPLATE.format(
            func_name           = getattr(candidate, "func_name", "?"),
            entry_addr          = getattr(candidate, "entry_addr", "?"),
            vuln_type           = getattr(candidate, "vuln_type", "?"),
            sink_fn             = getattr(candidate, "sink_fn", "?"),
            op_seq              = getattr(candidate, "op_seq", -1),
            taint_source        = getattr(candidate, "taint_source", ""),
            source_evidence     = ev_items["source"],
            forward_evidence    = ev_items["forward"],
            backward_evidence   = ev_items["backward"],
            semantic_evidence   = ev_items["semantic"],
            behavioral_evidence = ev_items["behavioral"],
            uncertainty_flags   = ev_items["uncertainty"],
            contradictions      = ev_items["contradictions"],
            pcode_block         = pcode_block,
            callee_context      = callee_ctx,
        )

        raw = self._call_llm(user_msg, getattr(candidate, "func_name", "?"))
        elapsed = time.perf_counter() - t0

        if raw is None:
            self._n_errors += 1
            return self._fallback_assessment(candidate, elapsed)

        parsed = self._parse(raw)
        if parsed is None:
            self._n_errors += 1
            return self._fallback_assessment(candidate, elapsed)

        assessment = self._build_assessment(candidate, parsed, elapsed)
        self._n_assessed += 1
        if assessment.confirmed:
            self._n_confirmed += 1
            self._persist_to_pattern_store(candidate, assessment)

        return assessment

    def evaluate_all(
        self,
        taint_results: list,
        func_map:      dict,
        delay_s:       float = 2.0,
    ) -> list[Assessment]:
        all_cands  = self._collect_sorted(taint_results)
        total      = len(all_cands)
        llm_budget = 40
        llm_count  = 0
        _per_fn: dict[str, int] = {}
        _MAX_PER_FN = 4

        log.info("Stage 4 (AI Analyst — peer review): evaluating %d candidates …", total)
        assessments: list[Assessment] = []

        _ALLOC_FRAGS = frozenset({"malloc","calloc","realloc","alloc","valloc","mmap","brk"})

        for i, (result, cand) in enumerate(all_cands, 1):
            func       = func_map.get(getattr(cand, "func_name", ""), {})
            match_kind = getattr(cand, "match_kind", "LIBRARY_MATCH")
            sink_lower = (getattr(cand, "sink_fn", "") or "").lower()
            is_alloc   = any(f in sink_lower for f in _ALLOC_FRAGS)

            if match_kind == "LIBRARY_MATCH" and not is_alloc:
                assessments.append(self._auto_confirm(cand, "LIBRARY_MATCH"))
                log.info("[%d/%d] AUTO-CONFIRMED (library rule)  %s — %s",
                         i, total, cand.func_name, cand.vuln_type)
                continue

            fp_assessment = self._check_cache(cand, match_kind)
            if fp_assessment is not None:
                assessments.append(fp_assessment)
                log.info("[%d/%d] AUTO-CONFIRMED (cache)  %s — %s",
                         i, total, cand.func_name, cand.vuln_type)
                continue

            if not self.llm_enabled:
                continue

            if getattr(cand, "confidence", 0.0) < 0.35:
                continue

            fn_used = _per_fn.get(cand.func_name, 0)
            if fn_used >= _MAX_PER_FN or llm_count >= llm_budget:
                continue

            llm_count += 1
            _per_fn[cand.func_name] = fn_used + 1
            log.info("[%d/%d] Peer reviewing  %s — %s",
                     i, total, cand.func_name, cand.vuln_type)

            time.sleep(1)
            assessment = self.evaluate(cand, func, taint_result=result)
            if assessment:
                assessments.append(assessment)
                log.info(
                    "  → support=%-10s  exploit=%s/%s  agreement=%-6s  uncertainty=%s",
                    assessment.hypothesis_support,
                    assessment.exploitability_assessment.exploitability,
                    assessment.exploitability_assessment.impact,
                    assessment.agreement.level,
                    assessment.uncertainty.overall,
                )
            if llm_count > 0:
                time.sleep(delay_s)

        confirmed_n = sum(1 for a in assessments if a.confirmed)
        log.info(
            "Stage 4 done — %d assessed, %d confirmed, %d errors, %d auto",
            self._n_assessed, confirmed_n, self._n_errors,
            len(assessments) - llm_count,
        )
        return assessments

    # ── Evidence extraction for peer-review prompt ────────────────────────────

    def _peer_review_items(self, candidate, taint_result) -> dict:
        """Extract per-item evidence for the peer-review prompt format."""
        evidences = (getattr(taint_result, "evidences", {}) or {}) if taint_result else {}
        fp = getattr(candidate, "fingerprint", "") or ""
        ev = evidences.get(fp)

        if ev is None:
            path = " → ".join(getattr(candidate, "taint_path", []))
            return {
                "source":       f"Taint source: {getattr(candidate,'taint_source','unknown')}",
                "forward":      f"Taint path: {path}",
                "backward":     "(not available — no EvidenceVector)",
                "semantic":     "(not available)",
                "behavioral":   "(not available)",
                "uncertainty":  "(not available)",
                "contradictions": "None detected.",
            }

        # Source (3A)
        role_name = ev.source_role.label if ev.source_role else "Unknown"
        source_s = (
            f"Role: {role_name}  via {ev.source_fn or '(none)'}  "
            f"(base conf {ev.source_base_conf:.0%})"
        )

        # Forward (3B)
        if ev.forward_reached:
            path_s = " → ".join(ev.taint_path[-4:]) if ev.taint_path else "(unavailable)"
            forward_s = (
                f"REACHED sink after {ev.transformation_count} transformations. "
                f"Path: {path_s}. "
                f"INT_MULT: {'YES' if ev.path_has_mult else 'NO'}. "
                f"Bounds checked: {'YES' if ev.path_checked else 'NO'}."
            )
        else:
            forward_s = "DID NOT reach sink — taint chain broken before sink."

        # Backward (3B+)
        if ev.backward_reached:
            backward_s = (
                f"CONFIRMED external source ({ev.backward_role}) "
                f"at depth {ev.backward_depth}."
            )
        else:
            backward_s = "NOT triggered (forward source identified, or not applicable)."

        # Semantic (3C)
        if ev.semantic_items:
            items_s = "; ".join(
                f"{i.name} [{i.source}]: {i.reason}"
                for i in ev.semantic_items[:4]
            )
            semantic_s = (
                f"{len(ev.semantic_items)} observations  "
                f"[role={ev.semantic_role}]: {items_s}"
            )
        else:
            semantic_s = "No semantic observations (role = " + ev.semantic_role + ")."

        # Behavioral prior
        if ev.behavioral_similarity > 0.0:
            behavioral_s = (
                f"Structural similarity={ev.behavioral_similarity:.2f}, "
                f"historical TP rate={ev.behavioral_prior_tp:.0%} for this sink."
            )
        else:
            behavioral_s = "No historical data — this sink has not been seen before."

        # Uncertainty flags
        u_flags = getattr(ev.uncertainty, "flags", []) if ev.uncertainty else []
        uncertainty_s = (
            "; ".join(u_flags) if u_flags
            else "No flags — evidence appears complete."
        )

        # Contradictions
        contrad = getattr(ev, "contradictions", []) or []
        contrad_s = (
            "\n".join(f"  ! {c}" for c in contrad)
            if contrad else "None detected."
        )

        return {
            "source":       source_s,
            "forward":      forward_s,
            "backward":     backward_s,
            "semantic":     semantic_s,
            "behavioral":   behavioral_s,
            "uncertainty":  uncertainty_s,
            "contradictions": contrad_s,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collect_sorted(self, taint_results: list) -> list:
        items = [
            (result, cand)
            for result in taint_results
            for cand in result.vulns
        ]
        items.sort(key=lambda x: (
            0 if getattr(x[1], "match_kind", "NO_MATCH") == "LIBRARY_MATCH" else 1,
            -getattr(x[1], "confidence", 0.0),
        ))
        return items

    def _auto_confirm(self, cand, source: str) -> Assessment:
        vtype = getattr(cand, "vuln_type", "?")
        sev   = _vuln_severity(vtype)
        return Assessment(
            func_name    = cand.func_name,
            entry_addr   = cand.entry_addr,
            vuln_type    = vtype,
            sink_fn      = cand.sink_fn or "",
            op_seq       = cand.op_seq,
            taint_source = cand.taint_source or "",
            taint_path   = list(cand.taint_path or []),
            peer_review  = {
                "source":        {"verdict": "Accepted", "reason": "Known dangerous function pattern."},
                "forward_taint": {"verdict": "Accepted", "reason": f"{source} confirmed."},
            },
            hypothesis_support   = "Strong",
            hypothesis_reasoning = f"Auto-confirmed via {source} rule.",
            hypothesis_ranking   = [HypothesisRank(vtype, "SUPPORTED", "Library rule match.")],
            exploitability_assessment = ExploitabilityAssessment(
                "High" if sev in ("critical", "high") else "Medium",
                "High" if sev in ("critical", "high") else "Medium",
                "High" if sev == "critical" else "Medium",
                f"Tainted argument flows directly into dangerous sink {cand.sink_fn}.",
            ),
            agreement   = EvidenceAgreement(source="supports", forward="supports"),
            uncertainty = UncertaintyEstimate(evidence_quality="Excellent", model_uncertainty="Low"),
            supporting_evidence = [f"Known dangerous sink: {cand.sink_fn}"],
            recommended_cwe = _default_cwe(vtype),
            model_used  = f"auto/{source}",
            match_kind  = source,
        )

    def _check_cache(self, cand, match_kind: str) -> Optional[Assessment]:
        if self._pattern_store is None:
            return None
        fp   = getattr(cand, "fingerprint", "") or ""
        sink = getattr(cand, "sink_fn", "") or ""
        args = list(getattr(cand, "arg_sizes", []) or [])
        try:
            if fp and match_kind != "STRUCTURAL_MATCH":
                res = self._pattern_store.lookup_fingerprint(fp)
                if res and res.get("confirmed"):
                    return self._auto_confirm(cand, "FINGERPRINT_MATCH")
            cached = self._pattern_store.lookup(sink, args)
            if cached is not None and cached.get("is_sink"):
                return self._auto_confirm(cand, "STRUCTURAL_MATCH")
        except Exception:
            pass
        return None

    def _is_fp_suppressed(self, cand) -> bool:
        if self._pattern_store is None:
            return False
        try:
            return self._pattern_store.is_fp_suppressed(
                fn_name=cand.func_name, vuln_type=cand.vuln_type, sink_fn=cand.sink_fn,
            )
        except Exception:
            return False

    def _pcode_context(self, func: dict, sink_seq: int) -> str:
        ops  = func.get("ops") or []
        half = self.context_ops // 2
        window = [o for o in ops if abs(o.get("seq", 0) - sink_seq) <= half]
        lines  = []
        for op in window:
            seq   = op.get("seq", "?")
            mnem  = op.get("op", "?")
            out   = op.get("output")
            inps  = op.get("inputs") or []
            out_s = out["name"] if isinstance(out, dict) and out else "_"
            inp_s = ", ".join(i["name"] for i in inps if isinstance(i, dict) and i.get("name"))
            mark  = " ← SINK" if seq == sink_seq else ""
            lines.append(f"  [{str(seq):<4}] {mnem:<12} {out_s}  ←  {inp_s}{mark}")
        return "\n".join(lines) if lines else "(no ops)"

    def _callee_context(self, func: dict) -> str:
        ops  = func.get("ops") or []
        seen: set[str] = set()
        lines = []
        for op in ops:
            if op.get("op") not in ("CALL", "CALLIND"):
                continue
            inps = op.get("inputs") or []
            if not inps:
                continue
            name = (inps[0].get("name", "") if isinstance(inps[0], dict) else str(inps[0]))
            if not name or name in seen or name.startswith("ram("):
                continue
            seen.add(name)
            role = "(unknown)"
            if self._pattern_store is not None:
                try:
                    s = self._pattern_store.get_learned_summary(name, [])
                    if s:
                        role = f"role={s.get('likely_role','?')}"
                except Exception:
                    pass
            lines.append(f"  {name:<40} {role}")
            if len(lines) >= 8:
                break
        return "\n".join(lines) if lines else "  (none)"

    def _build_assessment(self, cand, parsed: dict, elapsed: float) -> Assessment:
        pr = parsed.get("peer_review", {})

        # Step 3: Competing hypotheses
        ranking = [
            HypothesisRank(
                vuln_type = h.get("vuln_type", "?"),
                verdict   = h.get("verdict", "UNCERTAIN"),
                reason    = h.get("reason", ""),
            )
            for h in parsed.get("hypothesis_ranking", [])[:4]
            if isinstance(h, dict)
        ]

        # Step 4: Split exploitability
        ea = ExploitabilityAssessment(
            reachability   = parsed.get("reachability", "Low"),
            exploitability = parsed.get("exploitability", "Low"),
            impact         = parsed.get("impact", "Low"),
            description    = parsed.get("exploit_description", ""),
        )

        # Step 5a: Agreement (5 signals — gradated vocabulary)
        agr_raw = parsed.get("evidence_agreement", {})
        agr = EvidenceAgreement(
            source     = agr_raw.get("source",           "absent"),
            forward    = agr_raw.get("forward_taint",    "absent"),
            backward   = agr_raw.get("backward_slice",   "absent"),
            semantic   = agr_raw.get("semantic_risk",    "insufficient"),
            behavioral = agr_raw.get("behavioral_prior", "absent"),
        )

        # Step 5b: Two-dimensional uncertainty (LLM self-assessment)
        unc = UncertaintyEstimate(
            evidence_quality   = parsed.get("evidence_quality",  "Good"),
            model_uncertainty  = parsed.get("model_uncertainty", "Moderate"),
            # Structural flags: derived from agreement signals, not from LLM response
            incomplete_backward = agr.backward in ("absent", "insufficient"),
        )

        return Assessment(
            func_name    = getattr(cand, "func_name", "?"),
            entry_addr   = getattr(cand, "entry_addr", "?"),
            vuln_type    = getattr(cand, "vuln_type", "?"),
            sink_fn      = getattr(cand, "sink_fn", "") or "",
            op_seq       = getattr(cand, "op_seq", -1),
            taint_source = getattr(cand, "taint_source", "") or "",
            taint_path   = list(getattr(cand, "taint_path", []) or []),
            peer_review  = pr,
            hypothesis_support   = parsed.get("hypothesis_support", "Weak"),
            hypothesis_reasoning = parsed.get("hypothesis_reasoning", ""),
            hypothesis_ranking   = ranking,
            exploitability_assessment = ea,
            agreement    = agr,
            supporting_evidence    = list(parsed.get("supporting_evidence", [])),
            contradictory_evidence = list(parsed.get("contradictory_evidence", [])),
            missing_evidence       = list(parsed.get("missing_evidence", [])),
            uncertainty  = unc,
            recommended_cwe  = parsed.get("recommended_cwe", ""),
            recommended_cvss = parsed.get("recommended_cvss", ""),
            model_used      = f"{self.provider}/{self.model}",
            analysis_time_s = round(elapsed, 3),
            match_kind      = getattr(cand, "match_kind", "NO_MATCH"),
            description     = getattr(cand, "description", "") or "",
        )

    def _fallback_assessment(self, cand, elapsed: float) -> Assessment:
        return Assessment(
            func_name    = getattr(cand, "func_name", "?"),
            entry_addr   = getattr(cand, "entry_addr", "?"),
            vuln_type    = getattr(cand, "vuln_type", "?"),
            sink_fn      = getattr(cand, "sink_fn", "") or "",
            op_seq       = getattr(cand, "op_seq", -1),
            taint_source = getattr(cand, "taint_source", "") or "",
            taint_path   = list(getattr(cand, "taint_path", []) or []),
            hypothesis_support   = "Weak",
            hypothesis_reasoning = "LLM evaluation failed.",
            uncertainty  = UncertaintyEstimate(evidence_quality="Poor", model_uncertainty="High", evidence_complete=False),
            model_used   = "error/fallback",
            analysis_time_s = round(elapsed, 3),
        )

    def _persist_to_pattern_store(self, cand, assessment: Assessment) -> None:
        if self._pattern_store is None or getattr(cand, "match_kind", "") != "NO_MATCH":
            return
        try:
            fp = getattr(cand, "fingerprint", "") or ""
            if fp:
                self._pattern_store.store_fingerprint(
                    fingerprint=fp, confirmed=True,
                    confidence=assessment.confidence,
                    vuln_type=assessment.vuln_type,
                    example_func=assessment.func_name,
                )
            self._pattern_store.store_structural(
                cand.sink_fn or cand.vuln_type,
                list(getattr(cand, "arg_sizes", []) or []) or [1],
                {
                    "sink": True, "is_sink": True,
                    "sink_type": assessment.vuln_type,
                    "confidence": assessment.confidence,
                    "notes": (
                        f"AI Analyst [{assessment.hypothesis_support}] "
                        f"uncertainty={assessment.uncertainty.overall} "
                        f"agreement={assessment.agreement.level}"
                    ),
                },
            )
        except Exception as e:
            log.debug("Pattern store write failed: %s", e)

    # ── LLM infrastructure ────────────────────────────────────────────────────

    def _call_llm(self, user_message: str, func_name: str) -> Optional[str]:
        try:
            from llm_cost_tracker import GLOBAL_TRACKER
            tracker = GLOBAL_TRACKER
        except Exception:
            tracker = None
        t0 = time.perf_counter()
        if self.provider == "groq":
            text, usage = self._call_groq(user_message, func_name)
        elif self.provider == "openrouter":
            text, usage = self._call_openrouter(user_message, func_name)
        elif self.provider == "gemini":
            text, usage = self._call_gemini(user_message, func_name)
        elif self.provider == "anthropic":
            text, usage = self._call_anthropic(user_message, func_name)
        else:
            log.error("Unknown provider: %s", self.provider)
            return None
        elapsed = time.perf_counter() - t0
        if text is not None and tracker is not None:
            try:
                tracker.record(
                    stage="analyst", model=self.model,
                    input_tokens  = usage.get("input_tokens") or _est(user_message),
                    output_tokens = usage.get("output_tokens") or _est(text),
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
            log.warning("AI Analyst: no JSON in response")
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            log.warning("AI Analyst: JSON parse error: %s", e)
            return None

    def _call_groq(self, user_message: str, func_name: str):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")
        client = Groq(api_key=self.api_key)
        waits = [65, 130]
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    max_tokens=1600, temperature=0.0,
                )
                usage = {}
                if r.usage:
                    usage = {"input_tokens": r.usage.prompt_tokens,
                             "output_tokens": r.usage.completion_tokens}
                return r.choices[0].message.content, usage
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate" in err.lower()) and attempt < 2:
                    log.warning("Rate limited — waiting %ds", waits[attempt])
                    time.sleep(waits[attempt])
                else:
                    log.error("Groq error for %s: %s", func_name, e)
                    return None, {}
        return None, {}

    def _call_openrouter(self, user_message: str, func_name: str):
        import urllib.request as _ur
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/binary-vuln-pipeline",
            "X-Title":       "Binary Vulnerability Analysis Pipeline",
        }
        payload = json.dumps({
            "model": self.model, "max_tokens": 1600, "temperature": 0.0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
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
                    log.error("OpenRouter error for %s: %s", func_name, e)
                    return None, {}
        return None, {}

    def _call_gemini(self, user_message: str, func_name: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model, system_instruction=_SYSTEM_PROMPT)
        for attempt in range(3):
            try:
                r = model.generate_content(user_message)
                return r.text, {}
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    log.error("Gemini error for %s: %s", func_name, e)
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
                    model=self.model, max_tokens=1600, system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                usage = {"input_tokens": r.usage.input_tokens,
                         "output_tokens": r.usage.output_tokens}
                return r.content[0].text, usage
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    log.error("Anthropic error for %s: %s", func_name, e)
                    return None, {}
        return None, {}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _vuln_severity(vuln_type: str) -> str:
    return {
        "buffer_overflow":     "high",
        "heap_overflow":       "high",
        "integer_overflow":    "high",
        "out_of_bounds_write": "high",
        "write_what_where":    "critical",
        "format_string":       "high",
        "command_injection":   "critical",
        "use_after_free":      "high",
        "double_free":         "medium",
        "integer_truncation":  "medium",
        "null_dereference":    "medium",
    }.get(vuln_type, "medium")


def _default_cwe(vuln_type: str) -> str:
    return {
        "buffer_overflow":     "CWE-120",
        "heap_overflow":       "CWE-122",
        "integer_overflow":    "CWE-190",
        "integer_truncation":  "CWE-197",
        "integer_underflow":   "CWE-191",
        "out_of_bounds_read":  "CWE-125",
        "out_of_bounds_write": "CWE-787",
        "null_dereference":    "CWE-476",
        "use_after_free":      "CWE-416",
        "double_free":         "CWE-415",
        "format_string":       "CWE-134",
        "command_injection":   "CWE-78",
        "check_bypass":        "CWE-190",
        "write_what_where":    "CWE-787",
        "logic_bug":           "CWE-682",
    }.get(vuln_type, "CWE-119")
