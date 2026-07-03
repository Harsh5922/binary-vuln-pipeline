"""
reasoning_agent.py

Stage 5 — Reasoning Agent

Reviews VulnCandidates from the taint engine and confirms whether
each one is a real vulnerability or a false positive.

Why this stage is needed
-------------------------
The taint engine flags every case where tainted data reaches a dangerous
operation. But not all of these are real vulnerabilities:

  - The attacker may not control the taint source (e.g. getenv reads an
    env var that is set by the OS, not the user)
  - A bounds check may exist earlier in the call chain that the taint
    engine did not see (inter-procedural)
  - The buffer may be large enough that overflow is not practically
    possible
  - The dangerous operation may be inside an admin-only code path

The reasoning agent uses the LLM to think through these cases.

What the LLM receives per candidate
-------------------------------------
  - The VulnCandidate details (type, sink, taint path, confidence)
  - The raw P-code of the function (limited to relevant ops)
  - The pattern match results for the relevant CALL ops
  - The taint path from source to sink

What the LLM produces
----------------------
  - confirmed: True | False
  - severity: critical | high | medium | low | info
  - reasoning: why confirmed or rejected
  - exploit_condition: what the attacker needs to trigger this
  - false_positive_reason: why it might be a false positive (if any)
  - confidence: 0.0 – 1.0

Usage
-----
    from reasoning_agent import ReasoningAgent
    from taint_engine     import TaintEngine, TaintResult

    agent   = ReasoningAgent(provider="groq")
    results = agent.review_all(taint_results, func_map)

    for finding in results:
        if finding.confirmed:
            print(finding.summary())
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from taint_engine import VulnCandidate, TaintResult
from confidence_calibrator import calibrate_finding_confidence, CalibrationBreakdown

def _estimate_tokens_ra(text: str) -> int:
    return max(1, len(text) // 4)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    """
    A confirmed (or rejected) vulnerability finding.
    Output of the reasoning agent.
    """
    # Identity
    func_name:    str
    entry_addr:   str
    vuln_type:    str
    sink_fn:      str
    op_seq:       int

    # Taint path
    taint_source: str
    taint_path:   list[str]

    # Reasoning agent verdict
    confirmed:    bool    # True = real vulnerability
    severity:     str     # critical | high | medium | low | info
    reasoning:    str     # why confirmed or rejected
    exploit_condition: str  # what attacker needs
    false_positive_reason: str  # why it might be FP

    # Quality
    confidence:   float
    model_used:   str
    analysis_time_s: float

    # Task 7: Explanation graph — causal chain for every confirmed vulnerability.
    # List of nodes: [{"fn": str, "role": str, "effect": str}, ...]
    # Empty list if not yet built (backwards-compatible default).
    causal_chain: list = None  # type: ignore[assignment]

    # Item 3: per-signal calibration breakdown (dict form of CalibrationBreakdown)
    calibration:  dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.causal_chain is None:
            self.causal_chain = []
        if self.calibration is None:
            self.calibration = {}

    # ── Explanation graph builder (Task 7) ────────────────────────────────────
    _SOURCE_ROLES = {
        "fread": "file_read", "recv": "net_recv", "read": "file_read",
        "fgets": "file_read", "getenv": "env_read", "scanf": "stdin_read",
        "png_crc_read": "png_read", "psf_binheader_readf": "snd_read",
        "TIFFGetField": "tiff_read", "xmlParserInputShrink": "xml_read",
    }
    _VULN_IMPACT = {
        "buffer_overflow":    "heap/stack buffer overflow → arbitrary code execution",
        "integer_overflow":   "integer overflow → allocation undersize → heap overflow",
        "integer_truncation": "integer truncation → size mismatch → memory corruption",
        "unbounded_copy":     "unbounded memory copy → stack/heap overflow",
        "use_after_free":     "use-after-free → type confusion or arbitrary code execution",
        "format_string":      "format string injection → arbitrary read/write",
        "command_injection":  "command injection → OS command execution",
        "null_deref":         "null pointer dereference → crash / info leak",
        "double_free":        "double-free → heap metadata corruption",
        "check_bypass":       "bounds check bypass → out-of-bounds read/write",
        "off_by_one":         "off-by-one error → adjacent memory corruption",
        "parser_state":       "invalid parser state → undefined behavior or memory corruption",
        "incorrect_bounds":   "incorrect size limit → buffer overread/overwrite",
        "logic_bug":          "logic error → undefined behavior",
    }

    def build_causal_chain(self) -> list:
        """Build the explanation graph from this finding's fields (Task 7)."""
        chain = []

        # Source node
        source_fn   = self.taint_source or "unknown_source"
        source_role = self._SOURCE_ROLES.get(source_fn, "external_input")
        chain.append({
            "fn":     source_fn,
            "role":   source_role,
            "effect": "introduces attacker-controlled data",
        })

        # Intermediate taint path nodes (skip source and sink — already added)
        seen = {source_fn, self.sink_fn}
        for step in (self.taint_path or []):
            # steps are strings like "external:fread → var1 → memcpy"
            # Extract any function name from the step
            fn = step.strip()
            if fn and fn not in seen and not fn.startswith("var") and len(fn) < 60:
                seen.add(fn)
                chain.append({
                    "fn":     fn,
                    "role":   "propagates_taint",
                    "effect": "taint flows through return value or output argument",
                })

        # Sink node
        if self.sink_fn and self.sink_fn != "LOGIC":
            chain.append({
                "fn":     self.sink_fn,
                "role":   "sink",
                "effect": f"dangerous operation at seq {self.op_seq}",
            })

        # Impact leaf
        impact = self._VULN_IMPACT.get(self.vuln_type, "undefined behavior")
        chain.append({
            "fn":     "IMPACT",
            "role":   "vulnerability",
            "effect": impact,
        })

        self.causal_chain = chain
        return chain

    def causal_chain_str(self) -> str:
        """One-line rendering: fn[role] → fn[role] → …"""
        if not self.causal_chain:
            self.build_causal_chain()
        return " → ".join(
            f"{n['fn']}[{n['role']}]" for n in self.causal_chain
        )

    def summary(self) -> str:
        status = "CONFIRMED" if self.confirmed else "rejected"
        return (
            f"[{status}]  {self.vuln_type}  in {self.func_name}  "
            f"@ seq {self.op_seq}  sink={self.sink_fn}  "
            f"severity={self.severity}  conf={self.confidence:.0%}"
        )

    def report_block(self) -> str:
        """Full formatted block for the final report."""
        sep = "═" * 62
        status = "✓ CONFIRMED" if self.confirmed else "✗ REJECTED"
        lines = [
            sep,
            f"  {status}  [{self.vuln_type}]",
            sep,
            f"  Function  : {self.func_name}  @ {self.entry_addr}",
            f"  Sink      : {self.sink_fn}  (seq {self.op_seq})",
            f"  Severity  : {self.severity}",
            f"  Confidence: {self.confidence:.0%}",
            f"  Taint path: {' → '.join(self.taint_path)}",
        ]
        # Item 3: show per-signal breakdown if available
        if self.calibration:
            c = self.calibration
            lines.append(
                f"  Calibration: llm={c.get('llm_conf',0):.2f}  "
                f"pat={c.get('pattern_score',0):.2f}  "
                f"tnt={c.get('taint_conf',0):.2f}  "
                f"src={c.get('source_score',0):.2f}  "
                f"ver={c.get('verif_bonus',0):.2f}"
            )
        # Task 7: Explanation graph (build on first render if not already done)
        if self.confirmed:
            chain_str = self.causal_chain_str()
            lines.append(f"  Causal chain: {chain_str}")
        lines += [
            "",
            f"  Reasoning:",
        ]
        for line in self.reasoning.split(". "):
            if line.strip():
                lines.append(f"    {line.strip()}.")
        if self.exploit_condition:
            lines.append("")
            lines.append(f"  Exploit condition:")
            lines.append(f"    {self.exploit_condition}")
        if self.false_positive_reason:
            lines.append("")
            lines.append(f"  False positive risk:")
            lines.append(f"    {self.false_positive_reason}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior binary security researcher reviewing vulnerability candidates
found by automated taint analysis of P-code (Ghidra's intermediate representation).

Your job is to confirm or reject each candidate by reasoning carefully about:
1. Does the attacker actually control the taint source?
2. Is there a bounds check or validation that the taint engine missed?
3. Is the code path reachable by an attacker?
4. Is the vulnerability practically exploitable?

P-code context:
  CALL|fn|[args]  — function call with argument sizes
  CALLIND         — indirect call through a function pointer (pointer may be NULL)
  STORE           — write to memory through pointer
  LOAD            — read from memory through pointer
  PTRADD/INT_ADD  — pointer/integer arithmetic
  CBRANCH         — conditional branch (possible bounds check)
  INT_EQUAL       — compare two values (e.g. check for NULL == 0)

IMPORTANT — CWE-476 structural patterns detected by the engine:

  Pattern A — CALLIND + NULL check (taint_source=structural:callind_null_check):
    A function is called through a pointer (CALLIND), then within a few ops
    the RETURN VALUE is compared to zero (INT_EQUAL/NOTEQUAL vs const(0x0)).
    This is strong evidence the programmer knows the POINTER may be NULL —
    the NULL check acknowledges that CALLIND might fail or return NULL.
    The vulnerability: if the function pointer itself is NULL, CALLIND crashes
    before the check can run. DO NOT reject this pattern — it is CWE-476.

  Pattern B — Load-Load wrong-var check (taint_source=structural:load_load_wrong_var_check):
    Sequence: LOAD VAR_ptr = [struct_field], then LOAD VAR_val = [VAR_ptr],
    then INT_EQUAL(VAR_val, 0) — the programmer checked the LOADED VALUE
    (VAR_val) but NOT the ADDRESS used to load it (VAR_ptr).
    If VAR_ptr is NULL, the second LOAD crashes BEFORE the null check runs.
    The check is on the WRONG variable — this IS a null dereference CWE-476.

When you see taint_source containing "structural:callind_null_check" or
"structural:load_load_wrong_var_check", treat these as HIGH confidence
structural vulnerability patterns unless there is clear evidence of a guard.

Be precise and evidence-based. If you are uncertain, say so.
Respond with ONLY valid JSON — no markdown, no explanation outside the JSON.
"""

_USER_TEMPLATE = """\
Review this vulnerability candidate found by Stage 3 Hybrid Semantic Data-Flow Analysis.

FUNCTION: {func_name}  @ {entry_addr}
VULNERABILITY TYPE: {vuln_type}
SINK: {sink_fn}  at seq {op_seq}
TAINT SOURCE: {taint_source}
DESCRIPTION: {description}

{evidence_vector_block}

CALL GRAPH CONTEXT:
  Callers (who calls this function):
{caller_context}
  Callees with known roles (functions this function calls):
{callee_context}

REFERENCED STRINGS IN THIS FUNCTION:
{string_context}

RELEVANT P-CODE (around the sink):
{pcode_context}

PATTERN MATCHES IN THIS FUNCTION:
{pattern_matches}

Respond with ONLY this JSON:
{{
  "confirmed": true,
  "severity": "critical|high|medium|low|info",
  "reasoning": "2-3 sentences citing specific evidence items above",
  "exploit_condition": "what the attacker needs to trigger this (input format, code path, etc.)",
  "false_positive_reason": "why this might NOT be a real vulnerability, or empty string if clearly real",
  "confidence": 0.85
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Agent
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningAgent:
    """
    Reviews VulnCandidates and produces confirmed Findings.

    Uses LLM only for candidates (not all functions).
    If 50 functions produce 8 candidates → 8 LLM calls maximum.

    Parameters
    ----------
    provider  : "groq" | "gemini" | "anthropic" | "openrouter"
    model     : model string (defaults to best available per provider)
    api_key   : API key (reads from env if not given)
    context_ops : how many P-code ops to include around the sink
    """

    def __init__(
        self,
        provider:    str           = "groq",
        model:       Optional[str] = None,
        api_key:     Optional[str] = None,
        context_ops: int           = 20,
        llm_mode:    str           = "require",  # "require" | "warn" | "skip"
    ):
        self.provider    = provider.lower()
        self.context_ops = context_ops

        # Default models per provider
        _defaults = {
            "groq":        "llama-3.3-70b-versatile",
            "gemini":      "gemini-2.5-flash",
            "anthropic":   "claude-haiku-4-5-20251001",
            "openrouter":  "meta-llama/llama-3.3-70b-instruct",  # paid — uses $5 credit, no RPM limit
        }
        self.model = model or _defaults.get(self.provider, "llama-3.3-70b-versatile")

        # API key from env
        _env_keys = {
            "groq":       "GROQ_API_KEY",
            "gemini":     "GEMINI_API_KEY",
            "anthropic":  "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        # IMPORTANT: use `is None` not truthiness check.
        # `api_key=""` (passed by --no-llm) must NOT fall through to env vars.
        # `"" or os.environ.get(...)` would pick up GROQ_API_KEY anyway.
        if api_key is None:
            self.api_key = os.environ.get(
                _env_keys.get(self.provider, "GROQ_API_KEY"), ""
            )
        else:
            self.api_key = api_key  # empty string = explicitly disabled

        # Force disable when --no-llm passed (llm_mode="warn") or explicitly
        # api_key="". Belt-and-suspenders: either condition disables LLM.
        self.llm_enabled = bool(self.api_key) and llm_mode not in ("warn", "skip")

        if not self.llm_enabled:
            env_var = _env_keys.get(self.provider, "GROQ_API_KEY")
            if llm_mode == "require":
                raise EnvironmentError(
                    f"[reasoning_agent] No API key found for provider '{self.provider}'.\n"
                    f"Set the {env_var} environment variable, or pass api_key=..., "
                    f"or use llm_mode='warn' / llm_mode='skip' to disable LLM review."
                )
            elif llm_mode == "warn":
                log.warning(
                    "No API key for provider '%s' (%s not set). "
                    "LLM review disabled — only LIBRARY_MATCH/STRUCTURAL_MATCH "
                    "candidates will be auto-confirmed.",
                    self.provider, env_var,
                )

        self._confirmed = 0
        self._rejected  = 0
        self._errors    = 0

        # Pattern store — used to persist LLM decisions so future runs
        # auto-confirm the same structural pattern without an LLM call.
        # Loaded lazily; stays None if pattern_store.db is unavailable.
        self._pattern_store = None
        try:
            from pattern_store import PatternStore
            db_path = os.environ.get("PATTERN_STORE_PATH", "pattern_store.db")
            self._pattern_store = PatternStore(db_path)
            log.debug("ReasoningAgent: pattern store loaded from %s", db_path)
        except Exception as exc:
            log.debug("ReasoningAgent: pattern store unavailable: %s", exc)

    # ── Public API ────────────────────────────────────────────────────

    def review(
        self,
        candidate:    VulnCandidate,
        func:         dict,
        taint_result: "TaintResult | None" = None,
    ) -> Optional[Finding]:
        """
        Review one VulnCandidate.

        Parameters
        ----------
        candidate    : VulnCandidate from taint engine
        func         : raw function dict (for P-code context)
        taint_result : TaintResult (for full flow log context)
        """
        t0 = time.perf_counter()

        # Item 5: Active Learning — check FP suppression before any LLM call.
        # If a human previously marked this (fn, vuln_type, sink) as FP, skip it.
        if self._pattern_store is not None:
            try:
                if self._pattern_store.is_fp_suppressed(
                    fn_name   = candidate.func_name,
                    vuln_type = candidate.vuln_type,
                    sink_fn   = candidate.sink_fn,
                ):
                    log.info(
                        "FP-suppressed: skipping %s / %s / %s (user-rejected previously)",
                        candidate.func_name, candidate.vuln_type, candidate.sink_fn,
                    )
                    return None  # silently drop — same as no finding
            except Exception as _se:
                log.debug("FP suppression check error: %s", _se)

        # Build context — use flow log if available, else P-code window
        if taint_result and getattr(taint_result, "flow_steps", None):
            pcode_ctx = self._flow_context(taint_result, candidate)
        else:
            pcode_ctx = self._pcode_context(func, candidate.op_seq)

        pattern_info  = self._pattern_summary(func)
        callee_ctx    = self._callee_context(func)
        caller_ctx    = self._caller_context(
                            candidate.func_name,
                            getattr(self, "_all_funcs", [])
                        )
        string_ctx    = self._string_context(func)

        # Build evidence vector block for the LLM prompt.
        # When Stage3Orchestrator is used, taint_result.evidences contains
        # EvidenceVector objects keyed by candidate fingerprint.
        ev_block = ""
        if taint_result is not None:
            evidences = getattr(taint_result, "evidences", {}) or {}
            fp = getattr(candidate, "fingerprint", None) or ""
            ev = evidences.get(fp)
            if ev is not None:
                try:
                    ev_block = ev.format_for_llm()
                except Exception:
                    ev_block = ""
        if not ev_block:
            # Fallback: minimal evidence block for bare TaintEngine output
            ev_block = (
                f"TAINT PATH: {' -> '.join(candidate.taint_path)}\n"
                f"TAINT CONFIDENCE: {candidate.confidence:.0%}"
            )

        user_msg = _USER_TEMPLATE.format(
            func_name            = candidate.func_name,
            entry_addr           = candidate.entry_addr,
            vuln_type            = candidate.vuln_type,
            sink_fn              = candidate.sink_fn,
            op_seq               = candidate.op_seq,
            description          = candidate.description,
            taint_source         = candidate.taint_source,
            evidence_vector_block= ev_block,
            pcode_context        = pcode_ctx,
            pattern_matches      = pattern_info,
            callee_context       = callee_ctx,
            caller_context       = caller_ctx,
            string_context       = string_ctx,
        )

        raw = self._call_llm(user_msg, candidate.func_name)
        elapsed = time.perf_counter() - t0

        if raw is None:
            self._errors += 1
            return None

        parsed = self._parse(raw)
        if parsed is None:
            self._errors += 1
            return None

        confirmed = parsed.get("confirmed", False)
        if confirmed:
            self._confirmed += 1
        else:
            self._rejected += 1

        # Persist LLM decision to pattern store so future runs on the same
        # binary (or a similar binary) auto-confirm without an LLM call.
        # ── Pattern store learning ────────────────────────────────────────
        # Only store CONFIRMED results for NO_MATCH candidates.
        # Key strategy per vuln_type:
        #   unknown_call  → (fn_name, arg_sizes) structural fingerprint
        #                   Future binaries with same arg pattern auto-match
        #   write_what_where → (vuln_type, [1]) — structural STORE pattern
        #   null_dereference → (vuln_type, [1]) — structural CWE-476 pattern
        # REJECTED candidates are NOT stored — they would falsely
        # suppress future real vulnerabilities with the same pattern.
        if (self._pattern_store is not None
                and candidate.match_kind == "NO_MATCH"
                and confirmed):  # only store confirmed findings
            _conf = float(parsed.get("confidence", 0.5))
            _arg_sizes = list(getattr(candidate, "arg_sizes", []) or [])
            # Pick store key and sizes based on vuln type
            if candidate.vuln_type == "unknown_call" and _arg_sizes:
                # Best key: arg sizes pattern (binary-agnostic fingerprint)
                # This lets future binaries match ANY unknown function with
                # the same arg signature → STRUCTURAL_MATCH → no LLM needed
                _store_fn    = candidate.sink_fn  # raw address e.g. ram(0x...)
                _store_sizes = _arg_sizes
            elif candidate.vuln_type in ("write_what_where", "null_dereference"):
                # Structural pattern: keyed by type so same pattern auto-confirms
                _store_fn    = candidate.vuln_type
                _store_sizes = [len(candidate.taint_path)]
            else:
                _store_fn    = candidate.sink_fn
                _store_sizes = _arg_sizes if _arg_sizes else [1]
            rule = {
                "sink":          candidate.vuln_type in (
                                     "buffer_overflow", "format_string",
                                     "command_injection", "integer_overflow",
                                     "unknown_call"),
                "sink_type":     candidate.vuln_type,
                "is_sink":       True,
                "confidence":    _conf,
                "return_tainted": False,
                "external_input": [],
                "size_arg":      -1,
                "notes": (
                    f"LLM-confirmed [{candidate.vuln_type}] "
                    f"via {self.provider}/{self.model} "
                    f"(conf={_conf:.0%}). "
                    f"Sink: {candidate.sink_fn}. "
                    f"{parsed.get('reasoning', '')[:100]}"
                ),
            }
            try:
                # Store fingerprint for cross-binary transfer
                if fp:
                    try:
                        self._pattern_store.store_fingerprint(
                            fingerprint  = fp,
                            confirmed    = True,
                            confidence   = _conf,
                            vuln_type    = candidate.vuln_type,
                            example_func = candidate.func_name,
                        )
                    except Exception:
                        pass  # fingerprint storage is best-effort
                self._pattern_store.store_structural(
                    _store_fn,
                    _store_sizes,
                    rule,
                )
                log.info(
                    "Pattern stored: [%s] %s args=%s conf=%.0f%%",
                    candidate.vuln_type, _store_fn, _store_sizes, _conf*100,
                )
            except Exception as exc:
                log.debug("Pattern store write failed: %s", exc)

        # Item 3: calibrated confidence ensemble replaces raw LLM score
        llm_conf = float(parsed.get("confidence", 0.5))
        cal_conf, cal_breakdown = calibrate_finding_confidence(
            llm_conf      = llm_conf,
            candidate     = candidate,
            pattern_store = self._pattern_store,
        )
        log.debug("Calibrated confidence for %s: %.2f (raw LLM: %.2f)",
                  candidate.func_name, cal_conf, llm_conf)

        return Finding(
            func_name    = candidate.func_name,
            entry_addr   = candidate.entry_addr,
            vuln_type    = candidate.vuln_type,
            sink_fn      = candidate.sink_fn,
            op_seq       = candidate.op_seq,
            taint_source = candidate.taint_source,
            taint_path   = candidate.taint_path,
            confirmed    = confirmed,
            severity     = parsed.get("severity", "medium"),
            reasoning    = parsed.get("reasoning", ""),
            exploit_condition    = parsed.get("exploit_condition", ""),
            false_positive_reason = parsed.get("false_positive_reason", ""),
            confidence   = cal_conf,
            calibration  = cal_breakdown.as_dict(),
            model_used   = f"{self.provider}/{self.model}",
            analysis_time_s = round(elapsed, 3),
        )

    def review_all(
        self,
        taint_results: list[TaintResult],
        func_map:      dict[str, dict],
        delay_s:       float = 10.0,
    ) -> list[Finding]:
        """
        Review all VulnCandidates across all taint results.

        Three-tier decision for each candidate:

        Tier 1 — LIBRARY_MATCH (hardcoded rule, confidence=1.0)
          The pattern is a known libc function like strcpy or recv.
          The rule is a universal truth — no LLM needed.
          → auto-confirm immediately, zero API cost.

        Tier 2 — STRUCTURAL_MATCH (LLM-inferred rule, stored in DB)
          The pattern was analyzed by LLM previously and stored.
          The rule is already reasoned — no new LLM call needed.
          → auto-confirm with stored confidence.

        Tier 3 — NO_MATCH (unknown function)
          We do not know what this function does.
          → send to LLM for reasoning.

        Parameters
        ----------
        taint_results : list of TaintResult from taint engine
        func_map      : func_name → raw func dict (for P-code context)
        delay_s       : delay between LLM calls (rate limiting)
        """
        all_candidates = [
            (result, candidate)
            for result in taint_results
            for candidate in result.vulns
        ]

        # Round-robin ordering: give every function its best shot before any
        # function gets a second LLM call. Assign each candidate its rank within
        # its function (0 = first/best, 1 = second, …), then sort by
        # (LIBRARY_MATCH first, rank ASC, confidence DESC). This guarantees
        # png_safe_execute's single candidate is reviewed before the 4th candidate
        # of png_set_quantize, even if png_set_quantize has higher confidence.
        _fn_seen: dict[str, int] = {}
        # Sort by confidence first so rank 0 = highest-confidence per function
        _by_conf = sorted(all_candidates, key=lambda x: -x[1].confidence)
        _ranked: list[tuple[int, tuple]] = []
        for item in _by_conf:
            fn = item[1].func_name
            r = _fn_seen.get(fn, 0)
            _fn_seen[fn] = r + 1
            _ranked.append((r, item))

        _ranked.sort(key=lambda x: (
            0 if getattr(x[1][1], "match_kind", "LIBRARY_MATCH") == "LIBRARY_MATCH" else 1,
            x[0],              # round number (0 = first pick per function)
            -x[1][1].confidence,
        ))
        all_candidates = [item for _, item in _ranked]

        total      = len(all_candidates)
        auto_count = 0
        llm_count  = 0
        llm_budget = 40   # max LLM calls per binary
        _llm_per_fn: dict[str, int] = {}   # per-function LLM call counter
        _MAX_LLM_PER_FN = 4               # cap: no single function uses >4 budget slots
        log.info("Reviewing %d vulnerability candidates …", total)

        findings: list[Finding] = []
        for i, (result, candidate) in enumerate(all_candidates, 1):
            func = func_map.get(candidate.func_name, {})

            # ── Tier 1: auto-confirm LIBRARY_MATCH only ───────────────
            # STRUCTURAL_MATCH (cross-binary structural patterns) routes to LLM:
            # structural patterns have lower confidence and have caused codec FPs
            # where arg-size fingerprints matched unrelated functions in other libs.
            match_kind = getattr(candidate, "match_kind", "LIBRARY_MATCH")

            if match_kind == "LIBRARY_MATCH":
                # Stage 3D: allocators require arithmetic evidence — they are
                # not vulnerabilities on their own.  malloc() reached by taint
                # is a signal, not a finding; the LLM must verify the arithmetic.
                # Any sink_fn that is (or contains) an allocator name is demoted
                # to NO_MATCH so it goes through LLM review below.
                _ALLOC_FRAGS = frozenset({
                    "malloc", "calloc", "realloc", "alloc",
                    "valloc", "mmap", "brk",
                })
                _sink_lower = (candidate.sink_fn or "").lower()
                _is_alloc_sink = any(f in _sink_lower for f in _ALLOC_FRAGS)
                if _is_alloc_sink:
                    # Demote: fall through to LLM review (handled as NO_MATCH)
                    match_kind = "NO_MATCH"
                else:
                    auto_count += 1
                    self._confirmed += 1
                    finding = Finding(
                        func_name    = candidate.func_name,
                        entry_addr   = candidate.entry_addr,
                        vuln_type    = candidate.vuln_type,
                        sink_fn      = candidate.sink_fn,
                        op_seq       = candidate.op_seq,
                        taint_source = candidate.taint_source,
                        taint_path   = candidate.taint_path,
                        confirmed    = True,
                        severity     = self._default_severity(candidate.vuln_type),
                        reasoning    = (
                            f"Pattern-confirmed: {candidate.sink_fn} is a known "
                            f"{candidate.vuln_type} sink. {candidate.description}"
                        ),
                        exploit_condition    = self._default_exploit(candidate.vuln_type),
                        false_positive_reason = "",
                        confidence   = candidate.confidence,
                        model_used   = f"pattern_store/{match_kind}",
                        analysis_time_s = 0.0,
                    )
                    findings.append(finding)
                    log.info(
                        "[%d/%d] AUTO-CONFIRMED  %s — %s  (pattern match, no LLM)",
                        i, total, candidate.func_name, candidate.vuln_type,
                    )
                    continue

            # ── Tier 2.5: check if previously confirmed by LLM ─────
            # Before calling the LLM, look up the structural pattern DB.
            # Key must match what store_structural wrote:
            #   unknown_call  → lookup(sink_fn, arg_sizes) or lookup_by_pattern
            #   write_what_where / null_dereference → lookup(vuln_type, [path_len])
            if self._pattern_store is not None and match_kind == "NO_MATCH":
                _c_arg_sizes = list(getattr(candidate, "arg_sizes", []) or [])
                _cached = None
                if candidate.vuln_type == "unknown_call" and _c_arg_sizes:
                    # Try exact fn_name match first, then fingerprint
                    _cached = (self._pattern_store.lookup(candidate.sink_fn, _c_arg_sizes)
                               or self._pattern_store.lookup_by_pattern(_c_arg_sizes, None))
                elif candidate.vuln_type in ("write_what_where", "null_dereference"):
                    _cached = self._pattern_store.lookup(
                        candidate.vuln_type, [len(candidate.taint_path)])
                if _cached is not None and _cached.get("is_sink", False):
                    auto_count += 1
                    self._confirmed += 1
                    _cc = _cached.get("confidence", 0.7)
                    finding = Finding(
                        func_name    = candidate.func_name,
                        entry_addr   = candidate.entry_addr,
                        vuln_type    = candidate.vuln_type,
                        sink_fn      = candidate.sink_fn,
                        op_seq       = candidate.op_seq,
                        taint_source = candidate.taint_source,
                        taint_path   = candidate.taint_path,
                        confirmed    = True,
                        severity     = self._default_severity(candidate.vuln_type),
                        reasoning    = (
                            f"Previously confirmed by LLM: [{candidate.vuln_type}] "
                            f"{_cached.get('notes','')[:100]}"
                        ),
                        exploit_condition    = self._default_exploit(candidate.vuln_type),
                        false_positive_reason = "",
                        confidence   = _cc,
                        model_used   = "pattern_store/STRUCTURAL_MATCH",
                        analysis_time_s = 0.0,
                    )
                    findings.append(finding)
                    log.info(
                        "[%d/%d] AUTO-CONFIRMED (cached)  %s — %s  "
                        "(structural pattern, no LLM)",
                        i, total, candidate.func_name, candidate.vuln_type,
                    )
                    continue

            # ── Tier 2.5b: fingerprint-based structural match ───────
            # Check if we have seen this exact structural pattern before
            # (same vuln_type + taint_origin + sink_class + confidence)
            # This enables cross-binary transfer: pattern learned in libpng
            # auto-confirms in libsndfile without LLM.
            # STRUCTURAL_MATCH candidates skip fingerprint auto-confirm:
            # they go to LLM to avoid false confirmations from coarse fingerprints.
            fp = getattr(candidate, "fingerprint", "") or ""
            if fp and self._pattern_store is not None and match_kind != "STRUCTURAL_MATCH":
                try:
                    fp_result = self._pattern_store.lookup_fingerprint(fp)
                except Exception:
                    fp_result = None
                if fp_result and fp_result.get("confirmed"):
                    _fp_conf = max(candidate.confidence,
                                   fp_result.get("confidence", 0.7))
                    finding = Finding(
                        func_name    = candidate.func_name,
                        entry_addr   = candidate.entry_addr,
                        vuln_type    = candidate.vuln_type,
                        sink_fn      = candidate.sink_fn,
                        op_seq       = candidate.op_seq,
                        taint_source = candidate.taint_source,
                        taint_path   = candidate.taint_path,
                        confirmed    = True,
                        severity     = self._default_severity(candidate.vuln_type),
                        reasoning    = (
                            f"Fingerprint-confirmed: cross-binary pattern {fp}. "
                            f"First seen in: {fp_result.get('example_func', '?')}"
                        ),
                        exploit_condition     = self._default_exploit(candidate.vuln_type),
                        false_positive_reason = "",
                        confidence   = _fp_conf,
                        model_used   = "pattern_store/FINGERPRINT_MATCH",
                        analysis_time_s = 0.0,
                    )
                    findings.append(finding)
                    auto_count += 1
                    self._confirmed += 1
                    log.info("[%d/%d] AUTO-CONFIRMED (fingerprint)  %s — %s "
                             "(cross-binary pattern, no LLM)",
                             i, total, candidate.func_name,
                             candidate.vuln_type)
                    continue

            # ── Tier 3: unknown — send to LLM (if enabled + budget) ──

            # Pre-filter: skip low-confidence candidates without LLM call.
            # Candidates below this threshold have <30% LLM confirm rate
            # and mostly represent weak taint paths or over-broad sinks.
            _MIN_LLM_CONF = 0.50
            if candidate.confidence < _MIN_LLM_CONF:
                log.debug(
                    "[%d/%d] SKIPPED (conf=%.0f%% < %.0f%%) — %s — %s",
                    i, total, candidate.confidence * 100, _MIN_LLM_CONF * 100,
                    candidate.func_name, candidate.vuln_type,
                )
                continue

            fn_llm_used = _llm_per_fn.get(candidate.func_name, 0)
            if fn_llm_used >= _MAX_LLM_PER_FN:
                log.debug("Per-function LLM cap (%d) reached — skipping %s",
                          _MAX_LLM_PER_FN, candidate.func_name)
                continue

            if llm_count >= llm_budget:
                log.debug("LLM budget exhausted (%d/%d) — skipping %s",
                          llm_count, llm_budget, candidate.func_name)
                continue
            if not self.llm_enabled:
                log.info(
                    "[%d/%d] SKIPPED (LLM disabled) — %s — %s  (NO_MATCH, unreviewed)",
                    i, total, candidate.func_name, candidate.vuln_type,
                )
                continue

            llm_count += 1
            _llm_per_fn[candidate.func_name] = _llm_per_fn.get(candidate.func_name, 0) + 1
            _mk_label = {"NO_MATCH": "unknown", "STRUCTURAL_MATCH": "structural"}.get(
                match_kind, match_kind
            )
            log.info(
                "[%d/%d] %s — %s  (%s → LLM)",
                i, total, candidate.func_name, candidate.vuln_type, _mk_label,
            )

            # Proactive delay before LLM call — free models: 20 req/min = 3s min.
            time.sleep(1)
            finding = self.review(candidate, func, taint_result=result)
            if finding:
                findings.append(finding)
                status = "CONFIRMED" if finding.confirmed else "rejected"
                log.info("  → %s  (conf=%.0f%%)", status, finding.confidence * 100)

            if llm_count > 0:
                time.sleep(delay_s)

        log.info(
            "Done — auto-confirmed=%d  llm-reviewed=%d  confirmed=%d  rejected=%d  errors=%d",
            auto_count, llm_count, self._confirmed, self._rejected, self._errors,
        )

        # ── Post-processing: deduplicate + cap ────────────────────────
        # 1. Deduplicate confirmed findings: same function + vuln_type + sink_fn
        #    appearing from multiple taint paths → keep highest-confidence one.
        seen: dict[tuple, "Finding"] = {}
        deduped: list["Finding"] = []
        for f in findings:
            if not f.confirmed:
                deduped.append(f)
                continue
            key = (f.func_name, f.vuln_type, f.sink_fn or "")
            if key not in seen or f.confidence > seen[key].confidence:
                seen[key] = f
        deduped.extend(seen.values())

        n_dedup = len(findings) - len(deduped)
        if n_dedup > 0:
            log.info("Deduplication removed %d duplicate confirmed finding(s)", n_dedup)

        # 2. Per-binary cap: if more than MAX confirmed findings, keep top by confidence.
        #    Prevents a single "leaky" binary from dominating FP count.
        _MAX_CONFIRMED = 500
        confirmed   = sorted(
            [f for f in deduped if f.confirmed],
            key=lambda f: f.confidence,
            reverse=True,
        )
        unconfirmed = [f for f in deduped if not f.confirmed]
        if len(confirmed) > _MAX_CONFIRMED:
            log.info(
                "Per-binary cap: trimming %d confirmed findings to top %d by confidence",
                len(confirmed), _MAX_CONFIRMED,
            )
            confirmed = confirmed[:_MAX_CONFIRMED]

        return confirmed + unconfirmed

    @staticmethod
    def _default_severity(vuln_type: str) -> str:
        return {
            "buffer_overflow":   "high",
            "command_injection":  "critical",
            "format_string":      "high",
            "write_what_where":   "critical",
            "integer_overflow":   "medium",
            "use_after_free":     "high",
        }.get(vuln_type, "medium")

    @staticmethod
    def _default_exploit(vuln_type: str) -> str:
        return {
            "buffer_overflow":   "Send input longer than destination buffer capacity.",
            "command_injection":  "Control the string passed to system/popen/execve.",
            "format_string":      "Pass format specifiers (%x, %n) as user input.",
            "write_what_where":   "Control the write address through tainted pointer arithmetic.",
            "integer_overflow":   "Send a size value that wraps around to a small number.",
            "use_after_free":     "Trigger free then reuse the freed pointer.",
        }.get(vuln_type, "Provide crafted input to trigger the vulnerability.")

    def stats(self) -> dict:
        return {
            "confirmed": self._confirmed,
            "rejected":  self._rejected,
            "errors":    self._errors,
            "total":     self._confirmed + self._rejected + self._errors,
        }

    # ── Context builders ──────────────────────────────────────────────

    def _pcode_context(self, func: dict, sink_seq: int) -> str:
        """
        Extract P-code ops around the sink for LLM context.
        Shows N ops before and after the sink.
        """
        ops = func.get("ops") or []
        if not ops:
            return "(no ops available)"

        half    = self.context_ops // 2
        start   = max(0, sink_seq - half)
        end     = min(len(ops), sink_seq + half)
        window  = [o for o in ops if start <= o.get("seq", 0) <= end]

        lines = []
        for op in window:
            seq    = op.get("seq", "?")
            mnem   = op.get("op", "?")
            out    = op.get("output")
            inputs = op.get("inputs") or []

            out_str = out["name"] if isinstance(out, dict) and out else "_"
            inp_str = ", ".join(
                i["name"] for i in inputs
                if isinstance(i, dict) and i.get("name")
            )

            marker = " ← SINK" if seq == sink_seq else ""
            lines.append(
                f"  seq={str(seq):<4} {mnem:<12} {out_str}  ←  {inp_str}{marker}"
            )

        return "\n".join(lines) if lines else "(no ops in range)"

    def _flow_context(self, taint_result, candidate) -> str:
        """
        Build the full taint flow log as a string for LLM context.
        Much more informative than the raw P-code window.
        """
        steps = getattr(taint_result, "flow_steps", None) or []
        if not steps:
            return "(no flow log available)"

        lines = ["  seq    op            from               to              reason"]
        lines.append("  " + "─" * 70)
        for step in steps:
            mem_tag  = "[*mem]" if step.is_mem else "      "
            from_str = step.from_var if step.from_var else "⊕ SEED"
            lines.append(
                f"  {str(step.seq):<6} {step.op:<13} {from_str:<18} "
                f"{step.to_var:<14}{mem_tag}  {step.reason}"
            )

        # Highlight the path to this specific sink
        path = candidate.taint_path
        if path:
            lines.append("")
            lines.append(f"  Path to sink: {' → '.join(path)}")

        return "\n".join(lines)

    def _pattern_summary(self, func: dict) -> str:
        """
        Summarize the call sites in this function for LLM context.
        """
        call_sites = func.get("call_sites") or []
        if not call_sites:
            return "(no call sites)"
        return ", ".join(call_sites[:10])


    def _callee_context(self, func: dict) -> str:
        """
        Build callee context: for each function called by this function,
        show its learned semantic role if known.
        This tells the LLM what the callees do — critical for deciding
        whether a taint path is dangerous.
        """
        ops = func.get("ops") or []
        callee_names = []
        seen = set()
        for op in ops:
            if op.get("op") not in ("CALL", "CALLIND"):
                continue
            inputs = op.get("inputs") or []
            if not inputs:
                continue
            name = (inputs[0].get("name","") if isinstance(inputs[0], dict)
                    else str(inputs[0]))
            if name and name not in seen and not name.startswith("ram("):
                callee_names.append(name)
                seen.add(name)

        if not callee_names:
            return "    (no callee information available)"

        lines = []
        for callee in callee_names[:8]:  # limit to 8 callees
            # Check learned summary
            summary = None
            if self._pattern_store is not None:
                try:
                    summary = self._pattern_store.get_learned_summary(callee, [])
                except Exception:
                    pass

            if summary:
                role  = summary.get("likely_role", "?")
                conf  = summary.get("confidence", 0)
                notes = ""
                if summary.get("allocation"):
                    notes += " [allocates memory]"
                if summary.get("writes_memory"):
                    notes += " [writes memory]"
                if summary.get("external_input"):
                    notes += f" [reads external data at args {summary.get('external_input')}]"
                lines.append(f"    {callee:<40} role={role} conf={conf:.0%}{notes}")
            else:
                lines.append(f"    {callee:<40} (unknown role)")

        return "\n".join(lines) if lines else "    (no callee information)"

    def _caller_context(self, func_name: str, all_funcs: list) -> str:
        """
        Find functions that call this one and list their names.
        Gives the LLM context about where this function is called from
        — important for understanding the call graph neighborhood.
        """
        callers = []
        for f in (all_funcs or []):
            ops = f.get("ops") or []
            for op in ops:
                if op.get("op") not in ("CALL", "CALLIND"):
                    continue
                inputs = op.get("inputs") or []
                if not inputs:
                    continue
                callee = (inputs[0].get("name","") if isinstance(inputs[0], dict)
                          else str(inputs[0]))
                if callee == func_name:
                    callers.append(f.get("name","?"))
                    break
            if len(callers) >= 5:
                break

        if not callers:
            return "    (no callers found — may be an exported handler or entry point)"
        return "    " + ", ".join(callers[:5])

    def _string_context(self, func: dict) -> str:
        """
        Extract string references from the function.
        Strings like "POST /login", "Authorization:", "malloc failed"
        instantly reveal the function's purpose to the LLM.
        """
        strings = func.get("strings") or []
        if not strings:
            # Try extracting from ops — constants that look like addresses
            ops = func.get("ops") or []
            candidates = []
            for op in ops:
                for inp in (op.get("inputs") or []):
                    if isinstance(inp, dict):
                        name = inp.get("name","")
                        if name.startswith("const(0x") and len(name) > 10:
                            candidates.append(name)
            if not candidates:
                return "    (no strings found)"
            return "    " + ", ".join(candidates[:5])

        lines = []
        for s in strings[:6]:
            if isinstance(s, str):
                lines.append(f'    "{s[:60]}"')
            elif isinstance(s, dict):
                val = s.get("value","") or s.get("string","")
                lines.append(f'    "{str(val)[:60]}"')
        return "\n".join(lines) if lines else "    (no strings found)"

    # ── LLM call ──────────────────────────────────────────────────────

    def _call_llm(self, user_message: str, func_name: str) -> Optional[str]:
        """Call the configured LLM provider with retry. Records cost to GLOBAL_TRACKER."""
        from llm_cost_tracker import GLOBAL_TRACKER
        t_start = time.perf_counter()

        if self.provider == "groq":
            text, usage = self._call_groq_with_usage(user_message, func_name)
        elif self.provider == "openrouter":
            text, usage = self._call_openrouter_with_usage(user_message, func_name)
        elif self.provider == "gemini":
            text, usage = self._call_gemini_with_usage(user_message, func_name)
        elif self.provider == "anthropic":
            text, usage = self._call_anthropic_with_usage(user_message, func_name)
        else:
            log.error("Unknown provider: %s", self.provider)
            return None

        elapsed = time.perf_counter() - t_start
        if text is not None:
            in_tok  = usage.get("input_tokens")  or usage.get("prompt_tokens") or _estimate_tokens_ra(user_message)
            out_tok = usage.get("output_tokens") or usage.get("completion_tokens") or _estimate_tokens_ra(text)
            GLOBAL_TRACKER.record(
                stage         = "reasoning",
                model         = self.model,
                input_tokens  = int(in_tok),
                output_tokens = int(out_tok),
                latency_s     = elapsed,
                fn_name       = func_name,
            )
        return text

    def _call_groq_with_usage(self, user_message: str, func_name: str):
        """Returns (text, usage_dict) tuple."""
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")

        client = Groq(api_key=self.api_key)
        waits  = [65, 130]
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model    = self.model,
                    messages = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                )
                usage = {}
                if r.usage:
                    usage = {
                        "input_tokens":  r.usage.prompt_tokens,
                        "output_tokens": r.usage.completion_tokens,
                    }
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

    def _call_openrouter_with_usage(self, user_message: str, func_name: str):
        """
        Call any model via OpenRouter (https://openrouter.ai).
        Returns (text, usage_dict) tuple.
        """
        import urllib.request, json as _json

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/binary-vuln-pipeline",
            "X-Title":       "Binary Vulnerability Analysis Pipeline",
        }
        payload = _json.dumps({
            "model":       self.model,
            "max_tokens":  1000,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        }).encode()

        for attempt in range(3):
            try:
                req  = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=payload, headers=headers, method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body  = _json.loads(resp.read())
                    text  = body["choices"][0]["message"]["content"]
                    usage_raw = body.get("usage", {})
                    usage = {
                        "input_tokens":  usage_raw.get("prompt_tokens", 0),
                        "output_tokens": usage_raw.get("completion_tokens", 0),
                    }
                    return text, usage
            except urllib.error.HTTPError as e:
                body_text = e.read().decode(errors="replace")
                if e.code == 429:
                    wait = 60 * (attempt + 1)
                    log.warning("OpenRouter rate limited — waiting %ds", wait)
                    time.sleep(wait)
                elif e.code in (401, 403):
                    log.error("OpenRouter auth error: %s", body_text[:200])
                    return None, {}
                elif e.code == 402:
                    log.error("OpenRouter: insufficient credits. Add credits at openrouter.ai")
                    return None, {}
                else:
                    log.error("OpenRouter HTTP %d for %s: %s", e.code, func_name, body_text[:200])
                    if attempt == 2:
                        return None, {}
                    time.sleep(5)
            except Exception as exc:
                log.error("OpenRouter error for %s: %s", func_name, exc)
                if attempt == 2:
                    return None, {}
                time.sleep(5)
        return None, {}

    def _call_gemini_with_usage(self, user_message: str, func_name: str):
        """Gemini via REST. Returns (text, usage_dict) tuple."""
        import urllib.request, urllib.error, json as _json

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = _json.dumps({
            "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.0},
        }).encode()

        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body  = _json.loads(resp.read())
                    text  = body["candidates"][0]["content"]["parts"][0]["text"]
                    meta  = body.get("usageMetadata", {})
                    usage = {
                        "input_tokens":  meta.get("promptTokenCount", 0),
                        "output_tokens": meta.get("candidatesTokenCount", 0),
                    }
                    return text, usage
            except urllib.error.HTTPError as e:
                body_text = e.read().decode(errors="replace")
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    log.warning("Gemini rate limited — waiting %ds", wait)
                    time.sleep(wait)
                elif e.code in (400, 403, 404):
                    log.warning("Gemini %d for %s — trying Groq fallback", e.code, func_name)
                    text = self._fallback_groq(user_message, func_name)
                    return text, {}
                else:
                    log.error("Gemini HTTP %d for %s", e.code, func_name)
                    if attempt == 2:
                        text = self._fallback_groq(user_message, func_name)
                        return text, {}
                    time.sleep(5)
            except Exception as exc:
                log.error("Gemini error for %s: %s", func_name, exc)
                if attempt == 2:
                    text = self._fallback_groq(user_message, func_name)
                    return text, {}
                time.sleep(3)
        text = self._fallback_groq(user_message, func_name)
        return text, {}

    def _fallback_groq(self, user_message: str, func_name: str) -> Optional[str]:
        """
        LLM cascade: when primary provider (Gemini) fails, try Groq free tier.
        Called automatically — no config needed.
        This is the key resilience feature: two free APIs, zero cost.
        """
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key or self.provider == "groq":
            return None  # no fallback available or already on Groq

        log.info("Primary provider failed — falling back to Groq for %s", func_name)
        try:
            from groq import Groq
            client   = Groq(api_key=groq_key)
            waits    = [65, 130]
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model       = "llama-3.3-70b-versatile",
                        messages    = [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        max_tokens  = 1024,
                        temperature = 0.0,
                    )
                    log.info("Groq fallback succeeded for %s", func_name)
                    return r.choices[0].message.content
                except Exception as e:
                    err = str(e)
                    if ("429" in err or "rate" in err.lower()) and attempt < 2:
                        log.warning("Groq fallback rate limited — waiting %ds",
                                    waits[attempt])
                        time.sleep(waits[attempt])
                    else:
                        log.error("Groq fallback error for %s: %s", func_name, e)
                        return None
        except ImportError:
            log.warning("Groq fallback unavailable — run: pip install groq")
        return None

    def _call_anthropic_with_usage(self, user_message: str, func_name: str):
        """Returns (text, usage_dict) tuple."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            r = client.messages.create(
                model      = self.model,
                max_tokens = 1024,
                system     = _SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_message}],
            )
            usage = {}
            if r.usage:
                usage = {
                    "input_tokens":  r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                }
            return r.content[0].text, usage
        except Exception as e:
            log.error("Anthropic error for %s: %s", func_name, e)
            return None, {}

    # ── Response parsing ──────────────────────────────────────────────

    def _parse(self, raw: str) -> Optional[dict]:
        """Parse LLM JSON response with defensive handling."""
        text = raw.strip()

        # Strip markdown fences
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        # Find JSON object boundaries
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning("No JSON in response")
            return None

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            log.warning("JSON parse error: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    findings:       list[Finding],
    taint_results:  list[TaintResult],
    output_path:    str | Path = "vulnerability_report.txt",
) -> None:
    """
    Write a complete vulnerability report to a text file.
    Confirmed findings first, then rejected, then statistics.
    """
    output_path = Path(output_path)
    confirmed = [f for f in findings if f.confirmed]
    rejected  = [f for f in findings if not f.confirmed]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("VULNERABILITY ANALYSIS REPORT\n")
        fh.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        fh.write(f"Functions : {len(taint_results)}\n")
        fh.write(f"Candidates: {sum(len(r.vulns) for r in taint_results)}\n")
        fh.write(f"Confirmed : {len(confirmed)}\n")
        fh.write(f"Rejected  : {len(rejected)}\n")
        fh.write("\n" + "═" * 62 + "\n")

        if confirmed:
            fh.write("\nCONFIRMED VULNERABILITIES\n")
            fh.write("═" * 62 + "\n\n")
            for finding in sorted(confirmed, key=lambda f: (
                {"critical":0,"high":1,"medium":2,"low":3,"info":4}
                .get(f.severity, 5)
            )):
                fh.write(finding.report_block())
                fh.write("\n\n")
        else:
            fh.write("\nNo confirmed vulnerabilities found.\n\n")

        if rejected:
            fh.write("─" * 62 + "\n")
            fh.write("REJECTED CANDIDATES (false positives)\n")
            fh.write("─" * 62 + "\n\n")
            for finding in rejected:
                fh.write(
                    f"  ✗  {finding.vuln_type}  in  {finding.func_name}"
                    f"  (seq {finding.op_seq})\n"
                )
                fh.write(f"     Reason: {finding.false_positive_reason}\n\n")

        fh.write("═" * 62 + "\n")
        fh.write("TAINT ENGINE STATISTICS\n")
        fh.write("═" * 62 + "\n")
        for r in sorted(taint_results, key=lambda x: -len(x.vulns)):
            if r.vulns or r.unknown_calls:
                fh.write(
                    f"  {r.func_name:<40} "
                    f"vulns={len(r.vulns)}  unknown_calls={r.calls_unknown}\n"
                )

    log.info("Report written → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# HTML report generator
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_COLORS = {
    "critical": "#c0392b",
    "high":     "#e67e22",
    "medium":   "#d4ac0d",
    "low":      "#2980b9",
    "info":     "#7f8c8d",
}

_SEVERITY_BG = {
    "critical": "#fadbd8",
    "high":     "#fdebd0",
    "medium":   "#fef9e7",
    "low":      "#d6eaf8",
    "info":     "#f2f3f4",
}

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def generate_html_report(
    findings:       list["Finding"],
    taint_results:  list["TaintResult"],
    output_path:    "str | Path" = "vulnerability_report.html",
    binary_name:    str = "",
) -> None:
    """
    Write a self-contained HTML vulnerability report.

    Features:
    - Inline CSS only — no external CDN, no JavaScript
    - HTML5 <details>/<summary> for collapsible sections
    - Color-coded rows and severity badges per finding
    - Critical/high open by default; medium/low/info collapsed
    - Taint paths rendered as arrow chains
    """
    from datetime import datetime
    output_path = Path(output_path)

    confirmed  = [f for f in findings if f.confirmed]
    rejected   = [f for f in findings if not f.confirmed]
    confirmed.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 5))

    total_cands = sum(len(r.vulns) for r in taint_results)

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def badge(severity: str) -> str:
        color = _SEVERITY_COLORS.get(severity, "#888")
        return (
            f'<span style="background:{color};color:#fff;'
            f'padding:2px 8px;border-radius:4px;font-size:0.85em;'
            f'font-weight:bold;text-transform:uppercase;">{esc(severity)}</span>'
        )

    def taint_chain(path: list) -> str:
        parts = [f'<code style="background:#eee;padding:1px 4px;border-radius:3px;">{esc(p)}</code>'
                 for p in path]
        return ' <span style="color:#888;">→</span> '.join(parts)

    lines: list[str] = []

    # ── HEAD ──────────────────────────────────────────────────────────
    lines.append("<!DOCTYPE html>")
    lines.append('<html lang="en"><head><meta charset="UTF-8">')
    lines.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    lines.append(f'<title>Vulnerability Report — {esc(binary_name or "binary")}</title>')
    lines.append("<style>")
    lines.append("""
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
       background: #f5f5f5; color: #333; }
.container { max-width: 1100px; margin: 0 auto; }
h1 { color: #2c3e50; border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }
h2 { color: #2c3e50; margin-top: 32px; }
.summary-box { background: #fff; border-radius: 8px; padding: 16px 24px;
               box-shadow: 0 1px 4px rgba(0,0,0,.12); margin-bottom: 24px;
               display: flex; gap: 32px; flex-wrap: wrap; }
.stat { text-align: center; }
.stat .val { font-size: 2em; font-weight: bold; color: #2c3e50; }
.stat .lbl { font-size: 0.85em; color: #666; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,.12); margin-bottom: 24px; }
th { background: #2c3e50; color: #fff; padding: 10px 14px; text-align: left;
     font-size: 0.9em; }
td { padding: 9px 14px; border-bottom: 1px solid #eee; font-size: 0.9em; }
tr:last-child td { border-bottom: none; }
.finding { background: #fff; border-radius: 8px; margin-bottom: 16px;
           box-shadow: 0 1px 4px rgba(0,0,0,.12); overflow: hidden; }
.finding-header { padding: 12px 18px; font-weight: bold;
                  display: flex; align-items: center; gap: 10px; cursor: pointer; }
.finding-body { padding: 16px 18px; border-top: 1px solid #eee; }
.field-label { font-weight: bold; color: #555; min-width: 140px;
               display: inline-block; font-size: 0.9em; }
.field-row { margin-bottom: 8px; font-size: 0.9em; }
.reasoning { background: #f8f9fa; border-left: 4px solid #2c3e50;
             padding: 10px 14px; border-radius: 0 4px 4px 0;
             margin: 10px 0; font-size: 0.9em; line-height: 1.6; }
details > summary { list-style: none; cursor: pointer; }
details > summary::-webkit-details-marker { display: none; }
.pcode-block { background: #1e1e1e; color: #d4d4d4; font-family: monospace;
               font-size: 0.85em; padding: 12px; border-radius: 4px;
               overflow-x: auto; white-space: pre; margin-top: 8px; }
.rejected-list { background: #fff; border-radius: 8px; padding: 16px 24px;
                 box-shadow: 0 1px 4px rgba(0,0,0,.12); }
.rejected-item { padding: 6px 0; border-bottom: 1px solid #eee;
                 font-size: 0.9em; color: #555; }
.rejected-item:last-child { border-bottom: none; }
.taint-stats { background: #fff; border-radius: 8px; padding: 16px 24px;
               box-shadow: 0 1px 4px rgba(0,0,0,.12); font-size: 0.88em; }
.taint-row { display: flex; gap: 16px; padding: 4px 0;
             border-bottom: 1px solid #eee; }
.taint-row:last-child { border-bottom: none; }
footer { margin-top: 40px; color: #999; font-size: 0.8em; text-align: center; }
""")
    lines.append("</style></head><body><div class='container'>")

    # ── HEADER ────────────────────────────────────────────────────────
    title = f"Vulnerability Report — {esc(binary_name)}" if binary_name else "Vulnerability Report"
    lines.append(f"<h1>{title}</h1>")
    lines.append(
        f'<p style="color:#666;font-size:0.9em;">Generated: '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp; '
        f'Analysis tool: pcode_extractor</p>'
    )

    # ── SUMMARY BOX ───────────────────────────────────────────────────
    lines.append('<div class="summary-box">')
    for val, lbl in [
        (len(taint_results), "Functions Analyzed"),
        (total_cands,        "Candidates"),
        (len(confirmed),     "Confirmed"),
        (len(rejected),      "Rejected (FP)"),
    ]:
        lines.append(
            f'<div class="stat"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>'
        )
    lines.append("</div>")

    # ── SUMMARY TABLE ─────────────────────────────────────────────────
    if confirmed:
        lines.append("<h2>Confirmed Vulnerabilities</h2>")
        lines.append("<table>")
        lines.append(
            "<tr><th>#</th><th>Function</th><th>Type</th><th>Severity</th>"
            "<th>Sink</th><th>Confidence</th></tr>"
        )
        for i, f in enumerate(confirmed, 1):
            bg = _SEVERITY_BG.get(f.severity, "#fff")
            lines.append(
                f'<tr style="background:{bg};">'
                f"<td>{i}</td>"
                f"<td><code>{esc(f.func_name)}</code> @ {esc(f.entry_addr)}</td>"
                f"<td>{esc(f.vuln_type)}</td>"
                f"<td>{badge(f.severity)}</td>"
                f"<td><code>{esc(f.sink_fn)}</code></td>"
                f"<td>{f.confidence:.0%}</td>"
                "</tr>"
            )
        lines.append("</table>")

    # ── PER-FINDING DETAIL ────────────────────────────────────────────
        lines.append("<h2>Finding Details</h2>")
        for i, f in enumerate(confirmed, 1):
            color    = _SEVERITY_COLORS.get(f.severity, "#888")
            bg       = _SEVERITY_BG.get(f.severity, "#fff")
            is_open  = f.severity in ("critical", "high")
            open_attr = " open" if is_open else ""

            lines.append(f'<div class="finding">')
            lines.append(
                f'<details{open_attr}>'
                f'<summary class="finding-header" style="background:{bg};'
                f'border-left:5px solid {color};">'
                f'<span style="color:{color};font-size:1.1em;">#{i}</span>'
                f'&nbsp;{badge(f.severity)}&nbsp;'
                f'<span style="color:#2c3e50;">{esc(f.vuln_type)}</span>'
                f'&nbsp;<span style="font-weight:normal;color:#555;">in '
                f'<code>{esc(f.func_name)}</code></span>'
                f'</summary>'
            )
            lines.append('<div class="finding-body">')

            lines.append(f'<div class="field-row"><span class="field-label">Function:</span>'
                         f'<code>{esc(f.func_name)}</code> @ {esc(f.entry_addr)}</div>')
            lines.append(f'<div class="field-row"><span class="field-label">Sink:</span>'
                         f'<code>{esc(f.sink_fn)}</code> (seq {f.op_seq})</div>')
            lines.append(f'<div class="field-row"><span class="field-label">Confidence:</span>'
                         f'{f.confidence:.0%}</div>')
            lines.append(f'<div class="field-row"><span class="field-label">Model:</span>'
                         f'{esc(f.model_used)}</div>')
            lines.append(f'<div class="field-row"><span class="field-label">Taint Path:</span>'
                         f'{taint_chain(f.taint_path)}</div>')

            if f.reasoning:
                lines.append(f'<div class="reasoning">{esc(f.reasoning)}</div>')
            if f.exploit_condition:
                lines.append(f'<div class="field-row"><span class="field-label">'
                             f'Exploit condition:</span>{esc(f.exploit_condition)}</div>')
            if f.false_positive_reason:
                lines.append(f'<div class="field-row"><span class="field-label">'
                             f'FP risk:</span>{esc(f.false_positive_reason)}</div>')

            lines.append('</div></details></div>')

    else:
        lines.append('<p style="color:#27ae60;font-weight:bold;">No confirmed vulnerabilities found.</p>')

    # ── REJECTED ──────────────────────────────────────────────────────
    if rejected:
        lines.append("<h2>Rejected Candidates (False Positives)</h2>")
        lines.append('<details><summary style="cursor:pointer;color:#666;">'
                     f'Show {len(rejected)} rejected candidate(s)</summary>')
        lines.append('<div class="rejected-list">')
        for f in rejected:
            lines.append(
                f'<div class="rejected-item">✗ &nbsp;'
                f'<strong>{esc(f.vuln_type)}</strong> in '
                f'<code>{esc(f.func_name)}</code> — '
                f'{esc(f.false_positive_reason or "no reason given")}</div>'
            )
        lines.append("</div></details>")

    # ── TAINT STATS ───────────────────────────────────────────────────
    interesting = [r for r in taint_results if r.vulns or r.calls_unknown]
    if interesting:
        lines.append("<h2>Taint Engine Statistics</h2>")
        lines.append('<details><summary style="cursor:pointer;color:#666;">'
                     f'Show {len(interesting)} function(s) with findings or unknown calls'
                     '</summary>')
        lines.append('<div class="taint-stats">')
        for r in sorted(interesting, key=lambda x: -len(x.vulns)):
            lines.append(
                f'<div class="taint-row">'
                f'<span style="min-width:300px;"><code>{esc(r.func_name)}</code></span>'
                f'<span>vulns: <strong>{len(r.vulns)}</strong></span>'
                f'<span>unknown calls: {r.calls_unknown}</span>'
                f'<span>ops: {r.ops_analyzed}</span>'
                f'</div>'
            )
        lines.append("</div></details>")

    # ── FOOTER ────────────────────────────────────────────────────────
    lines.append('<footer>Generated by pcode_extractor vulnerability analysis pipeline</footer>')
    lines.append("</div></body></html>")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("HTML report written → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# JSON report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_json_report(
    findings:       list["Finding"],
    taint_results:  list["TaintResult"],
    output_path:    "str | Path" = "vulnerability_report.json",
    binary_name:    str = "",
) -> None:
    """
    Write a structured JSON vulnerability report (pcode-vuln/1.0 schema).
    Machine-readable, suitable for further tooling or SARIF conversion.
    """
    from datetime import datetime, timezone
    output_path = Path(output_path)

    confirmed = [f for f in findings if f.confirmed]
    rejected  = [f for f in findings if not f.confirmed]
    confirmed.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 5))

    report = {
        "schema_version": "pcode-vuln/1.0",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "binary":         binary_name,
        "summary": {
            "functions_analyzed": len(taint_results),
            "total_candidates":   sum(len(r.vulns) for r in taint_results),
            "confirmed":          len(confirmed),
            "rejected":           len(rejected),
        },
        "confirmed_vulnerabilities": [
            {
                "id":                f"VULN-{i:03d}",
                "func_name":         f.func_name,
                "entry_addr":        f.entry_addr,
                "vuln_type":         f.vuln_type,
                "severity":          f.severity,
                "sink_fn":           f.sink_fn,
                "op_seq":            f.op_seq,
                "taint_source":      f.taint_source,
                "taint_path":        f.taint_path,
                "bounded":           False,
                "confidence":        round(f.confidence, 4),
                "reasoning":         f.reasoning,
                "exploit_condition": f.exploit_condition,
                "false_positive_risk": f.false_positive_reason,
                "model_used":        f.model_used,
                "analysis_time_s":   f.analysis_time_s,
            }
            for i, f in enumerate(confirmed, 1)
        ],
        "rejected_candidates": [
            {
                "func_name":             f.func_name,
                "vuln_type":             f.vuln_type,
                "false_positive_reason": f.false_positive_reason,
            }
            for f in rejected
        ],
        "taint_statistics": [
            {
                "func_name":     r.func_name,
                "entry_addr":    r.entry_addr,
                "ops_analyzed":  r.ops_analyzed,
                "vulns_found":   len(r.vulns),
                "calls_matched": r.calls_matched,
                "unknown_calls": r.calls_unknown,
            }
            for r in sorted(taint_results, key=lambda x: -len(x.vulns))
            if r.vulns or r.calls_unknown
        ],
    }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON report written → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pattern_store   import PatternStore
    from pattern_matcher import PatternMatcher

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python reasoning_agent.py <pcode_ranked.jsonl> [limit] [--flow] [--no-llm]")
        print()
        print("  limit            — max candidates to review (default: 10)")
        print("  --flow           — print full variable flow for every function")
        print("  --no-llm         — disable LLM review (auto-confirms LIBRARY_MATCH only)")
        print()
        print("Environment:")
        print("  GROQ_API_KEY     — for Groq (default)")
        print("  GEMINI_API_KEY   — for Gemini")
        print("  ANTHROPIC_API_KEY — for Claude")
        print("  LLM_PROVIDER     — groq | gemini | anthropic")
        sys.exit(1)

    from taint_engine import TaintEngine

    ranked_path   = sys.argv[1]
    show_flow     = "--flow" in sys.argv
    no_llm        = "--no-llm" in sys.argv
    review_limit  = next(
        (int(a) for a in sys.argv[2:] if a.isdigit()), 10
    )

    # Load functions
    funcs: list[dict] = []
    with open(ranked_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("discarded"):
                funcs.append(d)

    print(f"\nLoaded {len(funcs)} functions from {ranked_path}")

    # Build func map for context lookup
    func_map = {f["name"]: f for f in funcs}

    # Run taint analysis
    store   = PatternStore("pattern_store.db")
    matcher = PatternMatcher(store)
    engine  = TaintEngine(matcher)

    print("Running taint analysis …")
    taint_results = engine.analyze_all(funcs)

    total_candidates = sum(len(r.vulns) for r in taint_results)
    print(f"Taint analysis complete — {total_candidates} candidates found\n")

    # Print full variable flow if requested
    if show_flow:
        print("\n" + "═" * 62)
        print("FULL VARIABLE FLOW (all functions)")
        print("═" * 62)
        for r in taint_results:
            r.print_flow()

    if total_candidates == 0:
        print("No vulnerability candidates to review.")
        sys.exit(0)

    # Run reasoning agent
    provider = os.environ.get("LLM_PROVIDER", "groq")
    llm_mode = "warn" if no_llm else "require"
    agent    = ReasoningAgent(provider=provider, llm_mode=llm_mode)
    print(f"Using provider={provider}  model={agent.model}  llm_enabled={agent.llm_enabled}")
    print(f"Reviewing {total_candidates} candidates …\n")

    # Apply limit — take only top candidates by confidence
    all_candidates = [
        (r, v) for r in taint_results for v in r.vulns
    ]
    # Sort by confidence descending, take top N
    all_candidates.sort(key=lambda x: -x[1].confidence)
    if len(all_candidates) > review_limit:
        print(f"Found {len(all_candidates)} candidates — reviewing top {review_limit}")
        # Zero out vulns on results not in top N
        kept = set(id(v) for _, v in all_candidates[:review_limit])
        for r in taint_results:
            r.vulns = [v for v in r.vulns if id(v) in kept]
    else:
        print(f"Found {len(all_candidates)} candidates — reviewing all")

    findings = agent.review_all(taint_results, func_map)

    # Print summary
    confirmed = [f for f in findings if f.confirmed]
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Candidates reviewed : {len(findings)}")
    print(f"  Confirmed vulns     : {len(confirmed)}")
    print(f"  Rejected (FP)       : {len(findings) - len(confirmed)}")
    print(f"  Errors              : {agent.stats()['errors']}")
    print(f"{sep}\n")

    # Print confirmed findings
    if confirmed:
        print("CONFIRMED VULNERABILITIES:\n")
        for f in confirmed:
            print(f.report_block())
            print()

    # Generate report
    stem        = Path(ranked_path).stem.replace("_ranked", "")
    report_path = stem + "_vulnerability_report.txt"
    generate_report(findings, taint_results, report_path)
    print(f"Full report → {report_path}")