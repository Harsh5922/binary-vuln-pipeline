"""
stage4_orthogonal.py  —  Stage 4.5: Orthogonal Semantic Analysis
================================================================
Five CHAINED semantic analyses that run REGARDLESS of Stage 3 results.

This is NOT a fallback for failed taint analysis.
This is an orthogonal analysis layer targeting semantic bugs that
data-flow analysis cannot find by design.

                "Stage 3 catches memory bugs.
                 Stage 4.5 catches semantic bugs.
                 They are complementary, not sequential."

Architecture:
  SEQUENTIAL CHAIN (each analysis sees previous conclusions):
    Logic Validation    — missing checks, wrong comparisons, unreachable validation
          ↓
    State Machine       — use-after-free via state, double-free, use-before-init
          ↓ (sees Logic results)
    Arithmetic Correct. — integer overflow in logic, wrong cast, signedness confusion
          ↓ (sees Logic + State Machine results)
    API Correctness     — malloc-without-null-check, unchecked read, double-free
          ↓ (sees all above)
    Protocol Compliance — format/protocol violations (PNG, XML, SQLite, …)
          ↓ (sees all above)
  SemanticAssessment per finding, with prior_analysis_reference

Design principles:
  - Runs on the TOP-N ranked functions (not just zero-candidate ones)
  - Sequential chain: each analysis builds on prior conclusions
  - Each analysis produces a SemanticAssessment (not a Finding — Stage 4.6 Judge decides)
  - No taint paths required — pure LLM reasoning over P-code + semantic context
  - chained=False falls back to parallel (legacy, for testing)

Public API:
    analyzer = OrthogonalSemanticAnalyzer(api_key=...)
    semantic_assessments = analyzer.analyze_all(funcs, budget=15)
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import enum

log = logging.getLogger(__name__)


# ─── Analysis Types ───────────────────────────────────────────────────────────

class AnalysisType(enum.Enum):
    LOGIC_VALIDATION    = "logic_validation"
    STATE_MACHINE       = "state_machine"
    ARITHMETIC          = "arithmetic_correctness"
    API_CORRECTNESS     = "api_correctness"
    PROTOCOL_COMPLIANCE = "protocol_compliance"


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SemanticAssessment:
    """
    One semantic bug finding from Stage 4.5 Orthogonal analysis.

    Not a "Finding" — Stage 4.6 (Fusion Judge) decides whether to emit a final Finding.
    Includes alternative explanations so the LLM is honest about uncertainty.

    prior_analysis_reference: how this analysis referenced prior chain conclusions.
    """
    func_name:   str
    entry_addr:  str
    analysis_type: str            # AnalysisType.value
    potential_bug: str            # one-line description of the bug
    reason:       str             # why this looks like a bug
    supporting_ops: list[str]     # relevant P-code op descriptions
    alternative_explanation: str  # why it might NOT be a bug
    confidence:   float           # [0.0, 1.0]
    vuln_type:    str             # inferred vulnerability class
    prior_analysis_reference: str = ""   # how prior chain findings were referenced

    # For Stage 5 report and Fusion Layer
    @property
    def confirmed(self) -> bool:
        """Semantic bug with confidence >= 0.60 is treated as confirmed."""
        return self.confidence >= 0.60

    def to_finding(self):
        """Convert to legacy Finding for Stage 5 report compatibility."""
        from reasoning_agent import Finding
        return Finding(
            func_name    = self.func_name,
            entry_addr   = self.entry_addr,
            vuln_type    = self.vuln_type,
            sink_fn      = f"semantic:{self.analysis_type}",
            op_seq       = -1,
            taint_source = "orthogonal_analysis",
            taint_path   = [],
            confirmed    = self.confirmed,
            severity     = "medium" if self.confidence >= 0.75 else "low",
            reasoning    = (
                f"[Orthogonal/{self.analysis_type}] {self.reason} "
                f"Alternative: {self.alternative_explanation}"
            ),
            exploit_condition     = self.potential_bug,
            false_positive_reason = self.alternative_explanation,
            confidence   = self.confidence,
            model_used   = "orthogonal_analysis",
            analysis_time_s = 0.0,
        )


# ─── Prompt Templates ─────────────────────────────────────────────────────────

_SYS = """\
You are an AI Security Analyst reviewing binary P-code for semantic vulnerabilities.
Respond with ONLY valid JSON — no markdown, no explanation outside the JSON."""

_PREAMBLE = """\
Function: {func_name}  @ {entry_addr}
Semantic role (Stage 2): {role}
Library context: {library_context}

P-code operations:
{ops_text}

"""

# Each analysis type has a focused prompt suffix

_PROMPTS = {
    AnalysisType.LOGIC_VALIDATION: """\
ANALYSIS: Logic Validation

Check for logic-level vulnerabilities that taint analysis cannot detect:
- Missing NULL check after allocation (malloc returns NULL on failure)
- Wrong comparison operator (< vs <= causing off-by-one)
- Missing error handling for failed reads (unchecked return value from recv/fread)
- Validation in wrong order (check after use, not before)
- Unreachable error paths (always-true conditions before error handler)

Respond with JSON:
{{
  "found": true|false,
  "potential_bug": "<one-line description or empty>",
  "reason": "<why this looks like a logic bug>",
  "supporting_ops": ["<seq=N op>"],
  "alternative_explanation": "<why it might be safe>",
  "confidence": <0.0-1.0>,
  "vuln_type": "<missing_null_check|off_by_one|unchecked_return|logic_error>"
}}""",

    AnalysisType.STATE_MACHINE: """\
ANALYSIS: State Machine Analysis

Check for state machine violations and lifecycle errors:
- Use-after-free: memory freed then accessed (CALL|free → LOAD of same ptr)
- Double-free: same pointer freed twice
- Use-before-initialization: variable read before any assignment
- Out-of-order API calls: open → close → read (wrong sequence)
- Missing teardown: resource allocated, exit path skips free

Respond with JSON:
{{
  "found": true|false,
  "potential_bug": "<one-line description or empty>",
  "reason": "<why this looks like a state violation>",
  "supporting_ops": ["<seq=N op>"],
  "alternative_explanation": "<why it might be safe>",
  "confidence": <0.0-1.0>,
  "vuln_type": "<use_after_free|double_free|use_before_init|api_misorder>"
}}""",

    AnalysisType.ARITHMETIC: """\
ANALYSIS: Arithmetic Correctness

Check for arithmetic-level vulnerabilities:
- Integer overflow BEFORE the result reaches a sink (no malloc needed to be dangerous)
  e.g. size = width * height * bytes_per_pixel — overflows to small value → wrong alloc
- Signedness confusion: signed value used as unsigned size argument
- Integer truncation: large value cast to smaller type (INT_ZEXT then narrow STORE)
- Wrong shift: INT_LEFT by too many bits → becomes negative
- Division by zero: divisor not checked for zero before INT_DIV

Respond with JSON:
{{
  "found": true|false,
  "potential_bug": "<one-line description or empty>",
  "reason": "<specific arithmetic ops that are dangerous>",
  "supporting_ops": ["<seq=N op>"],
  "alternative_explanation": "<why the arithmetic might be safe>",
  "confidence": <0.0-1.0>,
  "vuln_type": "<integer_overflow|integer_truncation|signedness_confusion|divide_by_zero>"
}}""",

    AnalysisType.API_CORRECTNESS: """\
ANALYSIS: API Usage Correctness

Check for incorrect use of library/system APIs:
- malloc without null-check (allocation failure not handled)
- read/recv result used directly as size without checking for -1
- Format string from external input passed to printf/sprintf family
- Buffer passed to memcpy/strcpy without size bound from the same source
- realloc result overwrites original pointer (NULL return → memory leak)
- free of stack memory or non-heap pointer

Respond with JSON:
{{
  "found": true|false,
  "potential_bug": "<one-line description or empty>",
  "reason": "<which API call is used incorrectly and why>",
  "supporting_ops": ["<seq=N op>"],
  "alternative_explanation": "<why the API usage might be safe>",
  "confidence": <0.0-1.0>,
  "vuln_type": "<api_misuse|format_string|unchecked_return|null_deref>"
}}""",

    AnalysisType.PROTOCOL_COMPLIANCE: """\
ANALYSIS: Protocol / Format Compliance

Check for protocol-level violations given the library context:
- Chunk size not validated against total stream size
- Field value accepted without range check (e.g. bit_depth must be 1/2/4/8/16)
- State flag not checked before processing (e.g. PNG IHDR must come first)
- Length field used directly without checking < remaining_bytes
- Integer fields combined with arithmetic before validation

This analysis applies best to image decoders (PNG, TIFF, JPEG), XML parsers,
database engines (SQLite), and network protocol handlers.

Respond with JSON:
{{
  "found": true|false,
  "potential_bug": "<one-line description or empty>",
  "reason": "<which protocol rule is violated>",
  "supporting_ops": ["<seq=N op>"],
  "alternative_explanation": "<why the protocol handling might be correct>",
  "confidence": <0.0-1.0>,
  "vuln_type": "<protocol_violation|missing_validation|unchecked_field>"
}}""",
}


# ─── Op Formatting ────────────────────────────────────────────────────────────

_TIER1_OPS = frozenset({
    "INT_MULT", "INT_ADD", "INT_SUB", "INT_LEFT", "INT_ZEXT", "INT_SEXT",
    "CALL", "CALLIND",
})
_TIER2_OPS = frozenset({
    "LOAD", "STORE", "INT_AND", "INT_OR",
    "INT_EQUAL", "INT_LESS", "INT_SLESS", "INT_LESSEQUAL",
    "CBRANCH", "RETURN",
})

def _format_ops(ops: list[dict], max_ops: int = 60, callee_roles: dict = None) -> str:
    """Format P-code ops for the LLM prompt, prioritizing arithmetic and calls."""
    callee_roles = callee_roles or {}

    if len(ops) <= max_ops:
        selected = ops
    else:
        keep: set[int] = set()
        # Always include first 10 ops (entry block)
        for i in range(min(10, len(ops))):
            keep.add(i)
        # Tier 1 + 2 context
        for i, op in enumerate(ops):
            if op.get("op") in _TIER1_OPS:
                for j in range(max(0, i-2), min(len(ops), i+3)):
                    keep.add(j)
        if len(keep) < max_ops:
            for i, op in enumerate(ops):
                if op.get("op") in _TIER2_OPS and i not in keep:
                    keep.add(i)
                    if len(keep) >= max_ops:
                        break
        selected = [ops[i] for i in sorted(keep)[:max_ops]]

    lines = []
    for op in selected:
        seq    = op.get("seq", "?")
        mnem   = op.get("op", "?")
        out    = op.get("output")
        inputs = op.get("inputs") or []
        out_s  = out["name"] if isinstance(out, dict) and out else "_"
        inp_parts = []
        for inp in inputs:
            if not isinstance(inp, dict):
                continue
            n = inp.get("name", "")
            if n in callee_roles:
                n += f"[{callee_roles[n]}]"
            inp_parts.append(n)
        lines.append(f"  [{seq}] {mnem:<12} {out_s}  ←  {', '.join(inp_parts)}")

    if len(ops) > max_ops:
        lines.append(f"  ... [{len(ops) - max_ops} ops omitted]")
    return "\n".join(lines)


# ─── OrthogonalSemanticAnalyzer ───────────────────────────────────────────────

class OrthogonalSemanticAnalyzer:
    """
    Stage 4.5: Five independent semantic analyses per function.

    Runs REGARDLESS of Stage 3 results.
    Its goal is semantic bugs, not memory bugs.

    Parameters
    ----------
    api_key  : OpenRouter API key (reads OPENROUTER_API_KEY from env if None)
    model    : LLM model string
    delay_s  : inter-function delay for rate limiting
    max_ops  : P-code ops to include per prompt
    """

    _DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"

    def __init__(
        self,
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
        delay_s:  float         = 2.0,
        max_ops:  int           = 60,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model   = model or self._DEFAULT_MODEL
        self.delay_s = delay_s
        self.max_ops = max_ops
        self.enabled = bool(self.api_key)

        if not self.enabled:
            log.warning("OrthogonalSemanticAnalyzer: OPENROUTER_API_KEY not set — disabled")

        self._pattern_store = None
        try:
            from pattern_store import PatternStore
            db_path = os.environ.get("PATTERN_STORE_PATH", "pattern_store.db")
            self._pattern_store = PatternStore(db_path)
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_function(
        self,
        func:         dict,
        callee_roles: dict = None,
        chained:      bool = True,
    ) -> list[SemanticAssessment]:
        """
        Run all 5 analyses on one function.

        chained=True (default):  Sequential — each analysis sees prior conclusions.
                                 Order: Logic → State Machine → Arithmetic → API → Protocol.
                                 Later analyses can build on or contradict earlier findings.
        chained=False:           Parallel (legacy mode) — 5 independent analyses.
                                 Use for testing or when latency < accuracy.
        """
        if not self.enabled:
            return []

        name  = func.get("name", "unknown")
        addr  = func.get("entry_addr", "")
        ops   = func.get("ops") or []
        role  = func.get("semantic_role", "unknown")

        if not ops:
            return []

        ops_text = _format_ops(ops, self.max_ops, callee_roles or {})
        library  = self._guess_library(func)
        preamble = _PREAMBLE.format(
            func_name        = name,
            entry_addr       = addr,
            role             = role,
            library_context  = library,
            ops_text         = ops_text,
        )

        if chained:
            results = self._analyze_chained(preamble, name, addr)
        else:
            results = self._analyze_parallel(preamble, name, addr)

        if results:
            log.info(
                "Stage 4.5 [%s]: %s — %d finding(s): %s",
                "chain" if chained else "parallel",
                name, len(results),
                ", ".join(f"{r.analysis_type}({r.confidence:.0%})" for r in results),
            )
        return results

    def _analyze_chained(
        self,
        preamble:   str,
        func_name:  str,
        entry_addr: str,
    ) -> list[SemanticAssessment]:
        """
        Sequential chain — each analysis sees prior conclusions.

        Chain order is fixed: Logic → State Machine → Arithmetic → API → Protocol.
        This ordering goes from general invariants (logic) to specific protocol rules,
        so each step can rule out or confirm what was found in earlier, broader passes.
        """
        chain_order = [
            AnalysisType.LOGIC_VALIDATION,
            AnalysisType.STATE_MACHINE,
            AnalysisType.ARITHMETIC,
            AnalysisType.API_CORRECTNESS,
            AnalysisType.PROTOCOL_COMPLIANCE,
        ]
        conclusions: list[SemanticAssessment] = []
        for at in chain_order:
            sa = self._run_one_analysis(
                analysis_type     = at,
                preamble          = preamble,
                func_name         = func_name,
                entry_addr        = entry_addr,
                prior_conclusions = conclusions,
            )
            if sa is not None:
                conclusions.append(sa)
        return conclusions

    def _analyze_parallel(
        self,
        preamble:   str,
        func_name:  str,
        entry_addr: str,
    ) -> list[SemanticAssessment]:
        """Parallel mode — 5 independent analyses (no shared context)."""
        results: list[SemanticAssessment] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(
                    self._run_one_analysis,
                    analysis_type     = at,
                    preamble          = preamble,
                    func_name         = func_name,
                    entry_addr        = entry_addr,
                    prior_conclusions = None,
                ): at
                for at in AnalysisType
            }
            for future in as_completed(futures):
                at = futures[future]
                try:
                    sa = future.result()
                    if sa is not None:
                        results.append(sa)
                except Exception as e:
                    log.debug("Orthogonal %s on %s failed: %s", at.value, func_name, e)
        return results

    def analyze_all(
        self,
        funcs:          list[dict],
        budget:         int  = 15,
        callee_roles:   dict = None,
        skip_fn_names:  set  = None,
    ) -> list[SemanticAssessment]:
        """
        Run orthogonal analysis on the top `budget` functions.

        Parameters
        ----------
        funcs          : ranked function list (highest-priority first)
        budget         : max functions to analyze (5 LLM calls each)
        callee_roles   : learned callee role map from PatternStore
        skip_fn_names  : function names already confirmed by Stage 3+4 (still analyzed —
                         orthogonal runs regardless; skip_fn_names is informational only)
        """
        if not self.enabled:
            log.info("Stage 4.5: disabled (no API key)")
            return []

        skip_fn_names = skip_fn_names or set()
        # Sort by score descending — top functions first
        sorted_funcs = sorted(funcs, key=lambda f: f.get("score", 0.0), reverse=True)
        targets      = sorted_funcs[:budget]

        log.info(
            "Stage 4.5: Orthogonal Semantic Analysis on %d functions (budget=%d) …",
            len(targets), budget,
        )

        callee_roles = callee_roles or self._load_callee_roles()
        all_assessments: list[SemanticAssessment] = []

        for i, func in enumerate(targets, 1):
            name = func.get("name", "?")
            log.info("[%d/%d] Orthogonal: %s", i, len(targets), name)

            assessments = self.analyze_function(func, callee_roles=callee_roles)
            all_assessments.extend(assessments)

            if i < len(targets):
                time.sleep(self.delay_s)

        confirmed = sum(1 for a in all_assessments if a.confirmed)
        log.info(
            "Stage 4.5 done — %d functions → %d semantic findings (%d confirmed)",
            len(targets), len(all_assessments), confirmed,
        )
        return all_assessments

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_one_analysis(
        self,
        analysis_type:     AnalysisType,
        preamble:          str,
        func_name:         str,
        entry_addr:        str,
        prior_conclusions: list[SemanticAssessment] = None,
    ) -> Optional[SemanticAssessment]:
        """
        Run one analysis type.

        prior_conclusions: list of SemanticAssessment from earlier chain steps.
        When non-empty, a "PRIOR ANALYSIS CONCLUSIONS" block is injected so the
        LLM can reference (and optionally contradict) earlier findings.
        """
        prior_block = ""
        if prior_conclusions:
            prior_block = (
                "\n\nPRIOR ANALYSIS CONCLUSIONS (earlier steps in this chain):\n"
            )
            for prior in prior_conclusions:
                prior_block += (
                    f"  [{prior.analysis_type}] {prior.potential_bug}  "
                    f"(confidence {prior.confidence:.0%})\n"
                    f"    Reason: {prior.reason}\n"
                )
            prior_block += (
                '\nReference these findings. Add "prior_analysis_reference" to your '
                "JSON to state whether they support or contradict your finding "
                '(or "none" if unrelated).\n'
            )

        prompt_suffix = _PROMPTS[analysis_type] + prior_block
        user_msg = preamble + prompt_suffix

        raw = self._call_openrouter(user_msg, func_name, analysis_type.value)
        if raw is None:
            return None

        parsed = self._parse(raw)
        if parsed is None or not parsed.get("found", False):
            return None

        confidence = float(parsed.get("confidence", 0.0))
        if confidence < 0.40:
            return None

        return SemanticAssessment(
            func_name    = func_name,
            entry_addr   = entry_addr,
            analysis_type  = analysis_type.value,
            potential_bug  = parsed.get("potential_bug", ""),
            reason         = parsed.get("reason", ""),
            supporting_ops = list(parsed.get("supporting_ops", [])),
            alternative_explanation = parsed.get("alternative_explanation", ""),
            confidence     = confidence,
            vuln_type      = parsed.get("vuln_type", analysis_type.value),
            prior_analysis_reference = parsed.get("prior_analysis_reference", ""),
        )

    def _guess_library(self, func: dict) -> str:
        """Infer the library from function name patterns."""
        name = (func.get("name") or "").lower()
        if any(x in name for x in ("png", "libpng")):
            return "libpng (PNG image decoder)"
        if any(x in name for x in ("tiff", "tif")):
            return "libtiff (TIFF image decoder)"
        if any(x in name for x in ("xml", "html")):
            return "libxml2 (XML/HTML parser)"
        if any(x in name for x in ("sqlite", "sql")):
            return "SQLite (embedded database)"
        if any(x in name for x in ("snd", "sf_", "psf_", "audio", "wav")):
            return "libsndfile (audio decoder)"
        return "unknown"

    def _load_callee_roles(self) -> dict:
        """Load learned callee roles from PatternStore."""
        if self._pattern_store is None:
            return {}
        try:
            return self._pattern_store.get_all_callee_roles()
        except Exception:
            return {}

    def _parse(self, raw: str) -> Optional[dict]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    def _call_openrouter(
        self, user_message: str, func_name: str, stage: str = "orthogonal"
    ) -> Optional[str]:
        import threading
        import urllib.request as _ur

        # Wall-clock timeout enforced via a daemon thread — the socket-level timeout
        # alone will not fire if the server stalls while streaming the response body.
        _WALL_TIMEOUT = 90   # seconds per attempt
        _RETRY_WAIT   = 15   # seconds before the one retry

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/binary-vuln-pipeline",
            "X-Title":       "Binary Vulnerability Analysis Pipeline",
        }
        payload = json.dumps({
            "model": self.model, "max_tokens": 600, "temperature": 0.0,
            "messages": [
                {"role": "system", "content": _SYS},
                {"role": "user",   "content": user_message},
            ],
        }).encode()

        def _attempt():
            slot_r: list = [None]
            slot_e: list = [None]
            def _inner():
                try:
                    req = _ur.Request(
                        "https://openrouter.ai/api/v1/chat/completions",
                        data=payload, headers=headers, method="POST",
                    )
                    with _ur.urlopen(req, timeout=_WALL_TIMEOUT) as resp:
                        slot_r[0] = json.loads(resp.read().decode())
                except Exception as exc:
                    slot_e[0] = exc
            t = threading.Thread(target=_inner, daemon=True)
            t.start()
            t.join(_WALL_TIMEOUT + 5)
            if t.is_alive():
                return None, TimeoutError(f"wall-clock timeout after {_WALL_TIMEOUT}s")
            return slot_r[0], slot_e[0]

        for attempt_num in range(2):
            body, err = _attempt()
            if err is not None:
                if attempt_num == 0:
                    log.warning("Stage4.5 [%s] %s error -- retrying in %ds: %s",
                                stage, func_name, _RETRY_WAIT, err)
                    time.sleep(_RETRY_WAIT)
                    continue
                log.error("Stage4.5 [%s] %s failed after retry: %s", stage, func_name, err)
                return None
            try:
                text = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                log.error("Stage4.5 bad response body [%s] %s: %s", stage, func_name, exc)
                return None
            try:
                from llm_cost_tracker import GLOBAL_TRACKER
                usage = body.get("usage", {})
                GLOBAL_TRACKER.record(
                    stage=f"orthogonal_{stage}", model=self.model,
                    input_tokens  = usage.get("prompt_tokens") or max(1, len(user_message)//4),
                    output_tokens = usage.get("completion_tokens") or max(1, len(text)//4),
                    latency_s=0.0, fn_name=func_name,
                )
            except Exception:
                pass
            return text
        return None
