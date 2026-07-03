"""
track2_agent.py — Track 2: Direct P-code → LLM analysis for logic bugs.

Track 1 (taint analysis) detects vulnerabilities by tracing data from external
inputs to dangerous sinks (malloc, memcpy). It misses bugs where:
  - The overflow happens before any allocator call
  - The bug is a pure arithmetic error (no memory sink)
  - The issue is use-after-free via function pointer
  - Race conditions / double-free patterns

Track 2 bypasses taint analysis and asks the LLM to reason directly over P-code,
finding bugs that are visible in the logic but not in the data-flow graph.

Typical FNs fixed by Track 2:
  png_check_IHDR       — integer overflow in dimension bounds check
  png_read_transform_info — overflow in bit_depth * width computation
  png_safe_execute     — use-after-free via longjmp/function pointer
  png_image_free       — double-free / use-after-free
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ── Smart op selection ────────────────────────────────────────────────────────

# Two-tier priority for op selection in long functions.
# Tier 1 (always include): arithmetic + calls — these harbour most bugs
# Tier 2 (fill up to budget): memory ops + comparisons for context
_TIER1_OPS = frozenset({
    "INT_MULT", "INT_ADD", "INT_SUB", "INT_LEFT",
    "INT_ZEXT", "INT_SEXT",
    "CALL", "CALLIND",
})
_TIER2_OPS = frozenset({
    "LOAD", "STORE",
    "INT_AND", "INT_OR", "INT_XOR",
    "INT_EQUAL", "INT_LESS", "INT_SLESS", "INT_LESSEQUAL", "INT_SLESSEQUAL",
})

_ENTRY_OPS = 15   # always show the first N ops (entry context)
_CONTEXT   = 2    # ops of context around each selected op


def _select_ops(ops: list[dict], max_ops: int = 80) -> list[dict]:
    """
    Return at most max_ops ops, prioritising arithmetic/call ops over memory ops.

    Phase 1: entry block (first _ENTRY_OPS)
    Phase 2: all Tier-1 ops + context (arithmetic, calls)
    Phase 3: fill remaining budget with Tier-2 ops + context
    """
    if len(ops) <= max_ops:
        return ops

    def add_with_context(indices: set[int], center: int) -> None:
        for j in range(max(0, center - _CONTEXT), min(len(ops), center + _CONTEXT + 1)):
            indices.add(j)

    selected: set[int] = set()

    # Phase 1: entry block
    for i in range(min(_ENTRY_OPS, len(ops))):
        selected.add(i)

    # Phase 2: Tier-1 ops (all of them, ignoring budget)
    for i, op in enumerate(ops):
        if op.get("op") in _TIER1_OPS:
            add_with_context(selected, i)

    # Phase 3: Tier-2 ops to fill remaining budget
    if len(selected) < max_ops:
        for i, op in enumerate(ops):
            if op.get("op") in _TIER2_OPS and i not in selected:
                add_with_context(selected, i)
                if len(selected) >= max_ops:
                    break

    sorted_idx = sorted(selected)[:max_ops]
    result = [ops[i] for i in sorted_idx]

    # Append summary if truncated
    if len(sorted_idx) < len(ops):
        result.append({
            "seq": "...",
            "op":  f"[{len(ops) - len(sorted_idx)} ops omitted — "
                   f"total {len(ops)} ops, showing {len(sorted_idx)} selected]",
            "output": None,
            "inputs": [],
        })

    return result

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior binary security researcher.
You analyze Ghidra P-code IR for vulnerabilities that static taint analysis misses.
Respond with ONLY valid JSON — no markdown, no explanation outside the JSON."""

_USER_TEMPLATE = """TRACK 2 VULNERABILITY ANALYSIS

Function: {fn_name}  (entry={entry_addr})
Note: Standard taint analysis found no dangerous sink. Look for logic-level bugs.

P-code operations (callee roles annotated in brackets):
{ops_text}

Look specifically for ALL of the following bug classes:

MEMORY CORRUPTION (classic):
1. INTEGER OVERFLOW — INT_MULT / INT_ADD / INT_LEFT on external-sized args,
   result used without overflow check (even without reaching malloc/memcpy)
2. INTEGER TRUNCATION — value widened then narrowed (INT_ZEXT → narrow COPY/STORE)
3. USE-AFTER-FREE — CALL to free/png_free, then LOAD/STORE via same pointer variable
4. MISSING NULL CHECK — pointer from CALL used immediately without INT_EQUAL null test
5. DOUBLE-FREE — same pointer freed twice via different code paths

LOGIC BUGS (parser/format libraries — libpng, libtiff, libxml2, etc.):
6. CHECK-BYPASS — a bounds check (INT_LESS/INT_LESSEQUAL) exists but uses a wrong
   limit (e.g., comparing against a size field from the SAME untrusted input), or the
   comparison direction is flipped (> instead of >=), allowing out-of-bounds access
7. OFF-BY-ONE — length or offset computed as N when N-1 is needed (or vice versa),
   typically seen as INT_ADD const(1) used as a limit that then reaches a LOAD/STORE
8. PARSER-STATE — state machine advances without validating current state, allowing
   a chunk/record handler to be invoked in an unexpected sequence (no predecessor check)
9. INCORRECT-BOUNDS — a size limit derived from user-controlled data (e.g., chunk length
   field) used directly as a loop bound or allocation size without cross-checking against
   a separate authoritative size constraint

Return JSON:
{{
  "found":       <true | false>,
  "vuln_type":   "<integer_overflow | integer_truncation | use_after_free | null_deref | double_free | check_bypass | off_by_one | parser_state | incorrect_bounds | logic_bug | none>",
  "confidence":  <0.0 – 1.0>,
  "trigger_seq": <seq number of the problematic op, or -1>,
  "affected_vars": ["<var1>", "<var2>"],
  "trigger_condition": "<what attacker-controlled value or state causes the bug>",
  "reasoning":   "<2-3 sentences explaining exactly where and why>"
}}

If no bug found: {{"found": false, "vuln_type": "none", "confidence": 0.9, "trigger_seq": -1,
  "affected_vars": [], "trigger_condition": "", "reasoning": "<why clean>"}}"""

_FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class Track2Result:
    func_name:       str
    entry_addr:      str
    found:           bool
    vuln_type:       str
    confidence:      float
    trigger_seq:     int
    affected_vars:   list[str]
    trigger_condition: str
    reasoning:       str
    model_used:      str
    analysis_time_s: float


# ── Agent class ───────────────────────────────────────────────────────────────

class Track2Agent:
    """
    Direct P-code → LLM analysis for logic bugs that taint analysis misses.

    Usage:
        agent = Track2Agent(api_key=os.environ["OPENROUTER_API_KEY"])
        results = agent.process_binary(functions, confirmed_fn_names, callee_roles)
    """

    def __init__(self, api_key: str = "", delay_s: float = 5.0, max_ops: int = 60):
        self.api_key = api_key
        self.delay_s = delay_s
        self.max_ops = max_ops
        self.enabled = bool(api_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def screen_candidates(
        self,
        functions: list[dict],
        confirmed_fn_names: set[str],
        top_n: int = 30,
        priority_fn_names: set[str] | None = None,
    ) -> list[dict]:
        """
        Return top-N functions that Track 1 didn't confirm.

        Priority order:
          1a. Zero Stage-3-candidate fns WITH INT_MULT (arithmetic-heavy, sink-free)
          1b. Zero Stage-3-candidate fns WITHOUT INT_MULT (orchestrators, lower value)
          2.  Remaining high-score fns sorted by score descending
        """
        priority = set(priority_fn_names or [])
        eligible = [
            f for f in functions
            if f.get("name") and
               f.get("ops") and
               f.get("name") not in confirmed_fn_names and
               f.get("score", 0.0) >= 0.3
        ]

        # Priority pool: all zero-candidate functions, sorted DESCENDING by score.
        # High-scoring zero-candidate fns are the most likely GT misses after Track 1
        # precision fixes (write sinks with param-only taint are now suppressed, so
        # high-score write functions like wav_write_header end up with 0 candidates).
        tier_priority = [f for f in eligible if f.get("name") in priority]
        tier_priority.sort(key=lambda f: f.get("score", 0.0), reverse=True)
        # Tier 2: remaining unconfirmed fns to fill remaining budget
        tier2 = [f for f in eligible if f.get("name") not in priority]
        tier2.sort(key=lambda f: f.get("score", 0.0), reverse=True)

        combined = tier_priority + tier2
        return combined[:top_n]

    def analyze_function(
        self,
        fn_name: str,
        ops: list[dict],
        entry_addr: str = "?",
        callee_roles: dict | None = None,
    ) -> Optional[Track2Result]:
        """Ask LLM to find logic bugs directly in the P-code."""
        if not self.enabled:
            return None

        from semantic_recovery_agent import _render_ops
        selected = _select_ops(ops, self.max_ops)
        ops_text = _render_ops(selected, len(selected), callee_roles=callee_roles)

        user_msg = _USER_TEMPLATE.format(
            fn_name    = fn_name,
            entry_addr = entry_addr,
            ops_text   = ops_text,
        )

        t0  = time.perf_counter()
        raw = self._call(user_msg, fn_name)
        elapsed = time.perf_counter() - t0

        if not raw:
            return None

        parsed = self._parse(raw, fn_name)
        if parsed is None:
            return None

        return Track2Result(
            func_name         = fn_name,
            entry_addr        = entry_addr,
            found             = bool(parsed.get("found", False)),
            vuln_type         = parsed.get("vuln_type", "logic_bug"),
            confidence        = float(parsed.get("confidence", 0.5)),
            trigger_seq       = int(parsed.get("trigger_seq", -1)),
            affected_vars     = list(parsed.get("affected_vars", [])),
            trigger_condition = str(parsed.get("trigger_condition", "")),
            reasoning         = str(parsed.get("reasoning", "")),
            model_used        = self._last_model or "track2/unknown",
            analysis_time_s   = round(elapsed, 3),
        )

    def process_binary(
        self,
        functions: list[dict],
        confirmed_fn_names: set[str],
        callee_roles: dict | None = None,
        budget: int = 15,
        priority_fn_names: set[str] | None = None,
    ) -> list[Track2Result]:
        """
        Run Track 2 on top-N high-priority functions missed by Track 1.
        priority_fn_names: functions with zero Stage 3 candidates — analyzed first.
        Returns only findings where found=True.
        """
        candidates = self.screen_candidates(
            functions, confirmed_fn_names,
            top_n=budget, priority_fn_names=priority_fn_names,
        )
        log.info("Track 2: screening %d candidates (budget=%d)", len(candidates), budget)

        results = []
        for i, func in enumerate(candidates, 1):
            fn_name    = func.get("name", "")
            ops        = func.get("ops", [])
            entry_addr = func.get("entry", "?")
            score      = func.get("score", 0.0)

            if not ops:
                continue

            log.info(
                "Track 2 [%d/%d] analyzing %s (score=%.2f, ops=%d)",
                i, len(candidates), fn_name, score, len(ops),
            )

            result = self.analyze_function(fn_name, ops, entry_addr, callee_roles)
            if result:
                status = "FOUND" if result.found else "clean"
                log.info(
                    "  => %s  %s  conf=%.0f%%  (%s)",
                    status,
                    result.vuln_type if result.found else "",
                    result.confidence * 100,
                    result.reasoning[:60].replace("\n", " "),
                )
                if result.found and result.confidence >= 0.5:
                    results.append(result)
            else:
                log.warning("  => no response for %s", fn_name)

            time.sleep(self.delay_s)

        log.info("Track 2 complete: %d findings from %d analyzed", len(results), len(candidates))
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    _last_model: str = ""

    def _call(self, user_message: str, fn_name: str) -> Optional[str]:
        import os
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return None

        for model in _FREE_MODELS:
            payload = json.dumps({
                "model": model,
                "max_tokens": 512,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            }).encode()
            try:
                from llm_cost_tracker import GLOBAL_TRACKER
                t_start = time.perf_counter()
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data    = payload,
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "https://github.com/binary-vuln-pipeline",
                        "X-Title":       "BinaryVulnPipeline-Track2",
                    },
                    method = "POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read())
                    if "error" in body:
                        log.warning("Track2 %s error: %s", model, body["error"])
                        continue
                    text  = body["choices"][0]["message"]["content"]
                    usage = body.get("usage", {})
                    GLOBAL_TRACKER.record(
                        stage         = "track2",
                        model         = model,
                        input_tokens  = int(usage.get("prompt_tokens", len(user_message)//4)),
                        output_tokens = int(usage.get("completion_tokens", len(text)//4)),
                        latency_s     = time.perf_counter() - t_start,
                        fn_name       = fn_name,
                    )
                    self._last_model = f"track2/openrouter/{model.split('/')[-1]}"
                    return text
            except urllib.error.HTTPError as e:
                e.read()
                if e.code in (429, 402):
                    log.warning("Track2 %d on %s — trying next model", e.code, model)
                    time.sleep(2)
                    continue
                elif e.code == 401:
                    log.error("Track2 401 — invalid OpenRouter key")
                    return None
                else:
                    log.warning("Track2 HTTP %d on %s", e.code, model)
                    continue
            except Exception as exc:
                log.warning("Track2 error for %s: %s", fn_name, exc)
                continue
        return None

    def _parse(self, raw: str, fn_name: str) -> Optional[dict]:
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.split("\n") if not l.strip().startswith("```")
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting JSON from response
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                except json.JSONDecodeError:
                    log.warning("Track2 JSON parse failed for %s", fn_name)
                    return None
            else:
                log.warning("Track2 no JSON in response for %s", fn_name)
                return None

        if not isinstance(parsed, dict):
            return None

        # Validate required fields
        if "found" not in parsed:
            log.warning("Track2 response missing 'found' field for %s", fn_name)
            return None

        # Normalize vuln_type — extended for Task 5 (check-bypass, off-by-one,
        # parser-state, incorrect-bounds for libpng/libtiff/libxml2)
        vt = str(parsed.get("vuln_type", "logic_bug")).lower()
        valid = {
            "integer_overflow", "integer_truncation",
            "use_after_free", "null_deref", "double_free",
            "check_bypass", "off_by_one", "parser_state", "incorrect_bounds",
            "logic_bug", "none",
        }
        if vt not in valid:
            parsed["vuln_type"] = "logic_bug"

        return parsed


# ── Integration helper: Track2Result → Finding ────────────────────────────────

def track2_result_to_finding(result: Track2Result):
    """
    Convert a Track2Result to a reasoning_agent.Finding for unified report output.
    Import Finding lazily to avoid circular imports.
    """
    from reasoning_agent import Finding
    severity_map = {
        "integer_overflow":    "high",
        "integer_truncation":  "medium",
        "use_after_free":      "high",
        "null_deref":          "medium",
        "double_free":         "high",
        # Task 5: parser/format library logic bugs
        "check_bypass":        "high",
        "off_by_one":          "high",
        "parser_state":        "medium",
        "incorrect_bounds":    "high",
        "logic_bug":           "medium",
    }
    return Finding(
        func_name             = result.func_name,
        entry_addr            = result.entry_addr,
        vuln_type             = result.vuln_type,
        sink_fn               = "LOGIC",   # no memory sink — logic-level bug
        op_seq                = result.trigger_seq,
        taint_source          = "function_args",
        taint_path            = result.affected_vars,
        confirmed             = True,
        severity              = severity_map.get(result.vuln_type, "medium"),
        reasoning             = f"[Track 2] {result.reasoning}",
        exploit_condition     = result.trigger_condition,
        false_positive_reason = "",
        confidence            = result.confidence,
        model_used            = result.model_used,
        analysis_time_s       = result.analysis_time_s,
    )
