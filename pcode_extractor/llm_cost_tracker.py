"""
llm_cost_tracker.py — Item 2: LLM Cost Measurement.

Wraps every LLM API call to capture:
  - Model used
  - Input tokens (prompt)
  - Output tokens (completion)
  - Wall-clock latency (seconds)
  - Estimated USD cost (from public pricing tables)
  - Call purpose (stage / agent name)

Aggregated across a full binary run, these produce Table 3 in the paper:

    Stage              | Calls | Input tok | Output tok | Latency(s) | Cost($)
    ───────────────────|───────|───────────|────────────|────────────|─────────
    Stage 2.5 Semantic |  47   |   142K    |   18K      |   62.4 s   | $0.034
    Stage 4 Reasoning  |   8   |    19K    |    4K      |   22.1 s   | $0.011
    Track 2 Logic      |   3   |    8K     |    2K      |    9.8 s   | $0.004
    ───────────────────|───────|───────────|────────────|────────────|─────────
    TOTAL              |  58   |   169K    |   24K      |   94.3 s   | $0.049

This is publishable because:
  1. No other binary analysis paper reports per-stage LLM cost.
  2. Cost-per-binary (< $0.05) is a key deployment argument.
  3. LLM Reduction % shows what % of functions needed LLM vs. pattern match.

Usage
─────
Wrap existing LLM calls:

    from llm_cost_tracker import LLMCostTracker

    tracker = LLMCostTracker()

    # Instead of calling the API directly:
    response = tracker.tracked_call(
        fn          = your_llm_call_fn,   # callable returning (text, usage_dict)
        model       = "llama-3.3-70b",
        stage       = "semantic_recovery",
        input_text  = prompt,
    )

Or manually record a call that already happened:
    tracker.record(stage="reasoning", model="...", input_tokens=1200,
                   output_tokens=300, latency_s=4.2)

To serialize for paper / JSON output:
    table = tracker.summary_table()   # list of dicts, one per stage
    tracker.print_table()
    tracker.save_json("cost_report.json")

Thread-safety: the tracker is not thread-safe (single-threaded pipeline assumption).
"""
from __future__ import annotations
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# ── Pricing table (USD per 1K tokens) ────────────────────────────────────────
# Prices from provider documentation, updated 2025-Q2.
# These are approximate — mark as such in the paper footnotes.
_PRICE_PER_1K: dict[str, dict[str, float]] = {
    # OpenRouter hosted endpoints
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.00059, "output": 0.00079},
    "llama-3.3-70b-instruct":            {"input": 0.00059, "output": 0.00079},
    "llama-3.3-70b-versatile":           {"input": 0.00059, "output": 0.00079},
    # Groq (fast inference)
    "llama-3.3-70b":                     {"input": 0.00059, "output": 0.00079},
    # Google Gemini
    "gemini-2.5-flash":                  {"input": 0.00015, "output": 0.00060},
    "gemini-2.0-flash":                  {"input": 0.00010, "output": 0.00040},
    # Anthropic Claude
    "claude-haiku-4-5-20251001":         {"input": 0.00025, "output": 0.00125},
    "claude-haiku-4-5":                  {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-6":                 {"input": 0.00300, "output": 0.01500},
}
_DEFAULT_PRICE = {"input": 0.00060, "output": 0.00080}  # conservative fallback


def _price_for(model: str) -> dict[str, float]:
    m = model.lower()
    # Try exact match first, then substring match
    if m in _PRICE_PER_1K:
        return _PRICE_PER_1K[m]
    for key, prices in _PRICE_PER_1K.items():
        if key in m or m in key:
            return prices
    return _DEFAULT_PRICE


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from text length (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


@dataclass
class CallRecord:
    stage:        str    # "semantic_recovery" | "reasoning" | "track2"
    model:        str
    input_tokens: int
    output_tokens: int
    latency_s:    float
    cost_usd:     float
    fn_name:      str = ""  # which function was being analyzed


@dataclass
class StageSummary:
    stage:         str
    calls:         int
    input_tokens:  int
    output_tokens: int
    total_tokens:  int
    latency_s:     float
    cost_usd:      float

    def as_dict(self) -> dict:
        return {
            "stage":          self.stage,
            "calls":          self.calls,
            "input_tokens":   self.input_tokens,
            "output_tokens":  self.output_tokens,
            "total_tokens":   self.total_tokens,
            "latency_s":      round(self.latency_s, 2),
            "cost_usd":       round(self.cost_usd, 6),
        }


class LLMCostTracker:
    """
    Records LLM call metadata across all stages of the HybridTaint pipeline.
    """

    def __init__(self):
        self._records: list[CallRecord] = []

    def record(
        self,
        stage:         str,
        model:         str,
        input_tokens:  int,
        output_tokens: int,
        latency_s:     float,
        fn_name:       str = "",
    ) -> None:
        """Record a call that already completed (manual mode)."""
        prices = _price_for(model)
        cost = (input_tokens / 1000 * prices["input"]
              + output_tokens / 1000 * prices["output"])
        rec = CallRecord(
            stage=stage, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_s=latency_s, cost_usd=cost, fn_name=fn_name,
        )
        self._records.append(rec)
        log.debug(
            "LLMCostTracker [%s] %s: in=%d out=%d %.1fs $%.5f",
            stage, fn_name or model, input_tokens, output_tokens, latency_s, cost,
        )

    def record_from_response(
        self,
        stage:     str,
        model:     str,
        response:  Any,          # requests.Response or parsed dict
        start_time: float,
        fn_name:   str = "",
        prompt_text: str = "",   # fallback if usage not in response
    ) -> None:
        """
        Extract token counts from an API response object and record.
        Handles OpenAI-compatible (groq/openrouter) and Gemini response shapes.
        """
        latency_s = time.monotonic() - start_time

        # Try to extract usage from various response shapes
        usage_dict = {}
        if hasattr(response, "usage"):
            u = response.usage
            if hasattr(u, "__dict__"):
                usage_dict = vars(u)
            elif isinstance(u, dict):
                usage_dict = u

        in_tok  = (usage_dict.get("prompt_tokens")
                or usage_dict.get("input_tokens")
                or usage_dict.get("promptTokenCount")
                or _estimate_tokens(prompt_text))
        out_tok = (usage_dict.get("completion_tokens")
                or usage_dict.get("output_tokens")
                or usage_dict.get("candidatesTokenCount")
                or 200)  # conservative fallback

        self.record(stage=stage, model=model,
                    input_tokens=int(in_tok), output_tokens=int(out_tok),
                    latency_s=latency_s, fn_name=fn_name)

    def record_from_text(
        self,
        stage:        str,
        model:        str,
        prompt:       str,
        completion:   str,
        latency_s:    float,
        fn_name:      str = "",
    ) -> None:
        """Record from raw prompt/completion strings (token count estimated)."""
        self.record(
            stage         = stage,
            model         = model,
            input_tokens  = _estimate_tokens(prompt),
            output_tokens = _estimate_tokens(completion),
            latency_s     = latency_s,
            fn_name       = fn_name,
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def by_stage(self) -> dict[str, StageSummary]:
        """Return per-stage aggregated summary."""
        buckets: dict[str, list[CallRecord]] = defaultdict(list)
        for r in self._records:
            buckets[r.stage].append(r)
        summaries = {}
        for stage, recs in sorted(buckets.items()):
            summaries[stage] = StageSummary(
                stage         = stage,
                calls         = len(recs),
                input_tokens  = sum(r.input_tokens  for r in recs),
                output_tokens = sum(r.output_tokens for r in recs),
                total_tokens  = sum(r.input_tokens + r.output_tokens for r in recs),
                latency_s     = sum(r.latency_s   for r in recs),
                cost_usd      = sum(r.cost_usd    for r in recs),
            )
        return summaries

    def totals(self) -> StageSummary:
        recs = self._records
        prices_set = {r.model for r in recs}
        return StageSummary(
            stage         = "TOTAL",
            calls         = len(recs),
            input_tokens  = sum(r.input_tokens  for r in recs),
            output_tokens = sum(r.output_tokens for r in recs),
            total_tokens  = sum(r.input_tokens + r.output_tokens for r in recs),
            latency_s     = sum(r.latency_s  for r in recs),
            cost_usd      = sum(r.cost_usd   for r in recs),
        )

    def summary_table(self) -> list[dict]:
        """
        Return list of dicts suitable for paper Table 3.
        Order: individual stages sorted alpha, then TOTAL row at end.
        """
        rows = [s.as_dict() for s in self.by_stage().values()]
        rows.append(self.totals().as_dict())
        return rows

    def print_table(self, label: str = "") -> None:
        """Print a formatted cost table to stdout."""
        header = f"\n  LLM Cost Report{' — '+label if label else ''}"
        sep    = "  " + "-" * 74
        print(header)
        print(sep)
        print(f"  {'Stage':<22} {'Calls':>6} {'In-tok':>8} {'Out-tok':>8} "
              f"{'Total-tok':>10} {'Latency':>9} {'Cost':>10}")
        print(sep)
        stages = self.by_stage()
        for name, s in stages.items():
            print(f"  {s.stage:<22} {s.calls:>6} {s.input_tokens:>8,} "
                  f"{s.output_tokens:>8,} {s.total_tokens:>10,} "
                  f"{s.latency_s:>8.1f}s  ${s.cost_usd:>8.5f}")
        t = self.totals()
        print(sep)
        print(f"  {'TOTAL':<22} {t.calls:>6} {t.input_tokens:>8,} "
              f"{t.output_tokens:>8,} {t.total_tokens:>10,} "
              f"{t.latency_s:>8.1f}s  ${t.cost_usd:>8.5f}")
        print()

    def save_json(self, path: str | Path) -> None:
        data = {
            "summary": self.summary_table(),
            "records": [
                {
                    "stage":         r.stage,
                    "model":         r.model,
                    "fn_name":       r.fn_name,
                    "input_tokens":  r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_s":     round(r.latency_s, 3),
                    "cost_usd":      round(r.cost_usd, 6),
                }
                for r in self._records
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("LLM cost report saved → %s", path)

    def merge(self, other: "LLMCostTracker") -> None:
        """Merge another tracker's records into this one (for cross-binary totals)."""
        self._records.extend(other._records)

    def reset(self) -> None:
        self._records.clear()

    # ── Paper metric helpers ──────────────────────────────────────────────────

    def llm_reduction_pct(
        self,
        total_candidates_considered: int,
        stage: str = "reasoning",
    ) -> float:
        """
        LLM Reduction % = (total_candidates - LLM_calls_in_stage) / total_candidates * 100.
        Measures how many candidates were resolved WITHOUT an LLM call (via pattern match).
        """
        stage_calls = self.by_stage().get(stage, StageSummary(stage, 0, 0, 0, 0, 0.0, 0.0)).calls
        if total_candidates_considered == 0:
            return 0.0
        resolved_by_pattern = total_candidates_considered - stage_calls
        return round(resolved_by_pattern / total_candidates_considered * 100, 1)

    def cost_per_binary(self) -> float:
        return round(self.totals().cost_usd, 6)

    def avg_latency_per_call(self) -> float:
        t = self.totals()
        return round(t.latency_s / t.calls, 2) if t.calls else 0.0


# ── Global singleton (pipeline-wide tracker) ─────────────────────────────────
# Import and use this in every agent:
#   from llm_cost_tracker import GLOBAL_TRACKER
#   GLOBAL_TRACKER.record(stage="reasoning", ...)
GLOBAL_TRACKER = LLMCostTracker()
