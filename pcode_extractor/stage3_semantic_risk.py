"""
stage3_semantic_risk.py  —  Stage 3C: Semantic Risk Analysis
=============================================================
Produces evidence WITHOUT requiring a taint-to-sink path.

Reviewer-defensible design:
  - No weights in evidence items — weights become paper hyperparameters.
    Each observable feature is recorded as a named observation with provenance.
  - Emit threshold is item COUNT (>= 2 independent observations), not a score.
  - Stage 4 (LLM) receives individual evidence items with full provenance.
  - Stage 3 does NOT decide "this is a vulnerability". It says:
    "here is what I observed in the binary, and which analysis observed it".

Evidence items produced (no weights — presence/absence only):

  parser_role              Function classified PARSER/DECODER by Stage 2
  unknown_with_allocator   Unknown role but allocator present
  multiplication_present   INT_MULT found — necessary for integer overflow
  large_shift              INT_LEFT >= 2 bits — equivalent to multiplication
  arithmetic_present       2+ arithmetic operations found
  allocator_call           Heap allocator called in the same function
  no_bounds_check          No CBRANCH with constant comparison before allocator
  unchecked_arith_store    Arithmetic result stored without intervening CBRANCH
  decoder_role             Function classified DECODER/PARSER (for buffer analysis)
  high_store_count         3+ STORE ops — substantial write activity
  validator_role           Function classified VALIDATOR by Stage 2
  few_branches             VALIDATOR with fewer than 2 CBRANCHes
  memory_writes            VALIDATOR performing memory writes

Emit threshold: >= 2 items (two independent observations required).
The LLM decides whether the observations warrant confirmation.

Usage
-----
    from stage3_semantic_risk import SemanticRiskAnalyzer

    analyzer   = SemanticRiskAnalyzer()
    candidates = analyzer.analyze(func_dict, ops, semantic_role="parser")
    # → list[SemRiskCandidate]

    for c in candidates:
        print(c.collector.items)   # individual evidence items with provenance
        print(c.collector.count)   # number of observations
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stage3_evidence import EvidenceCollector, EvidenceItem, SourceRole


@dataclass
class SemRiskCandidate:
    """
    A semantic risk finding — no taint-to-sink path required.

    The collector carries the individual evidence items with provenance.
    The orchestrator converts this to VulnCandidate + EvidenceVector.
    """
    func_name:    str
    entry_addr:   str
    pattern_name: str             # primary pattern that triggered
    collector:    EvidenceCollector  # all evidence observations
    vuln_type:    str             # probable vulnerability class
    description:  str

    @property
    def semantic_score(self) -> float:
        """Item count as float (for compatibility with EvidenceVector.semantic_score)."""
        return float(self.collector.count)

    @property
    def evidence_items(self) -> list[EvidenceItem]:
        return self.collector.items

    # Back-compat alias so code referencing .accumulator still works
    @property
    def accumulator(self) -> EvidenceCollector:
        return self.collector


# ─── Allocator fragment matching ──────────────────────────────────────────────
_ALLOC_FRAGS: tuple[str, ...] = (
    "malloc", "calloc", "realloc", "xmalloc", "zmalloc", "emalloc",
    "png_malloc", "png_zalloc", "_TIFFmalloc", "_TIFFrealloc",
    "xmlMalloc", "xmlRealloc", "g_malloc", "g_realloc",
    "HeapAlloc", "VirtualAlloc", "LocalAlloc",
)


class SemanticRiskAnalyzer:
    """
    Stage 3C — generate semantic evidence without requiring a taint path.

    Three function-level analyses run:
      A. Integer overflow risk (parser + arithmetic + allocator)
      B. Buffer overflow risk (decoder + high store count + unchecked offset)
      C. Validation bypass (validator role + insufficient branching)

    Each analysis uses EvidenceCollector to record named observations.
    Candidates are emitted when at least _EMIT_MIN_ITEMS independent
    observations fire — a single observation is too weak to report.
    """

    _EMIT_MIN_ITEMS = 2  # two independent observations required to emit

    def analyze(
        self,
        func:          dict,
        ops:           list[dict],
        semantic_role: str = "unknown",
    ) -> list[SemRiskCandidate]:
        """
        Evaluate `func` for semantic risk patterns.
        Returns list of SemRiskCandidates (may be empty).
        """
        name       = func.get("name", "unknown")
        entry_addr = func.get("entry_addr", "")
        role       = (semantic_role or "unknown").lower()

        if not ops:
            return []

        stats      = self._compute_stats(ops)
        candidates: list[SemRiskCandidate] = []

        # ── Analysis A: Integer Overflow / Heap Size ──────────────────────────
        coll_a = EvidenceCollector()

        if role in ("parser", "decoder"):
            coll_a.add(
                "parser_role", "SemanticRisk",
                f"Function role={role} (Stage 2 classifier) processes external data",
            )
        elif role == "unknown" and stats["has_allocator"]:
            coll_a.add(
                "unknown_with_allocator", "SemanticRisk",
                "Unknown role but allocator present — possible parser/decoder",
            )

        if stats["mult_ops"] >= 1:
            coll_a.add(
                "multiplication_present", "SemanticRisk",
                f"{stats['mult_ops']} INT_MULT operations found — necessary for integer overflow",
            )
        elif stats["large_shift_ops"] >= 1:
            coll_a.add(
                "large_shift", "SemanticRisk",
                f"{stats['large_shift_ops']} INT_LEFT >= 2 bits — equivalent to multiplication",
            )

        if stats["arithmetic_ops"] >= 2:
            coll_a.add(
                "arithmetic_present", "SemanticRisk",
                f"{stats['arithmetic_ops']} total arithmetic ops — size/index manipulation",
            )

        if stats["has_allocator"]:
            coll_a.add(
                "allocator_call", "SemanticRisk",
                f"Calls heap allocator ({stats['allocator_fn']}) — size may be attacker-controlled",
            )

        if not stats["has_bounds_check"] and coll_a.has("allocator_call"):
            coll_a.add(
                "no_bounds_check", "SemanticRisk",
                "No CBRANCH comparing against a constant — no visible size guard before allocator",
            )

        if stats["unchecked_arith_store"]:
            coll_a.add(
                "unchecked_arith_store", "SemanticRisk",
                "Arithmetic result (including INT_MULT) flows to STORE without CBRANCH guard",
            )

        if coll_a.count >= self._EMIT_MIN_ITEMS:
            candidates.append(SemRiskCandidate(
                func_name    = name,
                entry_addr   = entry_addr,
                pattern_name = coll_a.items[0].name,
                collector    = coll_a,
                vuln_type    = "integer_overflow",
                description  = (
                    f"Semantic evidence for integer overflow in {role.upper()} function: "
                    + ", ".join(i.name for i in coll_a.items)
                ),
            ))

        # ── Analysis B: Buffer Overflow (unchecked decode writes) ─────────────
        coll_b = EvidenceCollector()

        if role in ("decoder", "parser"):
            coll_b.add(
                "decoder_role", "SemanticRisk",
                f"Function role={role} — writes data to buffer during decode",
            )

        if stats["store_ops"] >= 3:
            coll_b.add(
                "high_store_count", "SemanticRisk",
                f"{stats['store_ops']} STORE ops — substantial buffer construction activity",
            )

        if stats["arithmetic_ops"] >= 2:
            coll_b.add(
                "arithmetic_present", "SemanticRisk",
                f"{stats['arithmetic_ops']} arithmetic ops — compute write offset or length",
            )

        if not stats["has_bounds_check"] and coll_b.has("high_store_count"):
            coll_b.add(
                "no_bounds_check", "SemanticRisk",
                "No constant-comparing CBRANCH — write offset or length appears unchecked",
            )

        if coll_b.count >= self._EMIT_MIN_ITEMS and "integer_overflow" not in {
            c.vuln_type for c in candidates
        }:
            candidates.append(SemRiskCandidate(
                func_name    = name,
                entry_addr   = entry_addr,
                pattern_name = coll_b.items[0].name,
                collector    = coll_b,
                vuln_type    = "buffer_overflow",
                description  = (
                    f"Semantic evidence for buffer overflow in {role.upper()} function: "
                    + ", ".join(i.name for i in coll_b.items)
                ),
            ))

        # ── Analysis C: Validation Bypass ─────────────────────────────────────
        if role == "validator":
            coll_c = EvidenceCollector()
            coll_c.add(
                "validator_role", "SemanticRisk",
                "Function is classified VALIDATOR by Stage 2",
            )
            if stats["cbranch_ops"] < 2:
                coll_c.add(
                    "few_branches", "SemanticRisk",
                    f"Only {stats['cbranch_ops']} CBRANCH ops — validation logic may be absent",
                )
            if stats["has_allocator"] or stats["store_ops"] >= 2:
                coll_c.add(
                    "memory_writes", "SemanticRisk",
                    "Performs memory writes despite being classified as a validator",
                )
            if coll_c.count >= self._EMIT_MIN_ITEMS:
                candidates.append(SemRiskCandidate(
                    func_name    = name,
                    entry_addr   = entry_addr,
                    pattern_name = "validator_bypass",
                    collector    = coll_c,
                    vuln_type    = "check_bypass",
                    description  = (
                        "VALIDATOR function with insufficient branching relative "
                        "to the memory operations it performs."
                    ),
                ))

        return candidates

    # ── Op stats ──────────────────────────────────────────────────────────────

    def _compute_stats(self, ops: list[dict]) -> dict:
        mult_ops             = 0
        large_shift_ops      = 0
        arithmetic_ops       = 0
        store_ops            = 0
        cbranch_ops          = 0
        allocator_fn         = ""
        has_bounds_check     = False
        unchecked_arith_store = False
        last_was_arith       = False

        for op in ops:
            ot = op.get("op", "")

            if ot == "INT_MULT":
                mult_ops += 1
                arithmetic_ops += 1
                last_was_arith = True

            elif ot == "INT_LEFT":
                inputs = op.get("inputs") or []
                shift  = 0
                if len(inputs) >= 2:
                    rhs = inputs[1].get("name", "")
                    if rhs.startswith("const("):
                        try:
                            shift = int(rhs[6:rhs.index(")")], 0)
                        except Exception:
                            pass
                if shift >= 2:
                    large_shift_ops += 1
                arithmetic_ops += 1
                last_was_arith = True

            elif ot in ("INT_ADD", "INT_SUB", "INT_AND", "INT_OR",
                        "INT_XOR", "INT_RIGHT", "INT_SRIGHT",
                        "PTRADD", "PTRSUB"):
                arithmetic_ops += 1
                last_was_arith = True

            elif ot == "STORE":
                store_ops += 1
                if last_was_arith:
                    unchecked_arith_store = True
                last_was_arith = False

            elif ot == "CBRANCH":
                cbranch_ops += 1
                last_was_arith = False

            elif ot in ("INT_LESS", "INT_LESSEQUAL",
                        "INT_EQUAL", "INT_NOTEQUAL"):
                inputs = op.get("inputs") or []
                if any(
                    isinstance(i, dict) and i.get("name", "").startswith("const(")
                    for i in inputs
                ):
                    has_bounds_check = True
                last_was_arith = False

            elif ot in ("CALL", "CALLIND"):
                inputs = op.get("inputs") or []
                fn     = (inputs[0].get("name", "") if inputs else "").lower()
                if not allocator_fn:
                    for frag in _ALLOC_FRAGS:
                        if frag.lower() in fn:
                            allocator_fn = fn
                            break
                last_was_arith = False

            else:
                last_was_arith = False

        return {
            "mult_ops":              mult_ops,
            "large_shift_ops":       large_shift_ops,
            "arithmetic_ops":        arithmetic_ops,
            "store_ops":             store_ops,
            "cbranch_ops":           cbranch_ops,
            "allocator_fn":          allocator_fn,
            "has_allocator":         bool(allocator_fn),
            "has_bounds_check":      has_bounds_check,
            "unchecked_arith_store": unchecked_arith_store,
        }
