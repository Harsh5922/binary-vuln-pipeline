"""
stage3_evidence.py  —  Core Evidence Data Structures for Stage 3
================================================================
Implements the Evidence Package architecture described in the paper.

Key design principles:
  1. Stage 3 is an evidence producer, not a decision-maker.
     Stage 4 (LLM) makes the final vulnerability determination.

  2. No weights in evidence items — weights become paper hyperparameters.
     Instead: named observations with provenance and human-readable reasons.
     Emit threshold is item COUNT (two independent observations), not a score.

  3. No confidence decay — TransformationCount (integer) replaces magic
     decay constants. Source confidence propagates unchanged.

  4. Uncertainty is explicit — Stage 4 knows where the analysis is uncertain.

  5. Contradictions are surfaced — if Forward says safe but Semantic says risky,
     Stage 4 is told explicitly and asked to reason about it.

Architecture:
    Forward Analysis  ||  Backward Slice  ||  Semantic Risk  ||  Behavioral Prior
                                      ↓
                         Evidence Fusion Engine  (stage3_evidence_fusion.py)
                                      ↓
                             Evidence Package  (EvidenceVector)
                                      ↓
                           Stage 4 LLM Verification
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# ─── Source Role ─────────────────────────────────────────────────────────────
#
# Why these specific base confidences?
#
#   1.00  DIRECT_IO      — The kernel delivers raw attacker bytes with no
#                          library filtering (recv, read, fread).
#
#   0.98  NETWORK_WRAPPED — Same as DIRECT_IO but through TLS/BIO. The
#                          crypto layer validates the TLS handshake, not
#                          the application payload content.
#
#   0.90  LIBRARY_READER — Library reads from its own I/O buffer
#                          (png_crc_read, TIFFReadDirectory). The format
#                          header has been parsed, but FIELD VALUES are
#                          attacker-controlled.
#
#   0.90  PARSER_CALLBACK — Library parser invokes a callback with
#                          attacker-supplied content. Confidence = LIBRARY_READER
#                          because the call path is the same: format → callback.
#
#   0.85  DB_INPUT        — SQL bound parameter. The SQL tokenizer has
#                          processed the query but does not sanitize VALUES.
#
#   0.75  CLI_ARGUMENT    — argv / command-line. Process may require auth
#                          before reaching this code, so -0.25 discount.
#
#   0.65  ENVIRONMENT     — getenv. Less commonly controlled by attackers
#                          than argv. OS may sanitize.
#
#   0.00  UNKNOWN         — Not a recognized source.


class SourceRole(enum.Enum):
    DIRECT_IO       = "direct_io"        # recv, read, fread, fgets, scanf
    NETWORK_WRAPPED = "network_wrapped"  # SSL_read, BIO_read, WSARecv
    LIBRARY_READER  = "library_reader"   # png_crc_read, TIFFReadDirectory, xmlGetProp
    PARSER_CALLBACK = "parser_callback"  # *_callback*, xmlTextReaderRead
    DB_INPUT        = "db_input"         # sqlite3_value_*, sqlite3_column_*
    CLI_ARGUMENT    = "cli_argument"     # argv, getopt
    ENVIRONMENT     = "environment"      # getenv, secure_getenv
    UNKNOWN         = "unknown"          # not a recognized source

    @property
    def base_conf(self) -> float:
        """
        Base confidence for this source role.

        Derived from the category description above, not empirical measurements.
        Encodes one prior belief: direct kernel I/O is 1.0; each additional
        library processing layer reduces confidence by 0.05–0.10.
        """
        return {
            SourceRole.DIRECT_IO:       1.00,
            SourceRole.NETWORK_WRAPPED: 0.98,
            SourceRole.LIBRARY_READER:  0.90,
            SourceRole.PARSER_CALLBACK: 0.90,
            SourceRole.DB_INPUT:        0.85,
            SourceRole.CLI_ARGUMENT:    0.75,
            SourceRole.ENVIRONMENT:     0.65,
            SourceRole.UNKNOWN:         0.00,
        }[self]

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


# ─── Evidence Item ────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """
    One named observation from a Stage 3 analysis.

    Fields
    ------
    name    : short identifier, e.g. "arithmetic_present"
    source  : which analysis produced this item
              ("SemanticRisk", "ForwardAnalysis", "BackwardSlice", "BehavioralPrior")
    reason  : human-readable explanation — what was actually observed in the binary
    """
    name:   str
    source: str   # provenance: which Stage 3 sub-analysis produced this
    reason: str   # what was observed — shown to LLM as fact, not score


# ─── Evidence Collector ───────────────────────────────────────────────────────

class EvidenceCollector:
    """
    Collect named evidence observations during a Stage 3 sub-analysis.

    No weights — Stage 4 reasons over the individual items, not a sum.
    Emit threshold is item COUNT (see SemanticRiskAnalyzer._EMIT_MIN_ITEMS).

    Example:
        coll = EvidenceCollector()
        if role == "parser":
            coll.add("parser_role", "SemanticRisk",
                     "Function is classified PARSER by Stage 2")
        if mult_ops >= 1:
            coll.add("multiplication_present", "SemanticRisk",
                     f"{mult_ops} INT_MULT operations found")
        # Stage 4 sees: ["parser_role", "multiplication_present"]  (2 observations)
    """

    def __init__(self):
        self._items: list[EvidenceItem] = []

    def add(self, name: str, source: str, reason: str = "") -> None:
        self._items.append(EvidenceItem(name=name, source=source, reason=reason))

    @property
    def items(self) -> list[EvidenceItem]:
        return list(self._items)

    @property
    def count(self) -> int:
        return len(self._items)

    def has(self, name: str) -> bool:
        return any(i.name == name for i in self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0


# ─── Uncertainty ─────────────────────────────────────────────────────────────

@dataclass
class Uncertainty:
    """
    Flags that indicate where Stage 3 analysis has incomplete information.

    Stage 4 uses these to know where to be more cautious.  An EvidenceVector
    with HIGH uncertainty should be treated with more skepticism even if
    forward taint reached the sink.

    unknown_source    — no recognized external source found by forward analysis
    unknown_call      — one or more CALL targets were not resolved
    missing_summary   — inter-procedural summary was missing for a called function
    incomplete_cfg    — control-flow graph may be missing edges (indirect jumps)
    """
    unknown_source:   bool = False
    unknown_call:     bool = False
    missing_summary:  bool = False
    incomplete_cfg:   bool = False

    @property
    def is_high(self) -> bool:
        """True if two or more uncertainty flags are set."""
        return sum([
            self.unknown_source, self.unknown_call,
            self.missing_summary, self.incomplete_cfg,
        ]) >= 2

    @property
    def flags(self) -> list[str]:
        """Human-readable list of active uncertainty flags."""
        out = []
        if self.unknown_source:  out.append("source not identified by forward analysis")
        if self.unknown_call:    out.append("unknown function calls in taint path")
        if self.missing_summary: out.append("inter-proc summaries incomplete")
        if self.incomplete_cfg:  out.append("CFG may be incomplete (indirect jumps)")
        return out


# ─── Evidence Vector ─────────────────────────────────────────────────────────

@dataclass
class EvidenceVector:
    """
    Complete evidence package for one vulnerability candidate.

    This is the output of Stage 3E (Evidence Fusion Engine).
    Stage 4 (LLM) reasons over this evidence package instead of a single
    confidence float.  Each field is independently interpretable.

    Stage 3 does NOT say "this is a vulnerability."
    Stage 4 makes that determination from the evidence.
    """
    func_name:    str
    entry_addr:   str
    vuln_type:    str
    sink_fn:      str
    sink_class:   str = "GENERIC"   # ALLOCATOR | COPY | FORMAT | FREE | GENERIC

    # ── Source evidence (3A) ──────────────────────────────────────────────────
    source_role:      SourceRole = SourceRole.UNKNOWN
    source_fn:        str        = ""   # which function seeded the taint
    source_base_conf: float      = 0.0  # SourceRole.base_conf (no decay)

    # ── Forward taint evidence (3B) ───────────────────────────────────────────
    forward_reached:      bool      = False
    transformation_count: int       = 0     # op hops from source to sink (not decayed)
    path_has_mult:        bool      = False  # INT_MULT in taint path
    path_checked:         bool      = False  # bounds-checked before sink
    taint_path:           list[str] = field(default_factory=list)

    # ── Backward slice evidence (3B+) ─────────────────────────────────────────
    backward_reached: bool = False  # backward slice found external source?
    backward_role:    str  = ""     # source role label found by backward slicer
    backward_depth:   int  = 0      # hops to reach external source backward

    # ── Semantic evidence (3C) — item COUNT, not weighted score ───────────────
    semantic_role:  str                = "unknown"
    semantic_items: list[EvidenceItem] = field(default_factory=list)
    # semantic_score is the item count (float for compatibility with existing code)
    semantic_score: float              = 0.0

    # ── Behavioral Prior (renamed from "Pattern Memory") ─────────────────────
    # "I have seen this behavior before. Here is what happened historically."
    behavioral_similarity: float = 0.0  # structural similarity to known-TP patterns
    behavioral_prior_tp:   float = 0.5  # historical TP rate for this sink type

    # ── Uncertainty ───────────────────────────────────────────────────────────
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    # ── Contradictions ────────────────────────────────────────────────────────
    contradictions: list[str] = field(default_factory=list)

    # ── Inter-procedural evidence ─────────────────────────────────────────────
    interproc_confirmed: bool = False

    # ── Graph evidence (from Stage 2) ─────────────────────────────────────────
    reachability_score: float = 1.0

    # ── Candidate description ─────────────────────────────────────────────────
    description: str = ""

    def has_any_evidence(self) -> bool:
        """True if any analysis produced positive evidence for this candidate."""
        return (
            self.forward_reached
            or self.backward_reached
            or len(self.semantic_items) >= 2
        )

    def format_for_llm(self) -> str:
        """
        Format the evidence package as a human-readable block for the LLM prompt.

        Design rules (Rec 7):
          - No weights shown — facts only, not numbers
          - Provenance shown for each item — which analysis produced it
          - Uncertainty block — where to be cautious
          - Contradictions block — explicitly flag conflicting signals
        """
        lines: list[str] = [
            "EVIDENCE PACKAGE (Stage 3 Hybrid Analysis — Stage 4 decides):",
        ]

        # ── Source ───────────────────────────────────────────────────────────
        if self.source_role != SourceRole.UNKNOWN:
            lines.append(
                f"  Source identified:  {self.source_role.label}"
                f"  via {self.source_fn or '(no function name)'}"
            )
        else:
            lines.append("  Source:             NOT IDENTIFIED by forward analysis")

        # ── Forward taint ─────────────────────────────────────────────────────
        if self.forward_reached:
            lines.append(
                f"  Forward taint:      REACHED sink  "
                f"({self.transformation_count} P-code transformations)"
            )
            if self.taint_path:
                path_str = " -> ".join(self.taint_path[-6:])
                lines.append(f"    Path:             {path_str}")
            lines.append(
                f"    INT_MULT in path: {'YES — integer arithmetic present' if self.path_has_mult else 'NO'}"
            )
            lines.append(
                f"    Bounds checked:   {'YES — CBRANCH with constant comparison found' if self.path_checked else 'NO — no size guard detected'}"
            )
        else:
            lines.append("  Forward taint:      DID NOT reach sink")

        # ── Backward slice ────────────────────────────────────────────────────
        if self.backward_reached:
            lines.append(
                f"  Backward slice:     CONFIRMED external source"
                f"  ({self.backward_role})  depth={self.backward_depth}"
            )
        else:
            lines.append("  Backward slice:     external source NOT confirmed")

        # ── Semantic evidence ─────────────────────────────────────────────────
        if self.semantic_items:
            lines.append(
                f"  Semantic analysis:  {len(self.semantic_items)} observation(s)"
                f"  [function role: {self.semantic_role}]"
            )
            for item in self.semantic_items[:6]:
                lines.append(f"    [{item.source}] {item.name}:  {item.reason}")
        else:
            lines.append(
                f"  Semantic analysis:  no observations  [function role: {self.semantic_role}]"
            )

        # ── Behavioral Prior ──────────────────────────────────────────────────
        tp = self.behavioral_prior_tp
        if self.behavioral_similarity > 0.0:
            lines.append(
                f"  Behavioral prior:   similarity={self.behavioral_similarity:.2f}  "
                f"historical TP rate={tp:.0%} for this sink type"
            )
        else:
            lines.append("  Behavioral prior:   no historical data for this sink")

        # ── Inter-proc ────────────────────────────────────────────────────────
        lines.append(
            f"  Inter-proc:         {'confirmed via summary' if self.interproc_confirmed else 'not confirmed'}"
        )

        # ── Uncertainty ───────────────────────────────────────────────────────
        flags = self.uncertainty.flags
        if flags:
            level = "HIGH" if self.uncertainty.is_high else "MEDIUM"
            lines.append(f"  Uncertainty:        {level} — {'; '.join(flags)}")
        else:
            lines.append("  Uncertainty:        LOW")

        # ── Contradictions ────────────────────────────────────────────────────
        if self.contradictions:
            lines.append("")
            lines.append("  CONTRADICTORY EVIDENCE — reason about each carefully:")
            for c in self.contradictions:
                lines.append(f"    ! {c}")

        return "\n".join(lines)
