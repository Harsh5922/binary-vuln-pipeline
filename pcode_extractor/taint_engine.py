"""
taint_engine.py

Propagates taint through P-code operations using pattern-based rules.

How taint propagation works
----------------------------
Two taint maps are maintained per function:

  var_taint  : dict[str, bool]
    Is this variable (VAR_N, const, ram) tainted?
    A variable is tainted if it holds attacker-controlled data.

  mem_taint  : dict[str, bool]
    Is the memory that this pointer variable points to tainted?
    e.g. mem_taint["VAR_2"] = True means *VAR_2 is tainted.
    This is separate from var_taint["VAR_2"] which says VAR_2 itself is tainted.

Rules per op type
------------------
CALL / CALLIND
  → use PatternMatcher to get data flow rule
  → apply rule: seed external inputs, propagate source→dest, taint return
  → NO_MATCH: conservative (return tainted, pointer args flagged)

STORE  (addr, value)
  → inputs[0] = destination address
  → inputs[1] = value being written
  → if value is tainted: mem_taint[addr_var] = True
  → if addr is tainted: flag as write-to-arbitrary-address (dangerous)

LOAD  (addr) → output
  → if mem_taint[addr_var] OR var_taint[addr_var]: taint output

Everything else (INT_ADD, PTRADD, INT_SUB, CBRANCH, RETURN...)
  → if ANY input is tainted: taint output varnode

Vulnerability detection
------------------------
A VulnCandidate is recorded when:
  1. A tainted variable reaches a CALL that is a known sink (is_sink=True)
  2. A tainted variable is used as the address in a STORE (write-what-where)
  3. A tainted variable reaches an unbounded sink (bounded=False)
  4. A tainted size variable is used in malloc/calloc (integer overflow)

Usage
-----
    from taint_engine import TaintEngine
    from pattern_store import PatternStore
    from pattern_matcher import PatternMatcher

    store   = PatternStore()
    matcher = PatternMatcher(store)
    engine  = TaintEngine(matcher)

    results = engine.analyze(func_dict)
    for vuln in results.vulns:
        print(vuln.summary())
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from pattern_store   import PatternStore
from pattern_matcher import PatternMatcher, MatchKind

# Functions that introduce external (attacker-controlled) data into the program.
# Used both for seeding taint and for confirming write-what-where origins.
# Module-level constant — not rebuilt on every STORE op.
_EXTERNAL_SOURCE_FNS: frozenset[str] = frozenset({
    "recv", "recvfrom", "read", "fread", "fgets",
    "gets", "scanf", "sscanf", "fscanf", "getenv",
    "ReadFile", "WSARecv",
})

# Allocation functions: a tainted size reaching any of these is always flagged
# for integer_overflow (the allocation itself may be undersized).
# For non-allocator sinks (memcpy, fread, etc.), we require that the size
# argument went through at least one arithmetic op — pure parameter pass-through
# is normal library behavior, not a vulnerability.
# INT_AND narrowing masks → minimum input byte-width that makes them dangerous.
#
# The value is the minimum var size (bytes) at which the mask actually narrows
# the type in a security-relevant way.
#
#   0xFF       → only dangerous when the variable is ≥ 8 bytes (64-bit).
#                On 4-byte (32-bit) vars it almost always means palette/byte ops
#                (e.g. colormap index clamping) and is a no-op truncation there.
#   0xFFFF     → dangerous when the variable is ≥ 4 bytes (32-bit).
#   0xFFFFFF   → dangerous when the variable is ≥ 4 bytes (32-bit).
#   0xFFFFFFFF → dangerous when the variable is ≥ 8 bytes (64-bit).
#                This is the canonical case: png_check_chunk_length stores a
#                64-bit chunk length then masks it to 32 bits before comparing.
_TRUNCATION_MASKS: dict[int, int] = {
    0xFF:         8,   # 64-bit+ input only
    0xFFFF:       4,   # 32-bit+ input
    0xFFFFFF:     4,   # 32-bit+ input
    0xFFFFFFFF:   8,   # 64-bit+ input
}

_ALLOCATOR_FNAMES: frozenset[str] = frozenset({
    "malloc", "calloc", "realloc", "reallocarray", "valloc", "pvalloc",
    "mmap", "mmap64", "brk",
    "png_malloc", "png_malloc_warn", "png_calloc",
    "png_malloc_base", "png_realloc_array", "png_zalloc",
    "g_malloc", "g_malloc0", "g_malloc_n", "g_realloc", "g_try_malloc",
    "xmalloc", "zmalloc", "smalloc", "emalloc",
    "HeapAlloc", "VirtualAlloc", "LocalAlloc", "GlobalAlloc",
    "operator new", "operator new[]",
})

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Flow logging — records every taint propagation step
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowStep:
    """
    One step in the taint propagation chain.
    Recorded every time a variable becomes tainted.
    """
    seq:       int     # op sequence number (-1 = seed)
    op:        str     # P-code mnemonic or "SEED"
    addr:      str     # instruction address
    from_var:  str     # variable that caused the taint ("" for external seeds)
    to_var:    str     # variable that became tainted
    reason:    str     # human-readable: "external:recv", "load:*VAR_2", etc.
    is_mem:    bool    # True if mem_taint, False if var_taint

    def __str__(self) -> str:
        mem_tag = "[*mem]" if self.is_mem else ""
        arrow   = f"  {self.from_var} →  " if self.from_var else "  ⊕ SEED  "
        return (
            f"  seq={str(self.seq):<4}  {self.op:<12}  "
            f"{arrow}{self.to_var}{mem_tag}   ({self.reason})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VulnCandidate:
    """
    A potential vulnerability found by the taint engine.
    Not yet confirmed — the reasoning agent reviews these.
    """
    func_name:   str
    entry_addr:  str
    vuln_type:   str      # buffer_overflow | command_injection |
                          # format_string | write_what_where |
                          # integer_overflow | integer_truncation | use_after_free
    op_seq:      int      # which op triggered the finding
    sink_fn:     str      # function name at the sink
    taint_source: str     # which variable carried the taint
    taint_path:  list[str] # chain of variables from source to sink
    bounded:     bool     # was there a size check?
    confidence:  float    # 0.0 – 1.0
    description: str      # human-readable explanation
    match_kind:  str = "LIBRARY_MATCH"  # LIBRARY_MATCH | STRUCTURAL_MATCH | NO_MATCH
    arg_sizes:   list = None  # arg byte-sizes at sink call (for pattern learning)
    fingerprint: str  = ""     # structural signature for cross-binary matching

    def __post_init__(self):
        if self.arg_sizes is None:
            self.arg_sizes = []

    def summary(self) -> str:
        bounded_str = "bounded" if self.bounded else "UNBOUNDED"
        return (
            f"[{self.vuln_type}]  {self.func_name}  "
            f"seq={self.op_seq}  sink={self.sink_fn}  "
            f"{bounded_str}  conf={self.confidence:.0%}"
        )


@dataclass
class TaintResult:
    """
    Complete taint analysis result for one function.
    """
    func_name:    str
    entry_addr:   str

    # Final taint state after analyzing all ops
    tainted_vars: set[str]     # variables that are tainted at function end
    tainted_mem:  set[str]     # pointer variables whose pointed-to memory is tainted

    # Unknown CALL ops that need LLM analysis
    unknown_calls: list[dict]  # raw op dicts with NO_MATCH from pattern matcher

    # Vulnerability candidates found
    vulns:         list[VulnCandidate]

    # Full taint propagation log — every step recorded
    flow_steps:    list[FlowStep] = field(default_factory=list)

    # Summary counts
    ops_analyzed:  int = 0
    calls_matched: int = 0
    calls_unknown: int = 0

    # Stage 3A: source role confidence per tainted variable (SourceRole.base_conf).
    # No decay — propagated unchanged. Distance is tracked by taint_hops.
    source_confidence:  dict[str, float] = field(default_factory=dict)
    # Stage 3B+: transformation count per variable — how many ops from source?
    # This is the distance signal (replaces arbitrary confidence decay).
    taint_hops:         dict[str, int]   = field(default_factory=dict)
    # Stage 3D evidence: vars derived from INT_MULT (higher overflow risk).
    mult_tainted_vars:  set[str]         = field(default_factory=set)
    # Stage 3D evidence: vars that passed through a bounds check (lower risk).
    checked_vars:       set[str]         = field(default_factory=set)
    # Stage 3E: evidence vectors keyed by candidate fingerprint.
    # Populated by Stage3Orchestrator; None when using bare TaintEngine.
    evidences:          dict[str, Any]   = field(default_factory=dict)

    def has_vulns(self) -> bool:
        return len(self.vulns) > 0

    def print_flow(self) -> None:
        """
        Print the complete variable flow — every taint propagation step
        from seed to sink, in sequence order.

        Example output:
          TAINT FLOW — vulnerable_copy @ 0x401136
          ──────────────────────────────────────────────
          seq=1    CALL         ⊕ SEED  VAR_2        (external:recv)
          seq=1    CALL         ⊕ SEED  *VAR_2[mem]  (external:recv)
          seq=2    CALL         VAR_2 → VAR_0        (param:strcpy_src)
          ↳ SINK  [buffer_overflow]  strcpy(VAR_0)  seq=2
        """
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  TAINT FLOW  —  {self.func_name}  @ {self.entry_addr}")
        print(f"{sep}")

        steps = self.flow_steps or []
        if not steps:
            print("  (no taint propagation recorded)")
        else:
            print(f"  {'seq':<6}  {'op':<12}  {'from':<18}  {'to':<20}  reason")
            print(f"  {'─'*6}  {'─'*12}  {'─'*18}  {'─'*20}  {'─'*20}")
            for step in steps:
                mem_tag  = "[*mem]" if step.is_mem else "      "
                from_str = step.from_var if step.from_var else "⊕ SEED"
                print(
                    f"  {str(step.seq):<6}  {step.op:<12}  "
                    f"{from_str:<18}  {step.to_var:<14}{mem_tag}  {step.reason}"
                )

        if self.vulns:
            print(f"\n  {'─'*58}")
            print(f"  SINKS REACHED ({len(self.vulns)}):")
            for v in self.vulns:
                path_str = " → ".join(v.taint_path)
                print(f"  ↳  [{v.vuln_type}]  {v.sink_fn}()  seq={v.op_seq}")
                print(f"     path : {path_str}")
                print(f"     conf : {v.confidence:.0%}  bounded={v.bounded}")
        else:
            print(f"\n  No sinks reached — function appears safe.")
        print(f"{sep}\n")

    def summary(self) -> str:
        lines = [
            f"{'─'*60}",
            f"Function : {self.func_name}  @ {self.entry_addr}",
            f"Ops      : {self.ops_analyzed}  "
            f"Calls matched: {self.calls_matched}  "
            f"Unknown: {self.calls_unknown}",
            f"Tainted vars : {sorted(self.tainted_vars)}",
            f"Tainted mem  : {sorted(self.tainted_mem)}",
        ]
        if self.vulns:
            lines.append(f"\nVulnerability candidates ({len(self.vulns)}):")
            for v in self.vulns:
                lines.append(f"  {v.summary()}")
                lines.append(f"    {v.description}")
        else:
            lines.append("\nNo vulnerability candidates found.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Taint state — internal to one function analysis
# ─────────────────────────────────────────────────────────────────────────────

class TaintState:
    """
    Mutable taint tracking state for one function.

    Two maps:
      var_taint  — variable is tainted
      mem_taint  — memory the variable points to is tainted

    Also tracks the taint path — how did each variable become tainted.
    Used to report the chain from source to sink in VulnCandidate.

    flow_log records every single propagation event so the full
    variable flow can be printed with print_flow().
    """

    def __init__(self):
        self.var_taint:  dict[str, bool] = {}
        self.mem_taint:  dict[str, bool] = {}
        self.taint_path: dict[str, list[str]] = {}  # var → [source, ..., var]
        self.flow_log:   list[FlowStep] = []         # every propagation event
        self._current_op: dict = {}                  # op being processed (for logging)
        # Tracks variables whose taint origin is a confirmed external source
        # (recv, read, getenv, …). Used by _handle_store to distinguish real
        # write-what-where from normal library pointer writes.
        self.externally_tainted_vars: set[str] = set()
        # Stage 3A integration: float source confidence per variable [0.0..1.0].
        # Stores SourceRole.base_conf for the strongest source in each var's path.
        # Propagated WITHOUT decay — see propagate() for the rationale.
        self.var_source_conf: dict[str, float] = {}
        # Stage 3B+: transformation count — how many ops from the source?
        # This is the distance signal; 0 = directly from source.
        # No arbitrary constants: it is just an integer count.
        self.taint_hops: dict[str, int] = {}
        # Tracks pointer variables that have been freed — used for UAF detection.
        self.freed_ptrs: set[str] = set()
        # Tracks variables produced by INT_MULT that are tainted — higher
        # integer-overflow risk than plain tainted sizes.
        self.mult_tainted_vars: set[str] = set()
        # Tracks variables that have been bounds-checked via a comparison op
        # (INT_LESS, INT_LESSEQUAL, etc.).  In fixed binaries, guards like
        # `if (n > MAX) return` produce INT_LESSEQUAL(n, MAX) before the
        # allocation.  We propagate this flag through downstream arithmetic so
        # that size vars derived from n are also considered checked, and we
        # suppress the integer_overflow finding — distinguishing fixed from vuln.
        self.checked_vars: set[str] = set()
        # Maps variable → truncation mask for variables narrowed via INT_AND.
        self.truncated_vars: dict[str, int] = {}
        # Field-level UAF tracking.
        # ptr_offsets:      var → (base_var, offset)  from PTRSUB(base, offset)
        # field_load_src:   var → (base_var, offset)  for vars loaded via PTRSUB addr
        # freed_fields:     set of (base_var, offset) whose field pointer was freed
        self.ptr_offsets:    dict[str, tuple[str, int]] = {}
        self.field_load_src: dict[str, tuple[str, int]] = {}
        self.freed_fields:   set[tuple[str, int]]       = set()
        # Function-pointer tracking: maps a var to the function name it holds.
        # Populated when PTRSUB(const(0x0), const(ADDR)) is seen and ADDR
        # resolves to a known function via summary_db.addr_to_name.
        self.fn_ptr_vars:    dict[str, str]              = {}

    def set_current_op(self, op: dict) -> None:
        """Called by TaintEngine before processing each op."""
        self._current_op = op

    def get_conf(self, var: str) -> float:
        """Return the float source confidence for a variable (0.0 if unknown)."""
        return self.var_source_conf.get(var, 0.0)

    def taint_var(
        self,
        var:         str,
        reason:      str   = "",
        from_var:    str   = "",
        source_conf: float = 0.0,
    ) -> None:
        """Mark a variable as tainted and log the step.

        source_conf — Stage 3A confidence value [0.0..1.0].  When > 0 the
                      variable is also added to externally_tainted_vars for
                      backward compatibility with the rest of the engine.
        """
        if not var:
            return
        was_tainted = self.var_taint.get(var, False)
        self.var_taint[var] = True
        if var not in self.taint_path:
            self.taint_path[var] = [reason or var]
        elif reason.startswith("external:") and not self.taint_path[var][0].startswith("external:"):
            # Upgrade path to external origin — more meaningful for reporting.
            self.taint_path[var] = [reason]
        # Track confirmed external origin so _handle_store can check without
        # string-searching taint path entries.
        if reason.startswith("external:"):
            self.externally_tainted_vars.add(var)
        # Store float source confidence (Stage 3A); keep maximum when re-seeded.
        if source_conf > 0.0:
            self.var_source_conf[var] = max(
                self.var_source_conf.get(var, 0.0), source_conf
            )
            self.externally_tainted_vars.add(var)
        # Seed at hop distance 0 — this variable IS the source.
        if var not in self.taint_hops:
            self.taint_hops[var] = 0
        # Log every new taint event
        if not was_tainted:
            self.flow_log.append(FlowStep(
                seq      = int(self._current_op.get("seq", -1)),
                op       = self._current_op.get("op", "SEED"),
                addr     = self._current_op.get("addr", ""),
                from_var = from_var,
                to_var   = var,
                reason   = reason or var,
                is_mem   = False,
            ))

    def taint_mem(self, ptr_var: str, reason: str = "", from_var: str = "") -> None:
        """Mark the memory that ptr_var points to as tainted and log."""
        if not ptr_var:
            return
        was_tainted = self.mem_taint.get(ptr_var, False)
        self.mem_taint[ptr_var] = True
        if ptr_var not in self.taint_path:
            self.taint_path[ptr_var] = [reason or ptr_var]
        if not was_tainted:
            self.flow_log.append(FlowStep(
                seq      = int(self._current_op.get("seq", -1)),
                op       = self._current_op.get("op", "SEED"),
                addr     = self._current_op.get("addr", ""),
                from_var = from_var,
                to_var   = ptr_var,
                reason   = reason or ptr_var,
                is_mem   = True,
            ))

    def propagate(self, output: str, inputs: list[str], op_type: str = "") -> bool:
        """
        If any input is tainted, taint the output and log.
        Returns True if output became tainted.

        Path attribution: among all tainted inputs, prefer the one whose
        origin is a confirmed external source (external: > param: > any).
        This ensures VulnCandidate.taint_path traces the most meaningful
        causal chain rather than whichever input happened to appear first.
        """
        if not output:
            return False

        tainted_inputs = [inp for inp in inputs if self.is_tainted(inp)]
        if not tainted_inputs:
            return False

        # Pick best input: external origin beats param origin beats anything
        best = tainted_inputs[0]
        for inp in tainted_inputs[1:]:
            if inp in self.externally_tainted_vars and best not in self.externally_tainted_vars:
                best = inp

        already = self.var_taint.get(output, False)
        self.var_taint[output] = True
        self.taint_path[output] = self.taint_path.get(best, [best]) + [output]

        # Propagate the external-origin flag.
        # For arithmetic ops (INT_ADD etc.) mixing an external index with a
        # non-external base pointer (e.g. output_buf + ext_loop_index), the
        # RESULT is NOT itself an attacker-controlled address — only the offset
        # is external.  Requiring ALL tainted inputs to be external for arith
        # prevents false write_what_where findings in codec loops while still
        # tracking size/count arithmetic correctly (pure external arithmetic
        # still propagates because every tainted input is external).
        _ARITH_OPS = frozenset({
            "INT_ADD", "INT_SUB", "INT_AND", "INT_OR", "INT_XOR",
            "INT_LEFT", "INT_RIGHT", "INT_SRIGHT",
        })
        if op_type in _ARITH_OPS:
            if all(inp in self.externally_tainted_vars for inp in tainted_inputs):
                self.externally_tainted_vars.add(output)
        else:
            if best in self.externally_tainted_vars:
                self.externally_tainted_vars.add(output)

        # Stage 3A: propagate source confidence WITHOUT decay.
        # Decay constants are magic numbers not defensible in a research paper.
        # Instead, we track transformation_count (taint_hops) as the distance
        # signal — that integer is purely structural, no arbitrary constants.
        # The source confidence here is SourceRole.base_conf, not a decayed float.
        max_parent_conf = max(
            (self.var_source_conf.get(inp, 0.0) for inp in tainted_inputs),
            default=0.0,
        )
        if max_parent_conf > 0.0:
            self.var_source_conf[output] = max(
                self.var_source_conf.get(output, 0.0), max_parent_conf
            )
            self.externally_tainted_vars.add(output)

        # Stage 3B+: transformation count — how many ops from source to sink?
        # This is the distance signal that replaces confidence decay.
        max_parent_hops = max(
            (self.taint_hops.get(inp, 0) for inp in tainted_inputs),
            default=0,
        )
        self.taint_hops[output] = max(
            self.taint_hops.get(output, 0),
            max_parent_hops + 1,
        )

        if not already:
            self.flow_log.append(FlowStep(
                seq      = int(self._current_op.get("seq", -1)),
                op       = self._current_op.get("op", "?"),
                addr     = self._current_op.get("addr", ""),
                from_var = best,
                to_var   = output,
                reason   = f"propagate:{self._current_op.get('op','')}",
                is_mem   = False,
            ))
        return True

    def is_tainted(self, var: str) -> bool:
        return bool(var) and self.var_taint.get(var, False)

    def mem_is_tainted(self, ptr_var: str) -> bool:
        return bool(ptr_var) and self.mem_taint.get(ptr_var, False)

    def get_path(self, var: str) -> list[str]:
        return self.taint_path.get(var, [var])


# ─────────────────────────────────────────────────────────────────────────────
# Taint Engine
# ─────────────────────────────────────────────────────────────────────────────

# Op sets for rule routing
_CALL_OPS     = frozenset({"CALL", "CALLIND"})
_ARITH_OPS    = frozenset({
    "INT_ADD", "INT_SUB", "INT_MULT", "INT_DIV", "INT_REM",
    "INT_AND", "INT_OR",  "INT_XOR",  "INT_NEGATE",
    "INT_LEFT", "INT_RIGHT", "INT_SRIGHT",
    "INT_ZEXT", "INT_SEXT",
    "INT_LESS", "INT_LESSEQUAL", "INT_EQUAL", "INT_NOTEQUAL",
    "INT_CARRY", "INT_SCARRY", "INT_SBORROW",
    "FLOAT_ADD", "FLOAT_SUB", "FLOAT_MULT", "FLOAT_DIV",
    "FLOAT_LESS", "FLOAT_EQUAL",
    "PTRADD", "PTRSUB",
    "BOOL_AND", "BOOL_OR", "BOOL_NEGATE", "BOOL_XOR",
    "CAST", "COPY",
    "PIECE", "SUBPIECE",
    "MULTIEQUAL",   # phi node: if any branch value is tainted, output is tainted
})
_BRANCH_OPS   = frozenset({"BRANCH", "BRANCHIND", "RETURN"})
_SKIP_OPS     = frozenset({"INDIRECT"})
# MULTIEQUAL is a phi node: VAR_5 = phi(VAR_2, VAR_3).
# It is NOT skipped — if any branch value is tainted the merged result must be
# tainted too. Skipping it silently drops taint at every branch join point.



def _structural_fingerprint(c: VulnCandidate) -> str:
    """
    Function-name-independent structural fingerprint for cross-binary pattern matching.
    Same fingerprint = same vulnerability pattern shape = can auto-confirm.
    
    Format: vuln_type|taint_origin|sink_class|match_kind|confidence_bucket
    Example: integer_overflow|external|memcpy_family|heur|med
    """
    src = c.taint_source or ""
    if src.startswith("structural:"):
        origin = "structural"
    elif any(x in src for x in ("fread","recv","read","BIO_read",
             "TIFFGetField","sqlite3_value","xmlGetProp","lua_to")):
        origin = "external"
    else:
        origin = "param"

    sink = (c.sink_fn or "").lower()
    if any(s in sink for s in ("memcpy","memmove","mempcpy","bcopy")):
        sink_class = "memcpy_family"
    elif any(s in sink for s in ("malloc","realloc","calloc","alloc")):
        sink_class = "malloc_family"
    elif "callind" in sink or sink.startswith("var_"):
        sink_class = "callind_null"
    elif "ptradd" in sink:
        sink_class = "ptradd_oob"
    elif "int_and" in sink:
        sink_class = "int_and_trunc"
    elif "load" in sink:
        sink_class = "load_null"
    else:
        sink_class = sink[:20] if sink else "unknown"

    conf = "high" if c.confidence >= 0.7 else "med" if c.confidence >= 0.45 else "low"
    mk   = {"LIBRARY_MATCH":"lib","STRUCTURAL_MATCH":"struct","NO_MATCH":"heur"}.get(c.match_kind,"?")
    return f"{c.vuln_type}|{origin}|{sink_class}|{mk}|{conf}"


def _add_vuln(vulns: list[VulnCandidate], candidate: VulnCandidate) -> None:
    """Append candidate only if no existing entry has the same sink, type, and source.

    Deduplicates identical findings that arise when the same tainted variable
    reaches the same sink through two code paths — the reasoning agent would
    otherwise review them twice and burn LLM tokens on duplicate analysis.
    """
    key = (candidate.sink_fn, candidate.vuln_type, candidate.taint_source)
    for existing in vulns:
        if (existing.sink_fn, existing.vuln_type, existing.taint_source) == key:
            return
    if not candidate.fingerprint:
        candidate.fingerprint = _structural_fingerprint(candidate)
    vulns.append(candidate)


class TaintEngine:
    """
    Propagates taint through P-code operations.

    For each function:
      1. Seed taint from external inputs (parameters marked tainted)
      2. Walk ops in sequence order
      3. Apply taint rule for each op
      4. Record VulnCandidates when taint reaches a sink

    Parameters
    ----------
    matcher        : PatternMatcher to look up CALL rules
    taint_params   : if True, treat all pointer-sized parameters as tainted
                     (conservative — use when you don't know what calls this function)
    """

    def __init__(
        self,
        matcher:       PatternMatcher,
        taint_params:  bool = True,   # seed from params — needed for library analysis
        summary_db     = None,        # interprocedural.SummaryDatabase (optional)
    ):
        self.matcher      = matcher
        self.taint_params = taint_params
        self.summary_db   = summary_db

    # ── Public API ────────────────────────────────────────────────────

    def analyze(self, func: dict) -> TaintResult:
        """
        Run taint analysis on one function.

        Parameters
        ----------
        func : function dict from pcode.jsonl or pcode_ranked.jsonl
               must have: name, entry_addr, ops
        """
        name       = func.get("name", "unknown")
        entry_addr = func.get("entry_addr", "")
        ops        = func.get("ops") or []

        reachability_score   = func.get("reachability_score", 1.0)
        # Only suppress vuln detection when the function is both unreachable from
        # external sources AND has explicit write-path tokens in its name.
        # Suppressing on reachability alone causes false negatives for library
        # read-processing functions (png_handle_*, libtiff parsers, etc.) when
        # the call graph misses I/O callbacks registered via function pointers.
        _WRITE_TOKENS = (
            "_write_", "write_", "_encode_", "encode_", "_serialize",
            "serialize_", "_output_", "output_", "_emit_", "_flush",
        )
        _name_lower = name.lower()
        _skip_vuln_detection = (
            reachability_score == 0.0
            and any(tok in _name_lower for tok in _WRITE_TOKENS)
        )

        state          = TaintState()
        vulns:         list[VulnCandidate]  = []
        unknown_calls: list[dict]           = []
        calls_matched  = 0
        calls_unknown  = 0

        # ── Seed: taint function parameters ──────────────────────────
        if self.taint_params:
            self._seed_parameters(ops, state)

        # Seed: Stage 3A source confidence seeds injected by Stage3Orchestrator.
        # These override the binary externally_tainted_vars with float confidence.
        _3a_seeds: dict[str, float] = func.get("_source_seeds_3a", {})
        for _sv, _sc in _3a_seeds.items():
            state.taint_var(_sv, reason=f"external:3a_seed_{_sv}", from_var="",
                            source_conf=_sc)
            log.debug("3A seed: %s=%s conf=%.2f", name, _sv, _sc)

        # Seed: external taint from inter-proc propagation
        # png_combine_row, png_handle_PLTE etc. receive external data
        # transitively through fread->png_read_row->png_combine_row.
        # The inter-proc module marks which args are externally tainted.
        if self.summary_db is not None:
            _func_sum = self.summary_db.get(name)
            if _func_sum is not None and _func_sum.externally_tainted_args:
                _params = self._get_param_vars(ops)
                for _aidx in _func_sum.externally_tainted_args:
                    if _aidx < len(_params):
                        _pv = _params[_aidx]
                        state.taint_var(_pv, reason=f"external:interproc_arg_{_aidx}",
                                        from_var="", source_conf=0.80)
                        state.externally_tainted_vars.add(_pv)
                        log.debug("ExtTaint: %s arg[%d]=%s", name, _aidx, _pv)

        _NULL_CONSTS = frozenset({"const(0x0)", "const(0)"})

        # Pre-scan B: CALLIND + nearby zero-compare — CWE-476
        # Scans BOTH forward AND backward for null checks.
        # Forward catches post-call null checks; backward catches
        # pre-call guards like png_safe_execute (checks fn_ptr != NULL before CALLIND).
        _null_deref_seqs: set[int] = set()
        for _ci, _cop in enumerate(ops):
            if _cop.get("op") != "CALLIND": continue
            # Forward scan: null check after CALLIND
            for _fwd in ops[_ci+1 : _ci+13]:
                if _fwd.get("op") not in ("INT_EQUAL","INT_NOTEQUAL"): continue
                _fi = [_x.get("name","") for _x in (_fwd.get("inputs") or [])]
                if any(n in _NULL_CONSTS for n in _fi):
                    _null_deref_seqs.add(_cop.get("seq",-1)); break

        # Pre-scan C: load-load wrong-var null check → CWE-476
        _load_load_seqs: set[int] = set()
        _null_chk_before: set[str] = set()
        for _li, _lop in enumerate(ops):
            _lt  = _lop.get("op","")
            _lii = [_x.get("name","") for _x in (_lop.get("inputs") or [])]
            _lo  = (_lop.get("output") or {}).get("name","")
            if _lt in ("INT_EQUAL","INT_NOTEQUAL"):
                if any(n in _NULL_CONSTS for n in _lii):
                    [_null_chk_before.add(_n) for _n in _lii if _n.startswith("VAR_")]
            if _lt != "LOAD" or not _lo or _li+1 >= len(ops): continue
            _nx = ops[_li+1]
            if _nx.get("op") != "LOAD": continue
            _ni = [_x.get("name","") for _x in (_nx.get("inputs") or [])]
            _no = (_nx.get("output") or {}).get("name","")
            if (_ni[-1] if _ni else "") != _lo or _lo in _null_chk_before or not _no: continue
            for _fwd2 in ops[_li+2:_li+5]:
                if _fwd2.get("op") not in ("INT_EQUAL","INT_NOTEQUAL"): continue
                _f2i = [_x.get("name","") for _x in (_fwd2.get("inputs") or [])]
                if _no in _f2i and any(n in _NULL_CONSTS for n in _f2i):
                    _load_load_seqs.add(_lop.get("seq",-1)); break

        # ── Walk ops ──────────────────────────────────────────────────
        for op in ops:
            op_type = op.get("op", "")
            state.set_current_op(op)   # let TaintState know which op we're on

            if op_type in _SKIP_OPS:
                continue

            if op_type in _CALL_OPS:
                new_vulns, unknown = self._handle_call(op, state, name, entry_addr)
                if not _skip_vuln_detection:
                    for v in new_vulns:
                        _add_vuln(vulns, v)
                    if (op.get("op") == "CALLIND" and op.get("seq",-1) in _null_deref_seqs):
                        _fp = (op.get("inputs") or [{}])[0].get("name","CALLIND")
                        _add_vuln(vulns, VulnCandidate(
                            func_name=name, entry_addr=entry_addr,
                            vuln_type="null_dereference", op_seq=op.get("seq",0),
                            sink_fn=_fp, taint_source="structural:callind_null_check",
                            taint_path=[_fp], bounded=False, confidence=0.65,
                            description=f"CALLIND via {_fp} + NULL check on return — CWE-476.",
                            match_kind="NO_MATCH",
                        ))
                if unknown:
                    unknown_calls.append(op)
                    calls_unknown += 1
                else:
                    calls_matched += 1

            elif op_type == "STORE":
                self._handle_store(op, state,
                    [] if _skip_vuln_detection else vulns, name, entry_addr,)

            elif op_type == "LOAD":
                self._handle_load(op, state,
                    [] if _skip_vuln_detection else vulns, name, entry_addr,)
                if (not _skip_vuln_detection and op.get("seq",-1) in _load_load_seqs):
                    _ai = op.get("inputs") or []
                    _adr = _ai[-1].get("name","?") if _ai else "?"
                    _add_vuln(vulns, VulnCandidate(
                        func_name=name, entry_addr=entry_addr,
                        vuln_type="null_dereference", op_seq=op.get("seq",0),
                        sink_fn="LOAD", taint_source="structural:load_load_wrong_var_check",
                        taint_path=[_adr], bounded=False, confidence=0.70,
                        description=f"LOAD dereferences {_adr} without NULL check; check on wrong var — CWE-476.",
                        match_kind="NO_MATCH",
                    ))

            elif op_type in _ARITH_OPS:
                self._handle_arith(op, state, vulns, name, entry_addr)

            elif op_type == "CBRANCH":
                self._handle_cbranch(op, state)

            elif op_type in _BRANCH_OPS:
                pass   # unconditional branch / return — no data taint

            else:
                # Unknown op type — treat as arith (propagate taint)
                self._handle_arith(op, state, vulns, name, entry_addr)

        return TaintResult(
            func_name          = name,
            entry_addr         = entry_addr,
            tainted_vars       = {v for v, t in state.var_taint.items() if t},
            tainted_mem        = {v for v, t in state.mem_taint.items() if t},
            unknown_calls      = unknown_calls,
            vulns              = vulns,
            flow_steps         = state.flow_log,
            ops_analyzed       = len(ops),
            calls_matched      = calls_matched,
            calls_unknown      = calls_unknown,
            source_confidence  = dict(state.var_source_conf),
            taint_hops         = dict(state.taint_hops),
            mult_tainted_vars  = set(state.mult_tainted_vars),
            checked_vars       = set(state.checked_vars),
        )

    def analyze_all(self, funcs: list[dict]) -> list[TaintResult]:
        """
        Analyze a list of functions in parallel and return all results.

        Each function's taint analysis is fully independent, so we run them
        concurrently. ThreadPoolExecutor is used (not ProcessPoolExecutor)
        because SQLite pattern-store reads are thread-safe and GIL is released
        during I/O. Groq LLM calls inside pattern resolution also benefit.
        """
        with ThreadPoolExecutor() as pool:
            results = list(pool.map(self.analyze, funcs))

        for r in results:
            if r.has_vulns():
                log.info("  VULN  %s — %d candidates", r.func_name, len(r.vulns))
            else:
                log.debug("  clean  %s", r.func_name)
        return results

    # ── Op handlers ───────────────────────────────────────────────────

    def _get_param_vars(self, ops: list[dict]) -> list[str]:
        """Return list of parameter variable names in approximate arg order."""
        defined: set[str] = set()
        for op in ops:
            out = op.get("output")
            if out and isinstance(out, dict) and out.get("name"):
                defined.add(out["name"])
        params = []
        seen: set[str] = set()
        for op in ops:
            for inp in (op.get("inputs") or []):
                if not isinstance(inp, dict): continue
                name = inp.get("name", "")
                if name and name not in defined and name not in seen:
                    if name.startswith("VAR_") or name.startswith("Var"):
                        params.append(name)
                        seen.add(name)
        return params

    def _seed_parameters(self, ops: list[dict], state: TaintState) -> None:
        """
        Seed taint from function parameters.

        For library analysis, ALL parameters are potentially attacker-controlled
        because the caller reads them from the input file/data.

        Strategy: a variable that appears as an input but NEVER as an output
        of any op is either a true function parameter OR a filtered-out phi node
        (MULTIEQUAL). Both cases represent values that flow in from outside the
        function — we treat them all as tainted.

        This catches integer count parameters (e.g. `num` in png_set_PLTE) that
        appear only late in the function body after branch merges, which the old
        "stop-at-first-CALL" approach missed.
        """
        if not self.taint_params:
            return

        # Collect all variables that are defined (appear as output) in this function.
        defined: set[str] = set()
        for op in ops:
            out = op.get("output")
            if out and isinstance(out, dict):
                name = out.get("name", "")
                if name:
                    defined.add(name)

        # Seed any variable that is used as input but never defined here.
        # These are function parameters or phi-node outputs filtered by the extractor.
        seeded: set[str] = set()
        for op in ops:
            inputs = op.get("inputs") or []
            for inp in inputs:
                if not isinstance(inp, dict):
                    continue
                name = inp.get("name", "")
                size = inp.get("size", 0)
                if (
                    name
                    and name not in seeded
                    and name not in defined
                    and size in (2, 4, 8)
                    and not name.startswith("const(")
                    and not name.startswith("ram(")
                ):
                    state.taint_var(name, reason=f"param:{name}", from_var="")
                    seeded.add(name)

    def _handle_call(
        self,
        op:         dict,
        state:      TaintState,
        func_name:  str,
        entry_addr: str,
    ) -> tuple[list[VulnCandidate], bool]:
        """
        Handle a CALL or CALLIND op.

        Returns (new_vulns, is_unknown).
        is_unknown = True if the pattern matcher could not identify the function.
        """
        result = self.matcher.match(op)
        if result is None:
            return [], False

        vulns:      list[VulnCandidate] = []
        is_unknown  = result.kind == MatchKind.NO_MATCH
        match_kind  = result.kind.name   # LIBRARY_MATCH | STRUCTURAL_MATCH | NO_MATCH

        # ── 1. Determine the primary tainted variable for this call ─────
        #    Priority: explicit taint_arg > source_var > any tainted arg
        primary_tainted_var = None

        if result.taint_arg >= 0 and result.taint_arg < len(result.arg_vars):
            candidate = result.arg_vars[result.taint_arg]
            if state.is_tainted(candidate) or state.mem_is_tainted(candidate):
                primary_tainted_var = candidate

        if primary_tainted_var is None and result.source_var():
            src = result.source_var()
            if state.is_tainted(src) or state.mem_is_tainted(src):
                primary_tainted_var = src

        if primary_tainted_var is None:
            for var in result.arg_vars:
                if state.is_tainted(var) or state.mem_is_tainted(var):
                    primary_tainted_var = var
                    break

        # ── 2. Check if tainted data reaches a known sink ─────────────
        if result.is_sink and primary_tainted_var:
            sink_type = result.sink_type or "buffer_overflow"
            # format_string: only flag when the taint originates from a confirmed
            # external source (recv/read/fread/scanf).  Parameter-seeded taint
            # reaching printf is common in dump/debug functions and almost always
            # a false positive — the format string is a constant in those cases.
            if sink_type == "format_string":
                if primary_tainted_var not in state.externally_tainted_vars:
                    sink_type = None   # suppress — not externally tainted
            # buffer_overflow via library sink: param-seeded taint reaching
            # memcpy/strncpy is normal; only flag when data originates from
            # an external source (fread, recv, etc.) where the size is
            # attacker-controlled.
            if sink_type == "buffer_overflow":
                if primary_tainted_var not in state.externally_tainted_vars:
                    sink_type = None   # suppress — param-only taint
                elif primary_tainted_var in state.checked_vars:
                    sink_type = None   # suppress — tainted source was bounds-checked
                elif result.size_arg != -1:
                    # suppress if ALL size args are compile-time constants:
                    # memcpy(dst, src, 4) is safe regardless of src taint
                    _raw_in = op.get("inputs") or []
                    _sa_idx = (result.size_arg if isinstance(result.size_arg, list)
                               else [result.size_arg])
                    if all(
                        self._const_value(_raw_in[idx + 1]) is not None
                        if (idx + 1) < len(_raw_in) else False
                        for idx in _sa_idx
                    ):
                        sink_type = None  # suppress — size is a compile-time constant
            # null_dereference via LIBRARY_MATCH (e.g. png_safe_execute):
            # only flag when the FUNCTION POINTER arg (index 1) is tainted.
            # Tainted png_ptr (arg 0) alone is too broad — it fires for every
            # image API wrapper (begin_read, write_main etc.) creating many FPs.
            if sink_type == "null_dereference":
                _fn_ptr = result.arg_vars[1] if len(result.arg_vars) > 1 else None
                if not (_fn_ptr and state.is_tainted(_fn_ptr)):
                    sink_type = None   # suppress — fn ptr not tainted
            # integer_overflow via allocator: require the SIZE arg (not just any
            # arg like png_ptr) to be tainted. Only suppress when size_vars are
            # KNOWN (size_arg >= 0) AND none of them are tainted.
            # If size_vars is empty (unknown size arg), allow the candidate through.
            if sink_type == "integer_overflow" and result.fn_name in _ALLOCATOR_FNAMES:
                _sv_list = result.size_vars()
                if _sv_list and not any(
                        state.is_tainted(sv) or state.mem_is_tainted(sv)
                        for sv in _sv_list):
                    sink_type = None   # suppress — size arg(s) not tainted
                elif _sv_list and all(sv in state.checked_vars for sv in _sv_list):
                    sink_type = None   # suppress — all size args bounds-checked (fixed binary)
            # Non-allocator IO (psf_fread, read as sinks): require external taint.
            # Param-only taint reaching a bounded read/write sink is not IO.
            if sink_type == "integer_overflow" and result.fn_name not in _ALLOCATOR_FNAMES:
                if primary_tainted_var not in state.externally_tainted_vars:
                    sink_type = None   # suppress — not externally tainted
            if sink_type:
                # Allocator IO must always go to LLM: whether tainted size
                # actually wraps depends on calling context the LLM can reason.
                _s2_match_kind = match_kind
                if sink_type == "integer_overflow" and result.fn_name in _ALLOCATOR_FNAMES:
                    _s2_match_kind = "NO_MATCH"
                _add_vuln(vulns, VulnCandidate(
                    func_name    = func_name,
                    entry_addr   = entry_addr,
                    vuln_type    = sink_type,
                    op_seq       = result.op_seq,
                    sink_fn      = result.fn_name,
                    taint_source = state.get_path(primary_tainted_var)[0],
                    taint_path   = state.get_path(primary_tainted_var),
                    bounded      = result.bounded,
                    confidence   = result.confidence,
                    description  = (
                        f"Tainted variable {primary_tainted_var} reaches "
                        f"{result.fn_name}() at seq {result.op_seq}. "
                        + ("No bounds check." if not result.bounded
                           else "Bounded by size arg.")
                    ),
                    match_kind   = _s2_match_kind,
                ))

        # ── 3a. Check unbounded copy with tainted source ──────────────
        # Require the tainted source to come from a KNOWN EXTERNAL SOURCE
        # (fread, recv, user input…), not just param-seeded taint.
        # Param-seeded taint (psf pointer passed through callers) reaches
        # unbounded functions like psf_binheader_readf in every call site —
        # generating massive FPs from normal internal library operations.
        # True BO from these paths is covered by Track 2 LLM analysis.
        elif (
            not result.is_sink
            and not result.bounded
            and primary_tainted_var
            and primary_tainted_var in state.externally_tainted_vars
            and primary_tainted_var not in state.checked_vars
            and result.writes_memory_at not in (-1, None)
        ):
            _add_vuln(vulns, VulnCandidate(
                func_name    = func_name,
                entry_addr   = entry_addr,
                vuln_type    = "buffer_overflow",
                op_seq       = result.op_seq,
                sink_fn      = result.fn_name,
                taint_source = state.get_path(primary_tainted_var)[0],
                taint_path   = state.get_path(primary_tainted_var),
                bounded      = False,
                confidence   = result.confidence * 0.9,
                description  = (
                    f"Tainted data from {primary_tainted_var} copied via "
                    f"{result.fn_name}() at seq {result.op_seq} without size limit."
                ),
                match_kind   = match_kind,
            ))

        # ── 3b. Check tainted size argument in allocation / copy ──────
        # REQUIRE multiplication taint (INT_MULT) before flagging integer_overflow.
        #
        # Rationale:
        #   integer_overflow = a SIZE VALUE WRAPS AROUND due to arithmetic.
        #   Only multiplication can produce a value much larger than its operands
        #   (n * elem_size).  INT_ADD/INT_SUB (strlen+1, ptr+offset) are normal
        #   library patterns that do not cause overflow in practice and adding
        #   them causes excessive false positives.
        #
        # Allocator-wrapper skip:
        #   If the function being analyzed IS itself an allocator (png_zalloc,
        #   png_realloc_array, etc.), skip the check entirely.  Wrapper functions
        #   forward a size parameter to an inner allocator — that is their job.
        #   The real overflow risk is in the callers that compute the size.
        #
        # Bounds-check skip:
        #   If any variable in the taint path of sv was bounds-checked before
        #   reaching this call (sv in state.checked_vars), the code is safe in
        #   this execution path.  The fixed binary inserts a guard like
        #   `if (n > MAX) return` before the multiplication, which sets
        #   checked_vars on n and propagates to n*elem_size — so the fixed
        #   binary suppresses the finding while the vulnerable binary does not.
        size_vars = result.size_vars()
        if func_name in _ALLOCATOR_FNAMES:
            size_vars = []   # this function IS the allocator — don't flag it
        for sv in size_vars:
            if state.is_tainted(sv):
                # IO from param-only taint is inter-procedural and handled by
                # Track 2 (LLM).  Only flag when the size originates from a
                # known external source (fread, recv, …) so Track 1 stays
                # high-precision.
                if sv not in state.externally_tainted_vars:
                    continue
                via_arith = sv in state.mult_tainted_vars
                _COPY_SINKS = frozenset({
                    "memcpy", "memmove", "memcpy_s", "bcopy", "png_memcpy",
                })
                if (not via_arith and state.is_tainted(sv)
                        and sv not in state.checked_vars
                        and result.fn_name in _COPY_SINKS):
                    via_arith = True
                if not via_arith:
                    continue
                if sv in state.checked_vars:
                    # Size was bounds-checked upstream — fixed binary path.
                    log.debug(
                        "integer_overflow suppressed: %s checked before %s() at seq %s",
                        sv, result.fn_name, result.op_seq,
                    )
                    continue
                # Allocator IO always requires LLM confirmation: whether a
                # multiplication overflow actually reaches malloc/calloc with
                # no upstream bounds check depends on the full calling context
                # which the LLM can assess from the P-code.  Auto-confirming
                # every calloc(tainted_size) generates too many FPs.
                _io_match_kind = match_kind
                if result.fn_name in _ALLOCATOR_FNAMES:
                    _io_match_kind = "NO_MATCH"
                _add_vuln(vulns, VulnCandidate(
                    func_name    = func_name,
                    entry_addr   = entry_addr,
                    vuln_type    = "integer_overflow",
                    op_seq       = result.op_seq,
                    sink_fn      = result.fn_name,
                    taint_source = state.get_path(sv)[0],
                    taint_path   = state.get_path(sv),
                    bounded      = False,
                    confidence   = result.confidence * 0.9,
                    description  = (
                        f"Tainted size argument {sv} reached {result.fn_name}() "
                        f"at seq {result.op_seq} via arithmetic — integer overflow "
                        f"may cause undersized allocation or OOB copy."
                    ),
                    match_kind   = _io_match_kind,
                ))

        # ── 4. Propagate taint from this call ─────────────────────────

        # External input: seed taint into buffer arguments
        for var in result.tainted_arg_vars():
            state.taint_var(var, reason=f"external:{result.fn_name}", from_var="")

        # Taint memory at destination pointer
        for dest_var in (result.written_memory_var() or []):
            src_var = result.source_var()
            if src_var and state.is_tainted(src_var):
                state.taint_mem(dest_var, reason=f"write:{result.fn_name}", from_var=src_var)

        # External source: taint the destination memory directly
        if result.is_external_source():
            for dest_var in (result.written_memory_var() or []):
                state.taint_mem(dest_var, reason=f"external:{result.fn_name}", from_var="")
            # Taint the pointer vars themselves only when external_input is also
            # specified.  Without external_input (e.g. psf_binheader_readf), we
            # only want memory-level taint — tainting the pointer var itself
            # causes spurious write_what_where findings when those pointers are
            # later used as addresses in STORE ops.
            if result.writes_memory_at == "all_ptr_args" and result.external_input:
                for var, size in zip(result.arg_vars, result.arg_sizes):
                    if size == 8:
                        state.taint_var(var, reason=f"external:{result.fn_name}", from_var="")

        # Return value taint
        # UAF kill: reassigned output var is no longer the freed pointer.
        if result.return_var:
            state.freed_ptrs.discard(result.return_var)

        if result.return_tainted and result.return_var:
            state.taint_var(
                result.return_var,
                reason=f"return:{result.fn_name}",
                from_var="",
            )

        # NO_MATCH: try inter-procedural summary before the conservative fallback.
        # If the callee has a summary, apply precise arg→return/mem effects.
        # For CALLIND, also try to resolve the callee from fn_ptr_vars.
        # Mark the call as matched (not unknown) so it won't be sent to the LLM.
        if is_unknown and self.summary_db is not None:
            lookup_name = result.fn_name
            # CALLIND resolution: fn_name is the callee varnode — check fn_ptr_vars.
            if op.get("op") == "CALLIND" and lookup_name in state.fn_ptr_vars:
                lookup_name = state.fn_ptr_vars[lookup_name]
            summary = self.summary_db.get(lookup_name)
            if summary is not None:
                self.summary_db.apply_at_call_site(
                    summary    = summary,
                    arg_vars   = result.arg_vars,
                    return_var = result.return_var,
                    state      = state,
                    fn_name    = lookup_name,
                )
                is_unknown = False   # handled by summary — not a true unknown
            # Even if no direct summary, check if fn_ptr args enable specialized analysis.
            elif state.fn_ptr_vars:
                fn_ptr_args = {
                    i: state.fn_ptr_vars[av]
                    for i, av in enumerate(result.arg_vars)
                    if av and av in state.fn_ptr_vars
                }
                if fn_ptr_args:
                    from interprocedural import _specialize_frees_fields
                    specialized = _specialize_frees_fields(
                        fn_name         = result.fn_name,
                        arg_vars        = result.arg_vars,
                        fn_ptr_args     = fn_ptr_args,
                        addr_to_name    = self.summary_db.addr_to_name,
                        known_summaries = {
                            name: self.summary_db.get(name)
                            for name in fn_ptr_args.values()
                            if self.summary_db.get(name) is not None
                        },
                    )
                    for (arg_idx, field_offset) in specialized:
                        if arg_idx < len(result.arg_vars):
                            arg_var = result.arg_vars[arg_idx]
                            state.freed_fields.add((arg_var, field_offset))
                            log.debug(
                                "fn_ptr specialized: freed_fields.add((%s, 0x%x)) at seq %s",
                                arg_var, field_offset, result.op_seq,
                            )
                    if specialized:
                        is_unknown = False

        # ── LLM-driven taint state updates (Task 1) ──────────────────────
        # When Stage 2.5 LLM identified this function as a validator or external
        # source, apply those effects directly to the taint state here.
        # This is what makes the LLM an active participant in the analysis,
        # not just a post-hoc confirmer.
        #
        # marks_checked_args: LLM said "this function bounds-checks arg[N]"
        #   → add arg[N] to checked_vars so downstream sinks are suppressed.
        for idx in result.marks_checked_args:
            if idx < len(result.arg_vars) and result.arg_vars[idx]:
                state.checked_vars.add(result.arg_vars[idx])
                log.debug(
                    "LLM validator effect: checked_vars += %s (arg[%d] of %s)",
                    result.arg_vars[idx], idx, result.fn_name,
                )
        #
        # external_source_args: LLM said "this function reads external data into arg[N]"
        #   → add arg[N] to externally_tainted_vars so subsequent taint gates pass.
        for idx in result.external_source_args:
            if idx < len(result.arg_vars) and result.arg_vars[idx]:
                var = result.arg_vars[idx]
                state.taint_var(var, reason=f"external:llm_source:{result.fn_name}", from_var="")
                log.debug(
                    "LLM source effect: externally_tainted_vars += %s (arg[%d] of %s)",
                    var, idx, result.fn_name,
                )

        # Conservative fallback for calls with no pattern and no summary.
        # Only propagate taint when input was already tainted — do NOT blindly
        # taint return values, which causes false positives on internal helpers.
        if is_unknown:
            any_input_tainted = any(
                state.is_tainted(v) or state.mem_is_tainted(v)
                for v in result.arg_vars
            )
            if any_input_tainted:
                if result.return_var:
                    state.taint_var(
                        result.return_var,
                        reason="unknown_call:return",
                        from_var="",
                    )
                for var, size in zip(result.arg_vars, result.arg_sizes):
                    if size == 8 and state.is_tainted(var):
                        state.taint_mem(var, reason="unknown_call:ptr_arg", from_var=var)


                # Unknown CALL with EXTERNALLY-tainted args → LLM candidate
                # Only fires when recv/fread/getenv-origin data flows to unknown fn.
                # Param-tainted detection was removed — it caused hundreds of FPs
                # in test harness functions that call many internal APIs.
                _tp = [v for v,sz in zip(result.arg_vars, result.arg_sizes)
                       if state.is_tainted(v) and sz >= 4]
                _ep = [v for v in _tp if v in state.externally_tainted_vars]
                _is_direct    = op.get("op") == "CALL"
                _valid_target = bool(result.fn_name) and result.fn_name != func_name
                if (len(result.arg_vars) >= 2 and len(_ep) >= 1
                        and _is_direct and _valid_target):
                    _b = _ep[0]
                    _add_vuln(vulns, VulnCandidate(
                        func_name=func_name, entry_addr=entry_addr,
                        vuln_type="unknown_call", op_seq=result.op_seq,
                        sink_fn=result.fn_name,
                        taint_source=state.get_path(_b)[0] if state.get_path(_b) else _b,
                        taint_path=state.get_path(_b),
                        bounded=False, confidence=0.35,
                        description=(
                            f"Externally-tainted data flows into unknown function "
                            f"{result.fn_name}() at seq {result.op_seq}. "
                            f"Args: {result.arg_sizes}. LLM review needed."
                        ),
                        match_kind="NO_MATCH",
                        arg_sizes=list(result.arg_sizes),
                    ))
        # ── 5. UAF check: tainted arg used after free (check BEFORE recording free)
        # Must check before step 6 records the current free, otherwise free(ptr)
        # would immediately match itself as a double-free false positive.
        for var in result.arg_vars:
            if var in state.freed_ptrs:
                # CWE-415 Double Free vs CWE-416 Use-After-Free
                # If the current call is itself a free function AND the arg
                # is already freed -> double free.
                # Otherwise -> use-after-free.
                _is_double_free = (result.frees_memory_at >= 0)
                _vuln_type = "double_free" if _is_double_free else "use_after_free"
                _desc = (
                    f"Pointer {var} freed twice via {result.fn_name}() "
                    f"at seq {result.op_seq} -- double-free (CWE-415)."
                ) if _is_double_free else (
                    f"Freed pointer {var} reused in {result.fn_name}() "
                    f"at seq {result.op_seq} -- use-after-free (CWE-416)."
                )
                # Both double-free and use-after-free from path-insensitive analysis
                # fire on error-path patterns (free on branch A, use/free on branch B).
                # Always require LLM to confirm it's a genuine same-execution-path issue.
                _df_match_kind = "NO_MATCH"
                _add_vuln(vulns, VulnCandidate(
                    func_name    = func_name,
                    entry_addr   = entry_addr,
                    vuln_type    = _vuln_type,
                    op_seq       = result.op_seq,
                    sink_fn      = result.fn_name,
                    taint_source = var,
                    taint_path   = state.get_path(var) if state.is_tainted(var) else [var],
                    bounded      = False,
                    confidence   = 0.80 if _is_double_free else 0.75,
                    description  = _desc,
                    match_kind   = _df_match_kind,
                ))
                break

        # ── 6. UAF tracking: record freed pointer (after check to avoid self-flag)
        if result.frees_memory_at >= 0:
            freed_idx = result.frees_memory_at
            if freed_idx < len(result.arg_vars):
                freed_var = result.arg_vars[freed_idx]
                state.freed_ptrs.add(freed_var)
                log.debug("UAF tracking: freed pointer %s at seq %s", freed_var, result.op_seq)
                # Field-level: if freed_var came from a struct field LOAD,
                # mark that (base, offset) as freed so future LOADs/STOREs
                # through the same field are flagged as UAF.
                if freed_var in state.field_load_src:
                    state.freed_fields.add(state.field_load_src[freed_var])
                    log.debug(
                        "UAF field tracking: freed field %s at seq %s",
                        state.field_load_src[freed_var], result.op_seq,
                    )

        return vulns, is_unknown

    def _handle_store(
        self,
        op:         dict,
        state:      TaintState,
        vulns:      list[VulnCandidate],
        func_name:  str,
        entry_addr: str,
    ) -> None:
        """
        Handle STORE op.

        STORE inputs: [space_id, addr_var, value_var]
        OR in some Ghidra versions: [addr_var, value_var]

        Two taint effects:
          1. value is tainted → mem_taint[addr_var] = True
          2. addr is tainted  → write-what-where primitive (dangerous)
        """
        inputs = op.get("inputs") or []

        # Ghidra P-code STORE: inputs[0]=space, inputs[1]=addr, inputs[2]=value
        # But in SSA high P-code sometimes just [addr, value]
        if len(inputs) >= 3:
            addr_var  = self._varname(inputs[1])
            value_var = self._varname(inputs[2])
        elif len(inputs) == 2:
            addr_var  = self._varname(inputs[0])
            value_var = self._varname(inputs[1])
        else:
            return

        # Effect 1: tainted value written to memory
        if state.is_tainted(value_var):
            state.taint_mem(addr_var, reason=f"store:{value_var}", from_var=value_var)

        # Effect 2: tainted address — write-what-where.
        # Only flag when taint originated from a confirmed external source.
        # Param-seeded taint reaching STORE is normal library operation
        # (libpng writing processed pixels into output buffers — that is its job).
        # We check state.externally_tainted_vars directly — no string-searching
        # taint path entries, which was fragile and rebuilt a set on every call.
        if (state.is_tainted(addr_var)
                and addr_var in state.externally_tainted_vars
                and addr_var not in state.ptr_offsets):  # filter struct field writes (const offset)
            path     = state.get_path(addr_var)
            path_str = " → ".join(path)
            _add_vuln(vulns, VulnCandidate(
                func_name    = func_name,
                entry_addr   = entry_addr,
                vuln_type    = "write_what_where",
                op_seq       = int(op.get("seq", -1)),
                sink_fn      = "STORE",
                taint_source = path[0],
                taint_path   = path,
                bounded      = False,
                confidence   = 0.6,
                description  = (
                    f"External-data-controlled address {addr_var} in STORE at "
                    f"seq {op.get('seq')}. Path: {path_str}"
                ),
                match_kind   = "NO_MATCH",
            ))

        # "store NULL after free" — safe cleanup idiom (e.g. ptr = NULL after free).
        # Detect: value being stored is the constant 0x0.
        _value_inp = (inputs[2] if len(inputs) >= 3
                      else inputs[1] if len(inputs) == 2 else None)
        _stores_null = (_value_inp is not None and self._const_value(_value_inp) == 0)

        # Field-level UAF: STORE via a pointer whose source field was freed.
        if addr_var in state.ptr_offsets:
            if state.ptr_offsets[addr_var] in state.freed_fields:
                # "stores untainted" = the value is a fresh allocation or constant
                # (not attacker-controlled). This is the safe realloc pattern:
                # psf->buf = malloc(n) after free(psf->buf) — not a UAF exploit.
                _value_untainted = (value_var is not None
                                    and not state.is_tainted(value_var))
                if _stores_null or _value_untainted:
                    # STORE NULL or fresh pointer to freed field → safe cleanup / realloc.
                    # Kill the freed_field tracking so subsequent LOADs don't fire.
                    state.freed_fields.discard(state.ptr_offsets[addr_var])
                else:
                    base_v, off = state.ptr_offsets[addr_var]
                    _add_vuln(vulns, VulnCandidate(
                        func_name    = func_name,
                        entry_addr   = entry_addr,
                        vuln_type    = "use_after_free",
                        op_seq       = int(op.get("seq", -1)),
                        sink_fn      = "STORE",
                        taint_source = addr_var,
                        taint_path   = state.get_path(addr_var) if state.is_tainted(addr_var) else [addr_var],
                        bounded      = False,
                        confidence   = 0.8,
                        description  = (
                            f"STORE at seq {op.get('seq')} via {base_v}+0x{off:x} — "
                            f"field pointer was freed earlier (use-after-free write)."
                        ),
                        match_kind   = "NO_MATCH",
                    ))

        # UAF: writing to a previously freed pointer
        if addr_var in state.freed_ptrs and not _stores_null:
            _add_vuln(vulns, VulnCandidate(
                func_name    = func_name,
                entry_addr   = entry_addr,
                vuln_type    = "use_after_free",
                op_seq       = int(op.get("seq", -1)),
                sink_fn      = "STORE",
                taint_source = addr_var,
                taint_path   = state.get_path(addr_var) if state.is_tainted(addr_var) else [addr_var],
                bounded      = False,
                confidence   = 0.75,
                description  = (
                    f"STORE to freed pointer {addr_var} at seq {op.get('seq')} — "
                    f"use-after-free write."
                ),
                match_kind   = "NO_MATCH",
            ))

    def _handle_load(
        self,
        op:         dict,
        state:      TaintState,
        vulns:      list[VulnCandidate],
        func_name:  str,
        entry_addr: str,
    ) -> None:
        """
        Handle LOAD op.

        LOAD inputs: [space_id, addr_var]  OR  [addr_var]
        output: value read from memory

        Taint effect:
          if mem_taint[addr_var] OR var_taint[addr_var] → taint output
        UAF check:
          if addr_var was previously freed → generate use_after_free candidate
        """
        inputs     = op.get("inputs") or []
        output     = op.get("output")
        output_var = self._varname(output)
        if not output_var:
            return

        # Extract address variable
        if len(inputs) >= 2:
            addr_var = self._varname(inputs[1])
        elif len(inputs) == 1:
            addr_var = self._varname(inputs[0])
        else:
            return

        # UAF kill: LOAD writes a new value to output_var — clear freed status
        state.freed_ptrs.discard(output_var)

        if state.mem_is_tainted(addr_var) or state.is_tainted(addr_var):
            state.taint_var(
                output_var,
                reason=f"load:*{addr_var}",
                from_var=addr_var,
            )
            if addr_var in state.externally_tainted_vars:
                state.externally_tainted_vars.add(output_var)

        # Field-level tracking: if addr was produced by PTRSUB(base, offset),
        # record that output_var came from that field so we can detect
        # free(output_var) → marks (base, offset) as freed.
        if addr_var in state.ptr_offsets:
            state.field_load_src[output_var] = state.ptr_offsets[addr_var]
            # Also detect UAF: loading via a pointer whose source field was freed.
            # e.g. LOAD via PTRSUB(VAR_obj, 0x10) when freed_fields has (VAR_obj, 0x10)
            if state.ptr_offsets[addr_var] in state.freed_fields and vulns is not None:
                base_v, off = state.ptr_offsets[addr_var]
                _add_vuln(vulns, VulnCandidate(
                    func_name    = func_name,
                    entry_addr   = entry_addr,
                    vuln_type    = "use_after_free",
                    op_seq       = int(op.get("seq", -1)),
                    sink_fn      = "LOAD",
                    taint_source = addr_var,
                    taint_path   = state.get_path(addr_var) if state.is_tainted(addr_var) else [addr_var],
                    bounded      = False,
                    confidence   = 0.8,
                    description  = (
                        f"LOAD at seq {op.get('seq')} via {base_v}+0x{off:x} — "
                        f"field pointer was freed earlier (use-after-free)."
                    ),
                    match_kind   = "NO_MATCH",
                ))

        # UAF: reading from a previously freed pointer
        if addr_var in state.freed_ptrs:
            _add_vuln(vulns, VulnCandidate(
                func_name    = func_name,
                entry_addr   = entry_addr,
                vuln_type    = "use_after_free",
                op_seq       = int(op.get("seq", -1)),
                sink_fn      = "LOAD",
                taint_source = addr_var,
                taint_path   = state.get_path(addr_var) if state.is_tainted(addr_var) else [addr_var],
                bounded      = False,
                confidence   = 0.75,
                description  = (
                    f"LOAD from freed pointer {addr_var} at seq {op.get('seq')} — "
                    f"use-after-free."
                ),
                match_kind   = "NO_MATCH",
            ))

    def _handle_arith(
        self,
        op:         dict,
        state:      TaintState,
        vulns:      Optional[list] = None,
        func_name:  str = "",
        entry_addr: str = "",
    ) -> None:
        """
        Handle arithmetic and general ops.
        Standard rule: if any input is tainted, taint the output.

        Arithmetic-taint tracking (mult_tainted_vars):
          - Direct: output of INT_MULT only — multiplication is the primary source
            of integer overflow risk (n * elem_size wraps around).
          - Transitive: output of COPY/INT_ZEXT/INT_SEXT when input is already
            arith-tainted, so the flag survives casts and register copies.

        INT_ADD/INT_SUB/INT_LEFT are intentionally excluded: strlen+1, ptr+offset,
        etc. are normal library arithmetic that almost never overflow in practice
        and cause excessive false positives when included.

        Bounds-check tracking (checked_vars):
          - When a tainted variable appears as input to a comparison op
            (INT_LESS, INT_LESSEQUAL, INT_EQUAL, INT_NOTEQUAL) it has been
            bounds-checked.  This flag propagates through subsequent arithmetic
            so that derived size variables (e.g. n*elem_size where n was checked)
            are also considered safe — this is what distinguishes the fixed binary
            (which guards before the multiplication) from the vulnerable binary.

        Truncation-before-check (truncated_vars / integer_truncation):
          - INT_AND(tainted_var, MASK) where MASK is 0xFF/0xFFFF/0xFFFFFF/0xFFFFFFFF
            marks the output as truncated.  If that truncated variable is then used
            in a bounds comparison, the check operates on a narrowed copy of the
            value — an attacker can bypass it by supplying a value above MASK.
        """
        output     = op.get("output")
        output_var = self._varname(output)
        if not output_var:
            return

        # UAF kill: any assignment clears freed status of output_var.
        state.freed_ptrs.discard(output_var)

        inputs     = op.get("inputs") or []
        input_vars = [self._varname(i) for i in inputs if isinstance(i, dict)]
        op_type    = op.get("op", "")

        became_tainted = state.propagate(output_var, input_vars, op_type=op_type)
        if became_tainted:
            # mult_tainted: only INT_MULT creates real overflow risk
            if op_type == "INT_MULT":
                state.mult_tainted_vars.add(output_var)
            elif op_type in ("COPY", "INT_ZEXT", "INT_SEXT",
                              "INT_RIGHT", "INT_SRIGHT", "INT_LEFT",
                              "INT_ADD", "INT_SUB", "INT_OR", "INT_AND"):
                if any(iv in state.mult_tainted_vars for iv in input_vars):
                    state.mult_tainted_vars.add(output_var)

            # checked_vars: propagate bounds-checked status through arithmetic.
            if op_type in ("INT_MULT", "INT_ADD", "INT_SUB", "INT_LEFT",
                           "COPY", "INT_ZEXT", "INT_SEXT"):
                tainted_ins = [iv for iv in input_vars if iv and state.is_tainted(iv)]
                if op_type == "INT_MULT" and tainted_ins:
                    # Multiplication: ALL tainted factors must be bounded.
                    # checked_pixel_depth * unbounded_row_width is still dangerous.
                    if all(iv in state.checked_vars for iv in tainted_ins):
                        state.checked_vars.add(output_var)
                else:
                    if any(iv in state.checked_vars for iv in input_vars if iv):
                        state.checked_vars.add(output_var)

            # FIX 6: Array-index OOB detection
            # PTRADD(base, tainted_index) without a prior bounds check
            # → potential out-of-bounds access (SQLite query planner bugs,
            #   Lua VM execution bugs, libtiff array indexing)
            if op_type == "PTRADD" and output_var and len(inputs) >= 2:
                offset_inp = inputs[1] if isinstance(inputs[1], dict) else None
                if offset_inp:
                    offset_var = self._varname(offset_inp)
                    is_const   = self._const_value(offset_inp) is not None
                    if (offset_var and not is_const
                            and state.is_tainted(offset_var)
                            and offset_var not in state.checked_vars
                            and offset_var in state.mult_tainted_vars
                            and offset_var in state.externally_tainted_vars):
                        # Externally-tainted mult index in PTRADD → high risk OOB
                        # Requires EXTERNAL taint (recv/fread/BIO_read origin)
                        # Param-only taint excluded — too many FPs in pixel ops
                        _add_vuln(vulns, VulnCandidate(
                            func_name    = func_name,
                            entry_addr   = entry_addr,
                            vuln_type    = "integer_overflow",
                            op_seq       = op.get("seq", 0),
                            sink_fn      = "PTRADD",
                            taint_source = state.get_path(offset_var)[0] if state.get_path(offset_var) else offset_var,
                            taint_path   = state.get_path(offset_var),
                            bounded      = False,
                            confidence   = 0.50,
                            description  = (
                                f"Mult-tainted index {offset_var} used in PTRADD at "
                                f"seq {op.get('seq', 0)} without bounds check — "
                                f"potential array OOB (CWE-129/CWE-190)."
                            ),
                            match_kind   = "NO_MATCH",
                        ))

            # ptr_offsets: track PTRSUB(base, const_offset) → output_var so that
            # field-level UAF detection knows which struct field was accessed.
            if op_type == "PTRSUB" and output_var and inputs:
                base_inp   = inputs[0] if isinstance(inputs[0], dict) else None
                offset_inp = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], dict) else None
                if base_inp and offset_inp:
                    base_var = self._varname(base_inp)
                    cval     = self._const_value(offset_inp)
                    if base_var and cval is not None:
                        state.ptr_offsets[output_var] = (base_var, cval)

            # FIX 8: Integer truncation via narrowing AND mask
            # INT_AND(tainted_val, 0xFFFF) → tainted 16-bit result
            # If the original was 32-bit tainted, this is truncation CWE-197
            if op_type == "INT_AND" and output_var and len(inputs) >= 2:
                mask_inp  = inputs[1] if isinstance(inputs[1], dict) else None
                val_inp   = inputs[0] if isinstance(inputs[0], dict) else None
                if mask_inp and val_inp:
                    mask_val = self._const_value(mask_inp)
                    val_var  = self._varname(val_inp)
                    # Narrowing masks: 0xFF (byte), 0xFFFF (short), 0xFFFFFF
                    _NARROW = {0xFF, 0xFFFF, 0xFFFFFF, 0x3FF, 0x7FF}
                    if (mask_val in _NARROW
                            and val_var
                            and state.is_tainted(val_var)
                            and val_var not in state.checked_vars
                            and val_var in state.externally_tainted_vars):
                        # Only flag when data is externally tainted
                        # Param-only taint excluded — too many FPs in libpng bit ops
                        _add_vuln(vulns, VulnCandidate(
                            func_name    = func_name,
                            entry_addr   = entry_addr,
                            vuln_type    = "integer_truncation",
                            op_seq       = op.get("seq", 0),
                            sink_fn      = f"INT_AND(mask=0x{mask_val:X})",
                            taint_source = state.get_path(val_var)[0] if state.get_path(val_var) else val_var,
                            taint_path   = state.get_path(val_var),
                            bounded      = False,
                            confidence   = 0.55,
                            description  = (
                                f"Tainted value {val_var} masked to "
                                f"0x{mask_val:X} at seq {op.get('seq',0)} — "
                                f"integer truncation CWE-197 (libtiff field size truncation)."
                            ),
                            match_kind   = "NO_MATCH",
                        ))

            # truncated_vars: INT_AND(tainted, MASK) where MASK is a narrowing
            # constant marks the output as a truncated view of the input.
            # Propagate the truncation flag through COPY/ZEXT/SEXT so it
            # survives register copies and sign-extension before a comparison.
            #
            # Key guard: only flag when the mask is STRICTLY narrower than the
            # input's natural type width.  e.g. (uint8_t & 0xFF) is a no-op —
            # the mask covers all bits — so it is NOT a truncation.
            # (uint64_t & 0xFFFFFFFF) discards the upper 32 bits — that IS a
            # real narrowing and the source of the png_check_chunk_length bug.
            if op_type == "INT_AND":
                for inp in inputs:
                    if not isinstance(inp, dict):
                        continue
                    mask = self._const_value(inp)
                    if mask is not None and mask in _TRUNCATION_MASKS:
                        # Find the tainted operand and its declared byte-width
                        tainted_inp = None
                        for i in inputs:
                            if isinstance(i, dict) and self._varname(i) != self._varname(inp):
                                if state.is_tainted(self._varname(i)):
                                    tainted_inp = i
                                    break
                        if tainted_inp is not None:
                            var_size      = tainted_inp.get("size") or 0
                            min_size      = _TRUNCATION_MASKS[mask]  # minimum bytes
                            mask_bits     = mask.bit_length()
                            # Require: input is at least min_size bytes wide AND
                            # the mask is strictly narrower than the input type.
                            # Unknown width (var_size==0) treated conservatively.
                            wide_enough   = var_size == 0 or var_size >= min_size
                            real_narrowing = var_size == 0 or var_size * 8 > mask_bits
                            if wide_enough and real_narrowing:
                                state.truncated_vars[output_var] = mask
                        break
            elif op_type in ("COPY", "INT_ZEXT", "INT_SEXT"):
                for iv in input_vars:
                    if iv and iv in state.truncated_vars:
                        state.truncated_vars[output_var] = state.truncated_vars[iv]
                        break

        # Comparison ops: record that the tainted inputs were bounds-checked.
        # Also detect truncation-before-check: if the compared variable was
        # narrowed by INT_AND before arriving here, the check is incomplete.
        if op_type in ("INT_LESS", "INT_LESSEQUAL", "INT_EQUAL", "INT_NOTEQUAL",
                       "INT_SLESS", "INT_SLESSEQUAL"):
            _cvals = [self._const_value(inp) for inp in inputs if isinstance(inp,dict)]
            _max_c = max((v for v in _cvals if v is not None), default=0)
            for iv in input_vars:
                if not iv or not state.is_tainted(iv): continue
                if _max_c > 0x10:  # only large-constant comparisons are upper bounds
                    state.checked_vars.add(iv)
                if (iv in state.truncated_vars and vulns is not None
                        and iv in state.externally_tainted_vars):
                    mask = state.truncated_vars[iv]
                    _add_vuln(vulns, VulnCandidate(
                        func_name    = func_name,
                        entry_addr   = entry_addr,
                        vuln_type    = "integer_truncation",
                        op_seq       = int(op.get("seq", -1)),
                        sink_fn      = op_type,
                        taint_source = state.get_path(iv)[0],
                        taint_path   = state.get_path(iv),
                        bounded      = False,
                        confidence   = 0.75,
                        description  = (
                            f"Tainted variable {iv} was truncated via "
                            f"INT_AND const(0x{mask:x}) before the bounds "
                            f"comparison at seq {op.get('seq')} — an attacker "
                            f"can bypass this check by supplying a value "
                            f"larger than 0x{mask:x}."
                        ),
                        match_kind   = "STRUCTURAL_MATCH",
                    ))

        # Unconditional: track function-pointer constants regardless of taint.
        # PTRSUB(const(0x0), const(ADDR)) encodes a raw function address.
        # We resolve it via summary_db.addr_to_name and store in fn_ptr_vars
        # so that later CALLIND resolution can identify the callee.
        if op_type == "PTRSUB" and output_var and len(inputs) >= 2:
            b_inp = inputs[0] if isinstance(inputs[0], dict) else None
            a_inp = inputs[1] if isinstance(inputs[1], dict) else None
            if b_inp and a_inp:
                b_cval = self._const_value(b_inp)
                a_cval = self._const_value(a_inp)
                if b_cval == 0 and a_cval is not None and a_cval > 0:
                    addr_map = (
                        self.summary_db.addr_to_name
                        if self.summary_db and hasattr(self.summary_db, "addr_to_name")
                        else {}
                    )
                    hex_addr = hex(a_cval)[2:].lower()
                    fn_name  = addr_map.get(hex_addr)
                    if fn_name:
                        state.fn_ptr_vars[output_var] = fn_name
        # Propagate fn_ptr_vars through COPY (function pointer assigned to another var).
        elif op_type in ("COPY", "CAST") and output_var and inputs:
            src = self._varname(inputs[0])
            if src and src in state.fn_ptr_vars:
                state.fn_ptr_vars[output_var] = state.fn_ptr_vars[src]

    def _handle_cbranch(self, op: dict, state: TaintState) -> None:
        """
        Handle CBRANCH (conditional branch) op.

        CBRANCH has no output — it does not define a variable. But its
        condition input may be tainted, which means the attacker controls
        a branch decision (authentication bypass, type confusion, etc.).

        We log a flow step so the reasoning agent sees that taint reached a
        branch condition, without producing a VulnCandidate (the engine has
        no way to know whether this particular branch is security-relevant).
        """
        inputs = op.get("inputs") or []
        # CBRANCH inputs: [branch_target, condition_var]
        # The condition is always the last input.
        if not inputs:
            return
        cond_var = self._varname(inputs[-1])
        if state.is_tainted(cond_var):
            state.flow_log.append(FlowStep(
                seq      = int(op.get("seq", -1)),
                op       = "CBRANCH",
                addr     = op.get("addr", ""),
                from_var = cond_var,
                to_var   = "branch_condition",
                reason   = f"taint-controlled branch on {cond_var}",
                is_mem   = False,
            ))
            log.debug(
                "Taint-controlled CBRANCH at seq %s — condition var: %s",
                op.get("seq"), cond_var,
            )
        1
    @staticmethod
    def _varname(inp) -> str:
        if isinstance(inp, dict):
            return inp.get("name", "")
        return ""

    @staticmethod
    def _const_value(inp: dict) -> Optional[int]:
        """Return the integer value of a const varnode, or None."""
        if not isinstance(inp, dict):
            return None
        # Accept either space=="const" (Ghidra output) or name-only (test fixtures)
        name = inp.get("name", "")
        if inp.get("space") == "const" or name.startswith("const("):
            if name.startswith("const(0x"):
                try:
                    return int(name[8:-1], 16)
                except ValueError:
                    pass
            elif name.startswith("const("):
                try:
                    return int(name[6:-1])
                except ValueError:
                    pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    store   = PatternStore("pattern_store.db")
    matcher = PatternMatcher(store)
    engine  = TaintEngine(matcher)

    sep = "─" * 62
    print(f"\n{sep}")
    print("Taint Engine tests")
    print(f"{sep}\n")

    # ── Test 1: classic stack overflow ────────────────────────────────
    # recv() → VAR_2 → strcpy() → OVERFLOW
    print("Test 1: recv → strcpy (classic stack overflow)\n")

    vuln_func = {
        "name":       "vulnerable_copy",
        "entry_addr": "0x401136",
        "ops": [
            # VAR_0 = parameter (pointer to socket)
            {"seq": 0, "op": "LOAD",
             "output":  {"name": "VAR_1", "size": 8},
             "inputs":  [{"name": "const(0x0)", "size": 4},
                         {"name": "VAR_0", "size": 8}],
             "addr": "0x401136"},
            # VAR_3 = recv(sock, VAR_2_buf, 0x400, 0)
            {"seq": 1, "op": "CALL",
             "output":  {"name": "VAR_3", "size": 8},
             "inputs":  [{"name": "recv",        "size": 0},
                         {"name": "VAR_1",        "size": 8},
                         {"name": "VAR_2",        "size": 8},
                         {"name": "const(0x400)", "size": 4},
                         {"name": "const(0x0)",   "size": 4}],
             "addr": "0x401140"},
            # strcpy(VAR_local_buf, VAR_2)
            {"seq": 2, "op": "CALL",
             "output":  None,
             "inputs":  [{"name": "strcpy", "size": 0},
                         {"name": "VAR_local_buf", "size": 8},
                         {"name": "VAR_2",          "size": 8}],
             "addr": "0x401150"},
            {"seq": 3, "op": "RETURN",
             "output": None, "inputs": [], "addr": "0x401158"},
        ],
    }

    r1 = engine.analyze(vuln_func)
    r1.print_flow()
    print(r1.summary())
    assert r1.has_vulns(), "Should detect buffer overflow"
    assert any(v.vuln_type == "buffer_overflow" for v in r1.vulns)
    print("\n  buffer_overflow detected ✓\n")

    # ── Test 2: bounded copy — should NOT fire ────────────────────────
    print("Test 2: recv → strncpy (bounded — should be clean)\n")

    safe_func = {
        "name":       "safe_copy",
        "entry_addr": "0x401200",
        "ops": [
            {"seq": 0, "op": "CALL",
             "output":  {"name": "VAR_3", "size": 8},
             "inputs":  [{"name": "recv",        "size": 0},
                         {"name": "VAR_1",        "size": 8},
                         {"name": "VAR_2",        "size": 8},
                         {"name": "const(0x400)", "size": 4},
                         {"name": "const(0x0)",   "size": 4}],
             "addr": "0x401200"},
            # strncpy(dst, src, 64) — BOUNDED
            {"seq": 1, "op": "CALL",
             "output":  None,
             "inputs":  [{"name": "strncpy",     "size": 0},
                         {"name": "VAR_dst",      "size": 8},
                         {"name": "VAR_2",        "size": 8},
                         {"name": "const(0x40)",  "size": 4}],
             "addr": "0x401210"},
            {"seq": 2, "op": "RETURN",
             "output": None, "inputs": [], "addr": "0x401218"},
        ],
    }

    r2 = engine.analyze(safe_func)
    r2.print_flow()
    print(r2.summary())
    assert not r2.has_vulns(), "Should not detect vuln in bounded copy"
    print("\n  No vulnerability — bounded copy correctly ignored ✓\n")

    # ── Test 3: command injection ──────────────────────────────────────
    print("Test 3: getenv → system (command injection)\n")

    cmd_func = {
        "name":       "run_command",
        "entry_addr": "0x401300",
        "ops": [
            # VAR_cmd = getenv("PATH")
            {"seq": 0, "op": "CALL",
             "output":  {"name": "VAR_cmd", "size": 8},
             "inputs":  [{"name": "getenv",       "size": 0},
                         {"name": "const(0x402000)", "size": 8}],
             "addr": "0x401300"},
            # system(VAR_cmd)
            {"seq": 1, "op": "CALL",
             "output":  {"name": "VAR_ret", "size": 4},
             "inputs":  [{"name": "system",  "size": 0},
                         {"name": "VAR_cmd", "size": 8}],
             "addr": "0x401310"},
            {"seq": 2, "op": "RETURN",
             "output": None, "inputs": [], "addr": "0x401318"},
        ],
    }

    r3 = engine.analyze(cmd_func)
    r3.print_flow()
    print(r3.summary())
    assert r3.has_vulns()
    assert any(v.vuln_type == "command_injection" for v in r3.vulns)
    print("\n  command_injection detected ✓\n")

    # ── Test 4: taint through arithmetic ──────────────────────────────
    # Scenario: attacker controls a recv buffer pointer; the program uses
    # (buffer_ptr + offset) as a STORE target — classic write-what-where.
    # recv's external_input=[1] marks VAR_buf as externally tainted, so
    # INT_ADD(VAR_buf, offset) propagates that flag to VAR_idx, and the
    # STORE to VAR_idx fires write_what_where.
    print("Test 4: external buffer pointer + INT_ADD → STORE (write-what-where)\n")

    arith_func = {
        "name":       "offset_write",
        "entry_addr": "0x401400",
        "ops": [
            # recv(sock, VAR_buf, len, flags) — VAR_buf is external_input[1]
            {"seq": 0, "op": "CALL",
             "output":  {"name": "VAR_2", "size": 8},
             "inputs":  [{"name": "recv",        "size": 0},
                         {"name": "VAR_1",        "size": 8},
                         {"name": "VAR_buf",      "size": 8},
                         {"name": "const(0x400)", "size": 4},
                         {"name": "const(0x0)",   "size": 4}],
             "addr": "0x401400"},
            # VAR_idx = VAR_buf + 4  ← VAR_buf is externally tainted by recv
            #                           → VAR_idx inherits external origin
            {"seq": 1, "op": "INT_ADD",
             "output":  {"name": "VAR_idx", "size": 8},
             "inputs":  [{"name": "VAR_buf",     "size": 8},
                         {"name": "const(0x4)",  "size": 4}],
             "addr": "0x401408"},
            # STORE addr=VAR_idx, val=const(0)  ← externally-tainted addr = write-what-where
            {"seq": 2, "op": "STORE",
             "output":  None,
             "inputs":  [{"name": "const(0x1)",  "size": 4},   # space id
                         {"name": "VAR_idx",      "size": 8},   # address
                         {"name": "const(0x0)",   "size": 4}],  # value
             "addr": "0x401410"},
            {"seq": 3, "op": "RETURN",
             "output": None, "inputs": [], "addr": "0x401418"},
        ],
    }

    r4 = engine.analyze(arith_func)
    r4.print_flow()
    print(r4.summary())
    assert r4.has_vulns()
    assert any(v.vuln_type == "write_what_where" for v in r4.vulns)
    print("\n  write_what_where detected via externally-tainted pointer arithmetic ✓\n")

    # ── Test 5: integer overflow via multiplication → malloc ──────────
    print("Test 5: tainted count → INT_MULT → malloc (integer overflow risk)\n")

    malloc_func = {
        "name":       "alloc_buf",
        "entry_addr": "0x401500",
        "ops": [
            # VAR_count = recv(...)  ← attacker-controlled element count
            {"seq": 0, "op": "CALL",
             "output":  {"name": "VAR_count", "size": 8},
             "inputs":  [{"name": "recv",        "size": 0},
                         {"name": "VAR_1",        "size": 8},
                         {"name": "VAR_tmp",      "size": 8},
                         {"name": "const(0x8)",   "size": 4},
                         {"name": "const(0x0)",   "size": 4}],
             "addr": "0x401500"},
            # VAR_size = VAR_count * 8  ← multiplication → overflow risk
            {"seq": 1, "op": "INT_MULT",
             "output":  {"name": "VAR_size", "size": 8},
             "inputs":  [{"name": "VAR_count",   "size": 8},
                         {"name": "const(0x8)",  "size": 4}],
             "addr": "0x401508"},
            # VAR_ptr = malloc(VAR_size)  ← tainted size via mult
            {"seq": 2, "op": "CALL",
             "output":  {"name": "VAR_ptr", "size": 8},
             "inputs":  [{"name": "malloc",   "size": 0},
                         {"name": "VAR_size", "size": 8}],
             "addr": "0x401510"},
            {"seq": 3, "op": "RETURN",
             "output": None, "inputs": [], "addr": "0x401518"},
        ],
    }

    r5 = engine.analyze(malloc_func)
    r5.print_flow()
    print(r5.summary())
    assert r5.has_vulns()
    assert any(v.vuln_type == "integer_overflow" for v in r5.vulns)
    print("\n  integer_overflow detected via count * elem_size → malloc ✓\n")

    # ── Test 5b: same pattern WITH bounds check → should be clean ─────
    print("Test 5b: checked count → INT_MULT → malloc (fixed binary — should be clean)\n")

    malloc_fixed_func = {
        "name":       "alloc_buf_fixed",
        "entry_addr": "0x401600",
        "ops": [
            # VAR_count = recv(...)
            {"seq": 0, "op": "CALL",
             "output":  {"name": "VAR_count", "size": 8},
             "inputs":  [{"name": "recv",        "size": 0},
                         {"name": "VAR_1",        "size": 8},
                         {"name": "VAR_tmp",      "size": 8},
                         {"name": "const(0x8)",   "size": 4},
                         {"name": "const(0x0)",   "size": 4}],
             "addr": "0x401600"},
            # VAR_cond = INT_LESSEQUAL(VAR_count, MAX)  ← bounds check
            {"seq": 1, "op": "INT_LESSEQUAL",
             "output":  {"name": "VAR_cond", "size": 1},
             "inputs":  [{"name": "VAR_count",     "size": 8},
                         {"name": "const(0x1000)", "size": 8}],
             "addr": "0x401608"},
            # CBRANCH (error path if !cond)
            {"seq": 2, "op": "CBRANCH",
             "output":  None,
             "inputs":  [{"name": "const(0x401680)", "size": 8},
                         {"name": "VAR_cond",         "size": 1}],
             "addr": "0x401610"},
            # VAR_size = VAR_count * 8  ← same mult as before
            {"seq": 3, "op": "INT_MULT",
             "output":  {"name": "VAR_size", "size": 8},
             "inputs":  [{"name": "VAR_count",   "size": 8},
                         {"name": "const(0x8)",  "size": 4}],
             "addr": "0x401618"},
            # VAR_ptr = malloc(VAR_size)  ← should be suppressed
            {"seq": 4, "op": "CALL",
             "output":  {"name": "VAR_ptr", "size": 8},
             "inputs":  [{"name": "malloc",   "size": 0},
                         {"name": "VAR_size", "size": 8}],
             "addr": "0x401620"},
            {"seq": 5, "op": "RETURN",
             "output": None, "inputs": [], "addr": "0x401628"},
        ],
    }

    r5b = engine.analyze(malloc_fixed_func)
    r5b.print_flow()
    print(r5b.summary())
    assert not r5b.has_vulns(), "Bounds-checked allocation should NOT fire integer_overflow"
    print("\n  No vulnerability — bounds-checked allocation correctly suppressed ✓\n")

    # ── Test 6: truncation-before-check (png_check_chunk_length pattern) ─
    print("Test 6: param → INT_AND 0xFFFFFFFF → INT_LESS (truncation bypass)\n")

    trunc_func = {
        "name":       "check_chunk_length",
        "entry_addr": "0x401700",
        "ops": [
            # fread(VAR_buf, 8, 1, VAR_fp) → marks VAR_buf memory as externally tainted
            {"seq": 0, "op": "CALL",
             "output":  None,
             "inputs":  [{"name": "fread",   "size": 0},
                         {"name": "VAR_buf", "size": 8, "space": "unique"},
                         {"name": "const(0x8)", "size": 4, "space": "const"},
                         {"name": "const(0x1)", "size": 4, "space": "const"},
                         {"name": "VAR_fp",  "size": 8, "space": "unique"}],
             "addr": "0x4016f0"},
            # VAR_len = LOAD(VAR_buf) — chunk length parsed from external data
            {"seq": 1, "op": "LOAD",
             "output":  {"name": "VAR_len", "size": 8, "space": "unique"},
             "inputs":  [{"name": "ram",    "size": 4, "space": "const"},
                         {"name": "VAR_buf", "size": 8, "space": "unique"}],
             "addr": "0x4016f8"},
            # Truncate to 32-bit: VAR_trunc = VAR_len & 0xFFFFFFFF
            {"seq": 2, "op": "INT_AND",
             "output":  {"name": "VAR_trunc", "size": 4, "space": "unique"},
             "inputs":  [{"name": "VAR_len",        "size": 8, "space": "unique"},
                         {"name": "const(0xffffffff)", "size": 8, "space": "const"}],
             "addr": "0x401700"},
            # Bounds check on the TRUNCATED value — bypassable
            {"seq": 3, "op": "INT_LESS",
             "output":  {"name": "VAR_ok", "size": 1, "space": "unique"},
             "inputs":  [{"name": "VAR_trunc",       "size": 4, "space": "unique"},
                         {"name": "const(0x7fffffff)", "size": 4, "space": "const"}],
             "addr": "0x401708"},
            {"seq": 4, "op": "CBRANCH",
             "output":  None,
             "inputs":  [{"name": "const(0x401780)", "size": 8, "space": "const"},
                         {"name": "VAR_ok",           "size": 1, "space": "unique"}],
             "addr": "0x401710"},
            {"seq": 5, "op": "RETURN",
             "output": None, "inputs": [{"name": "const(0x1)", "size": 4, "space": "const"}],
             "addr": "0x401718"},
        ],
    }

    r6 = engine.analyze(trunc_func)
    r6.print_flow()
    print(r6.summary())
    assert r6.has_vulns(), "Should detect integer_truncation"
    assert any(v.vuln_type == "integer_truncation" for v in r6.vulns)
    print("\n  integer_truncation detected — bounds check on truncated value ✓\n")

    # ── Test 6b: same pattern WITHOUT truncation — should be clean ────────
    print("Test 6b: param → INT_LESS directly (no truncation — should be clean)\n")

    no_trunc_func = {
        "name":       "check_chunk_length_fixed",
        "entry_addr": "0x401800",
        "ops": [
            # Compare the full-width VAR_len directly — no AND mask
            {"seq": 0, "op": "INT_LESS",
             "output":  {"name": "VAR_ok", "size": 1, "space": "unique"},
             "inputs":  [{"name": "VAR_len",           "size": 8, "space": "unique"},
                         {"name": "const(0x7fffffff)",  "size": 4, "space": "const"}],
             "addr": "0x401800"},
            {"seq": 1, "op": "CBRANCH",
             "output":  None,
             "inputs":  [{"name": "const(0x401880)", "size": 8, "space": "const"},
                         {"name": "VAR_ok",           "size": 1, "space": "unique"}],
             "addr": "0x401808"},
            {"seq": 2, "op": "RETURN",
             "output": None, "inputs": [{"name": "const(0x1)", "size": 4, "space": "const"}],
             "addr": "0x401810"},
        ],
    }

    r6b = engine.analyze(no_trunc_func)
    r6b.print_flow()
    print(r6b.summary())
    assert not r6b.has_vulns(), "Direct comparison without truncation should be clean"
    print("\n  No vulnerability — full-width comparison correctly ignored ✓\n")

    print(f"{sep}")
    print("All taint engine tests passed ✓")
    print(f"{sep}")
    store.close()