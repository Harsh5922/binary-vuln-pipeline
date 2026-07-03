"""
pattern_matcher.py

Matches CALL operations in P-code against the pattern store.

What this does
--------------
Given a raw CALL op from the extractor:
  {
    "op": "CALL",
    "inputs": [
      {"name": "strcpy", "size": 0},   ← function pointer (input[0])
      {"name": "VAR_2",  "size": 8},   ← arg 0 = destination
      {"name": "VAR_0",  "size": 8},   ← arg 1 = source
    ],
    "output": null,
    "seq": 5,
    "addr": "0x401150"
  }

It extracts:
  fn_name   = "strcpy"
  arg_sizes = [8, 8]

Looks up in PatternStore → returns MatchResult with the data flow rule.

MatchResult tells the taint engine exactly:
  - which argument variables receive external data
  - which argument's pointed-to memory gets written
  - whether the return value is tainted
  - whether this is a dangerous sink
  - what type of vulnerability this represents

Three match outcomes
---------------------
  LIBRARY_MATCH   — exact hit in hardcoded library patterns (confident)
  STRUCTURAL_MATCH — hit in LLM-inferred structural patterns (less confident)
  NO_MATCH        — unknown, taint engine uses conservative defaults

Usage
-----
    from pattern_matcher import PatternMatcher, MatchResult
    from pattern_store   import PatternStore

    store   = PatternStore()
    matcher = PatternMatcher(store)

    for op in func["ops"]:
        if op["op"] in ("CALL", "CALLIND"):
            result = matcher.match(op)
            if result:
                print(result.fn_name, result.sink_type, result.writes_memory_at)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from pattern_store import PatternStore

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Match result
# ─────────────────────────────────────────────────────────────────────────────

class MatchKind(Enum):
    LIBRARY_MATCH    = auto()   # exact hit in hardcoded library patterns
    STRUCTURAL_MATCH = auto()   # hit in LLM-inferred structural patterns
    NO_MATCH         = auto()   # unknown function — conservative defaults apply


@dataclass
class MatchResult:
    """
    Data flow rule resolved for one CALL operation.

    All fields derived from the pattern store rule.
    The taint engine reads this to know how to propagate taint.
    """

    # Identity
    op_seq:       int            # sequence number of the CALL op
    fn_name:      str            # resolved function name
    arg_sizes:    list[int]      # sizes of each argument in bytes
    arg_vars:     list[str]      # canonical variable names of each argument
    return_var:   Optional[str]  # output varnode name (None if void)
    kind:         MatchKind      # LIBRARY_MATCH | STRUCTURAL_MATCH | NO_MATCH

    # Data flow — what the taint engine needs
    external_input:    list[int]   # arg indices that receive external data
    writes_memory_at:  object      # arg index whose memory gets written
                                   # int | "all_ptr_args" | -1
    reads_from:        int         # arg index used as source (-1 = none)
    return_tainted:    bool        # is the return value tainted?
    bounded:           bool        # is this copy size-bounded?
    size_arg:          object      # which arg holds the size limit
                                   # int | list[int] | -1
    frees_memory_at:   int         # arg index being freed (-1 = not a free)
    return_is_buffer:  bool        # return value is a heap buffer

    # Sink information
    is_sink:    bool               # is this a dangerous operation?
    sink_type:  str                # buffer_overflow | command_injection |
                                   # format_string | uaf | ""
    taint_arg:  int                # which arg being tainted makes it a sink

    # LLM-driven taint propagation (Task 1)
    # When the LLM identifies a function as a validator or external source,
    # these fields tell the taint engine to update state — not just confirm.
    marks_checked_args:   list[int]  # arg indices to add to checked_vars
    external_source_args: list[int]  # arg indices to add to externally_tainted_vars

    # Quality
    confidence: float              # 0.0 – 1.0
    vuln_score: float              # inherent danger level (1–10, from pattern store)
    notes:      str

    def is_external_source(self) -> bool:
        """Does this call bring external data into the program?"""
        return (
            bool(self.external_input)
            or self.writes_memory_at == "all_ptr_args"
        )

    def tainted_arg_vars(self) -> list[str]:
        """
        Which argument variables receive external data directly.
        Used to seed taint propagation.
        """
        result = []
        for idx in self.external_input:
            if idx < len(self.arg_vars):
                result.append(self.arg_vars[idx])
        return result

    def written_memory_var(self) -> Optional[str]:
        """
        Which argument variable points to memory that gets written.
        The taint engine marks *this_var as tainted in the memory map.
        """
        if self.writes_memory_at == "all_ptr_args":
            # All 8-byte (pointer) args receive data
            return [
                v for v, s in zip(self.arg_vars, self.arg_sizes)
                if s == 8
            ]
        if isinstance(self.writes_memory_at, int) and self.writes_memory_at >= 0:
            idx = self.writes_memory_at
            if idx < len(self.arg_vars):
                return [self.arg_vars[idx]]
        return []

    def source_var(self) -> Optional[str]:
        """Which argument variable is the data source (e.g. src in memcpy)."""
        if self.reads_from >= 0 and self.reads_from < len(self.arg_vars):
            return self.arg_vars[self.reads_from]
        return None

    def size_vars(self) -> list[str]:
        """Which argument variables hold the size limit."""
        if self.size_arg == -1:
            return []
        indices = self.size_arg if isinstance(self.size_arg, list) else [self.size_arg]
        return [
            self.arg_vars[i] for i in indices
            if i < len(self.arg_vars)
        ]

    def summary(self) -> str:
        """One-line human-readable summary for logging and reports."""
        parts = [f"{self.fn_name}({', '.join(str(s) for s in self.arg_sizes)})"]

        if self.is_external_source():
            parts.append("EXTERNAL_INPUT")
        if self.is_sink:
            parts.append(f"SINK({self.sink_type})")
        if not self.bounded and self.writes_memory_at != -1:
            parts.append("UNBOUNDED")
        elif self.bounded:
            parts.append("bounded")

        parts.append(f"conf={self.confidence:.0%}")
        parts.append(f"[{self.kind.name}]")

        return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Matcher
# ─────────────────────────────────────────────────────────────────────────────

class PatternMatcher:
    """
    Matches CALL and CALLIND ops against the pattern store.

    For each match it returns a MatchResult describing exactly
    how data flows through that call — what the taint engine needs.
    """

    def __init__(self, store: PatternStore):
        self.store = store

    # ── Public API ────────────────────────────────────────────────────

    def match(self, op: dict) -> Optional[MatchResult]:
        """
        Match one P-code op against the pattern store.

        Resolution order:
          1. Name known → exact DB lookup
          2. Name is raw address (ram(0xADDR)) → Groq identifies → store → re-lookup
          3. Still nothing → NO_MATCH conservative result

        Parameters
        ----------
        op : a single op dict from pcode.jsonl
             must have: op, inputs, output, seq

        Returns MatchResult if the op is a CALL/CALLIND, None otherwise.
        """
        op_type = op.get("op", "")
        if op_type not in ("CALL", "CALLIND"):
            return None

        inputs     = op.get("inputs") or []
        output     = op.get("output")
        seq        = int(op.get("seq", -1))

        # inputs[0] is always the function pointer / target
        # inputs[1:] are the actual arguments
        if not inputs:
            return self._no_match("", [], [], output, seq)

        fn_input   = inputs[0]
        arg_inputs = inputs[1:]

        fn_name    = self._extract_fn_name(fn_input, op_type)
        arg_sizes  = [self._extract_size(i) for i in arg_inputs]
        arg_vars   = [self._extract_varname(i) for i in arg_inputs]

        # ① Try pattern store by name (known functions)
        rule = self.store.lookup(fn_name, arg_sizes)

        # Fingerprint matching only for unresolved static addresses (0x...).
        # CALLIND through a variable (INDIRECT:VAR_N) targets an unknown runtime
        # function — fingerprint matching against arg shapes gives too many false
        # positives (e.g. any [8,8] indirect call matched to popen).
        if rule is None and self._is_static_unresolved(fn_name):

            # ② Pattern fingerprint lookup — arg sizes + return shape
            #    Stable across all binaries — no address dependency
            rule = self.store.lookup_by_pattern(arg_sizes, output)
            if rule:
                fn_name = rule.pop("fn_name", fn_name)  # use resolved name
                # Always STRUCTURAL_MATCH — we inferred the name from shape,
                # not confirmed it directly. Confidence already penalised in lookup.
                rule["source"] = "pattern_inferred"
                log.debug("Pattern fingerprint match: %s → %s", fn_name, fn_name)

            # ③ Groq — only when pattern also misses.
            # Build call-site context hints to improve identification accuracy.
            if rule is None:
                ctx: dict = {}
                # Constant arguments narrow the candidate set — e.g. a const
                # 4th arg of 0 strongly suggests recv(..., 0) or send(..., 0).
                ctx["const_args"] = [
                    (i, inp.get("name", ""))
                    for i, inp in enumerate(arg_inputs)
                    if isinstance(inp, dict) and (inp.get("name") or "").startswith("const(")
                ]
                resolved = self._groq_resolve(
                    fn_name, arg_sizes, arg_vars,
                    output=output, context=ctx,
                )
                if resolved:
                    log.debug("Groq resolved %s → %s", fn_name, resolved)
                    lib_rule = self.store.lookup(resolved, arg_sizes)
                    if lib_rule:
                        # Store by PATTERN key — reusable across binaries
                        self.store.store_structural(
                            fn_name    = resolved,
                            arg_sizes  = arg_sizes,
                            rule       = dict(lib_rule),
                            confidence = 0.75,
                            notes      = f"Groq identified from address {fn_name}",
                            output     = output,
                        )
                        rule    = lib_rule
                        fn_name = resolved

        # ④ Learned patterns (Phase 4) — semantic summaries converted to taint rules.
        #    Stage 2.5 learns what unknown functions do (read_input, copy, allocator…)
        #    and stores taint rules. This enables propagation through previously-unknown
        #    library functions like psf_binheader_readf, png_read_row, lua_rawget etc.
        #    Cross-binary: rule learned in PNG001 propagates taint in PNG002-007.
        if rule is None:
            learned = self.store.get_learned_rule(fn_name, arg_sizes)
            if learned:
                role = learned.get("role", "other")
                # Only use learned rules that actually affect taint propagation.
                # Validator/logger/other roles have sink=False and vuln_score=0 —
                # using them as LEARNED_MATCH blocks taint propagation without benefit.
                # Let them fall through to NO_MATCH so the taint engine handles them.
                # validator role now included: it may mark args as checked,
                # changing downstream taint propagation (Task 1).
                _USEFUL_ROLES = {"allocator", "copy", "read_input", "exec", "validator"}
                if role in _USEFUL_ROLES:
                    rule = learned
                    rule["source"] = "learned"
                    log.debug("Learned pattern match: %s (role=%s)", fn_name, role)
                else:
                    log.debug("Learned rule for %s (role=%s) skipped — not a taint sink",
                              fn_name, role)

        if rule is None:
            return self._no_match(fn_name, arg_sizes, arg_vars, output, seq)

        # Determine match kind from rule source
        source = rule.get("source", "hardcoded")
        kind   = (
            MatchKind.LIBRARY_MATCH
            if source == "hardcoded"
            else MatchKind.STRUCTURAL_MATCH   # llm_inferred OR pattern_inferred
        )

        return_var = self._extract_varname(output) if output else None

        # Post-match return-shape validation for inferred matches.
        # A STRUCTURAL_MATCH that claims a non-void return but the actual
        # op has no output varnode (or vice-versa) is likely a fingerprint
        # collision (e.g. memcpy vs strncmp both have [8,8,8] args).
        # Downgrade confidence rather than silently trusting the wrong rule.
        confidence = rule.get("confidence", 0.8)
        notes      = rule.get("notes", "")
        if kind == MatchKind.STRUCTURAL_MATCH:
            rule_has_ret   = rule.get("return_tainted", False) or rule.get("return_is_buffer", False)
            actual_has_ret = output is not None
            if rule_has_ret != actual_has_ret:
                confidence = max(0.3, confidence * 0.6)
                notes = notes + " [return-shape mismatch — confidence reduced]"
                log.debug(
                    "Return shape mismatch for %s: rule=%s actual=%s → conf=%.2f",
                    fn_name, rule_has_ret, actual_has_ret, confidence,
                )

        return MatchResult(
            op_seq            = seq,
            fn_name           = fn_name,
            arg_sizes         = arg_sizes,
            arg_vars          = arg_vars,
            return_var        = return_var,
            kind              = kind,

            external_input    = rule.get("external_input", []),
            writes_memory_at  = rule.get("writes_memory_at", -1),
            reads_from        = rule.get("reads_from", -1),
            return_tainted    = rule.get("return_tainted", False),
            bounded           = rule.get("bounded", False),
            size_arg          = rule.get("size_arg", -1),
            frees_memory_at   = rule.get("frees_memory_at", -1),
            return_is_buffer  = rule.get("return_is_buffer", False),

            is_sink           = rule.get("sink", False),
            sink_type         = rule.get("sink_type", ""),
            taint_arg         = rule.get("taint_arg", -1),

            marks_checked_args   = rule.get("marks_checked_args", []),
            external_source_args = rule.get("external_source_args", []),

            confidence        = confidence,
            vuln_score        = rule.get("vuln_score", 1.0),
            notes             = notes,
        )

    def match_all(self, ops: list[dict]) -> list[MatchResult]:
        """
        Match ALL CALL/CALLIND ops in a function's op list, including
        NO_MATCH results.  The taint engine must see unknown calls so it
        can apply conservative defaults (pointer-return taint, etc.) rather
        than silently skipping them and losing taint paths.

        Use match_all_known() if you only want confirmed pattern hits.
        """
        results = []
        for op in ops:
            r = self.match(op)
            if r is not None:
                results.append(r)
        return results

    def match_all_known(self, ops: list[dict]) -> list[MatchResult]:
        """
        Match all CALL/CALLIND ops and return only LIBRARY_MATCH and
        STRUCTURAL_MATCH results — NO_MATCH entries are excluded.
        Use this when you need only confirmed patterns (e.g. for scoring).
        """
        return [r for r in self.match_all(ops) if r.kind != MatchKind.NO_MATCH]

    def find_rule(self, fn_name: str, arg_sizes: list) -> Optional[MatchResult]:
        """
        Check if a named function has a known pattern rule (any kind).
        Returns MatchResult if known, None if NO_MATCH.
        Used by filter_agent to count unknown callees for scoring.
        """
        fake_op = {
            "op": "CALL",
            "inputs": [{"name": fn_name, "size": 8}] + [{"size": s} for s in arg_sizes],
            "output": None,
            "seq": -1,
        }
        result = self.match(fake_op)
        if result is None or result.kind == MatchKind.NO_MATCH:
            return None
        return result

    def find_unknown_calls(self, ops: list[dict]) -> list[dict]:
        """
        Return the raw op dicts for CALL ops that have no pattern match.
        These are candidates for further LLM analysis.
        """
        unknown = []
        for op in ops:
            if op.get("op") not in ("CALL", "CALLIND"):
                continue
            r = self.match(op)
            if r is not None and r.kind == MatchKind.NO_MATCH:
                unknown.append(op)
        return unknown

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_fn_name(fn_input: dict, op_type: str) -> str:
        """
        Extract the function name from inputs[0].

        Direct calls (CALL):
          inputs[0].name is usually the symbol name e.g. "strcpy"
          or a ram address e.g. "ram(0x401050)"

        Indirect calls (CALLIND):
          inputs[0] is a variable holding the function pointer
          Name is a VAR_N — we cannot resolve the target statically
        """
        if not isinstance(fn_input, dict):
            return "unknown"

        name = fn_input.get("name", "")

        if not name:
            return "unknown"

        # Clean up Ghidra symbol format
        # "<strcpy>@plt" → "strcpy"
        # "ram(0x401050)" → "0x401050"
        # Strip @plt/@got first, then strip < >
        if "@" in name:
            name = name.split("@")[0]
        if name.startswith("<") and name.endswith(">"):
            name = name[1:-1]
        if name.startswith("ram(") and name.endswith(")"):
            name = name[4:-1]

        # CALLIND — function pointer, name is a variable
        if op_type == "CALLIND":
            return f"INDIRECT:{name}"

        return name.strip()

    @staticmethod
    def _extract_size(inp: dict) -> int:
        """Extract the byte size of an argument varnode."""
        if not isinstance(inp, dict):
            return 0
        return int(inp.get("size", 0))

    @staticmethod
    def _extract_varname(inp) -> str:
        """Extract the canonical variable name from a varnode dict."""
        if not isinstance(inp, dict):
            return ""
        return inp.get("name", "")

    def _no_match(
        self,
        fn_name:   str,
        arg_sizes: list[int],
        arg_vars:  list[str],
        output,
        seq:       int,
    ) -> MatchResult:
        """
        Build a conservative NO_MATCH result.
        Taint engine will use safe defaults:
          - if any arg is pointer-sized → mark its memory as potentially tainted
          - return value → tainted only when it is pointer-sized (8B)

        Marking every unknown call's return value as tainted regardless of
        size floods the taint engine with speculative paths. A 4-byte int
        return (error code, count) rarely becomes the buffer used in a later
        overflow; an 8-byte pointer return very often does.
        """
        return_var  = self._extract_varname(output) if output else None
        ret_size    = output.get("size", 0) if isinstance(output, dict) else 0
        # Only propagate taint through the return value when it is pointer-sized.
        ret_tainted = ret_size == 8

        return MatchResult(
            op_seq            = seq,
            fn_name           = fn_name or "unknown",
            arg_sizes         = arg_sizes,
            arg_vars          = arg_vars,
            return_var        = return_var,
            kind              = MatchKind.NO_MATCH,

            external_input    = [],
            writes_memory_at  = -1,
            reads_from        = -1,
            return_tainted    = ret_tainted,
            bounded           = False,
            size_arg          = -1,
            frees_memory_at   = -1,
            return_is_buffer  = False,

            is_sink           = False,
            sink_type         = "",
            taint_arg         = -1,

            marks_checked_args   = [],
            external_source_args = [],

            confidence        = 0.0,
            vuln_score        = 0.0,
            notes             = (
                f"unknown function — conservative defaults applied "
                f"(return_tainted={ret_tainted}, ret_size={ret_size}B)"
            ),
        )


    @staticmethod
    def _is_raw_address(fn_name: str) -> bool:
        """True if fn_name looks unresolved — raw address or indirect through variable."""
        return fn_name.startswith("0x") or fn_name.startswith("INDIRECT:")

    @staticmethod
    def _is_static_unresolved(fn_name: str) -> bool:
        """True only for static unresolved calls (raw hex address from stripped binary).
        INDIRECT:VAR_N calls are excluded — we cannot fingerprint a runtime function pointer."""
        return fn_name.startswith("0x")

    def _groq_resolve(
        self,
        fn_name:    str,
        arg_sizes:  list[int],
        arg_vars:   list[str],
        output:     Optional[dict]  = None,
        context:    Optional[dict]  = None,
    ) -> Optional[str]:
        """
        Ask Groq to identify an unknown CALL by its structural pattern.
        Returns guessed function name (e.g. 'recv') or None.
        Silent no-op if GROQ_API_KEY is not set.

        Parameters
        ----------
        context : optional dict with call-site hints:
            const_args    — list of (index, hex_value) for constant arguments
            return_checked — True if a CBRANCH follows the call (return is error-checked)
            caller_proto  — decompiled signature of the containing function
        """
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None

        try:
            from groq import Groq

            ctx = context or {}

            arg_desc = "\n".join(
                f"  arg{i}: size={s}B  var={v}"
                for i, (s, v) in enumerate(zip(arg_sizes, arg_vars))
            ) or "  (none)"

            ret_size = output.get("size", 0) if isinstance(output, dict) else 0
            ret_line = f"  return_size : {ret_size}B" if output else "  return_size : void"

            # Constant args narrow the candidate set significantly
            const_args = ctx.get("const_args", [])
            const_desc = (
                "\n".join(f"  arg{i} is constant: {v}" for i, v in const_args)
                if const_args else "  (none)"
            )

            # Error-check pattern: CBRANCH immediately after call means the
            # return value is compared (typical for fd checks, byte counts, etc.)
            checked = ctx.get("return_checked", False)
            caller  = ctx.get("caller_proto", "(unknown)")

            prompt = (
                "You are a binary reverse engineering assistant.\n"
                "Identify the libc / POSIX / WinSock / system function for "
                "this unidentified call in a stripped binary.\n\n"
                f"  target        : {fn_name}\n"
                f"  num_args      : {len(arg_sizes)}\n"
                f"{ret_line}\n"
                f"  return_checked: {checked}  "
                f"(True = return value used in a branch — suggests error code)\n\n"
                f"Arguments:\n{arg_desc}\n\n"
                f"Constant arguments (known values at call site):\n{const_desc}\n\n"
                f"Caller function signature: {caller}\n\n"
                "Reply with ONLY the function name (e.g. recv). "
                "If unsure reply: UNKNOWN"
            )

            client   = Groq()
            response = client.chat.completions.create(
                model       = "llama-3.3-70b-versatile",
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 16,
                temperature = 0.0,
            )
            name = response.choices[0].message.content.strip().lower()
            return None if name == "unknown" else name

        except Exception as exc:
            log.debug("Groq resolve failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Function Ranker — top-N by vulnerability score
# ─────────────────────────────────────────────────────────────────────────────

# Scoring weights
_W = {
    "is_sink"         : 10,   # dangerous sink confirmed
    "unbounded_write" : 5,    # sink + unbounded + writes memory
    "external_input"  : 3,    # brings attacker data in
    "return_tainted"  : 2,    # return value carries taint
    "structural_mult" : 0.8,  # confidence penalty for LLM-inferred
    "no_match_mult"   : 0.5,  # conservative penalty for unknowns
    "sink_bonus": {
        "command_injection" : 5,
        "buffer_overflow"   : 4,
        "format_string"     : 3,
        "uaf"               : 2,
    },
}

_SINK_PRIORITY = ["command_injection", "buffer_overflow", "format_string", "uaf"]


class FunctionScore:
    """Aggregated vulnerability score for one function."""
    __slots__ = (
        "fn_name", "entry_addr", "total_score",
        "call_scores", "sink_count", "source_count", "top_sink_type",
    )

    def __init__(
        self,
        fn_name:       str,
        entry_addr:    str,
        total_score:   float,
        call_scores:   list,
        sink_count:    int,
        source_count:  int,
        top_sink_type: str,
    ):
        self.fn_name       = fn_name
        self.entry_addr    = entry_addr
        self.total_score   = total_score
        self.call_scores   = call_scores
        self.sink_count    = sink_count
        self.source_count  = source_count
        self.top_sink_type = top_sink_type


class FunctionRanker:
    """
    Scores every function by vulnerability risk and returns the top-N.

    Scoring per CALL op
    -------------------
      base          = MatchResult.vuln_score   (1–10 from pattern store)
      +10           if is_sink
      +5            if sink + unbounded + writes memory
      +3            if external_input
      +2            if return_tainted
      +bonus        per sink_type
      × confidence  (0.5 for NO_MATCH, 0.8 for STRUCTURAL, 1.0 for LIBRARY)

    Function score  = sum of all call scores in the function.

    Usage
    -----
        ranker  = FunctionRanker(matcher)
        # functions: {fn_name: {"entry": addr, "ops": [op, ...]}}
        top50   = ranker.rank(functions, top_n=50)
        ranker.print_ranking(top50)
    """

    def __init__(self, matcher: PatternMatcher):
        self.matcher = matcher

    # ── Public API ────────────────────────────────────────────────────

    def rank(
        self,
        functions: dict,
        top_n: int = 50,
    ) -> list[FunctionScore]:
        """
        Score all functions and return top-N sorted by vuln score.

        Parameters
        ----------
        functions : {fn_name: {"entry": str, "ops": list[dict]}}
        top_n     : how many to return (default 50)
        """
        scored = []
        for fn_name, fn_data in functions.items():
            scored.append(self._score_function(
                fn_name    = fn_name,
                entry_addr = fn_data.get("entry", ""),
                ops        = fn_data.get("ops", []),
            ))
        scored.sort(key=lambda x: x.total_score, reverse=True)
        return scored[:top_n]

    def rank_from_matches(
        self,
        fn_name:    str,
        entry_addr: str,
        matches:    list[MatchResult],
    ) -> FunctionScore:
        """Score a single function from already-computed MatchResults."""
        return self._score_from_matches(fn_name, entry_addr, matches)

    def print_ranking(self, ranking: list[FunctionScore], top_n: int = 50) -> None:
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  TOP {min(top_n, len(ranking))} FUNCTIONS BY VULNERABILITY SCORE")
        print(f"{sep}")
        print(f"  {'#':<4} {'Score':>7}  {'Sinks':>5}  {'Sources':>7}  "
              f"{'Top sink type':<22}  Function")
        print(f"  {'─'*4} {'─'*7}  {'─'*5}  {'─'*7}  {'─'*22}  {'─'*30}")
        for rank, fs in enumerate(ranking[:top_n], 1):
            print(
                f"  {rank:<4} {fs.total_score:>7.1f}  "
                f"{fs.sink_count:>5}  {fs.source_count:>7}  "
                f"{(fs.top_sink_type or '—'):<22}  "
                f"{fs.fn_name}  [{fs.entry_addr}]"
            )
        print(f"{sep}\n")

    # ── Scoring internals ────────────────────────────────────────────

    def _score_function(
        self,
        fn_name:    str,
        entry_addr: str,
        ops:        list[dict],
    ) -> FunctionScore:
        matches = [
            r for op in ops
            for r in [self.matcher.match(op)]
            if r is not None
        ]
        return self._score_from_matches(fn_name, entry_addr, matches)

    def _score_from_matches(
        self,
        fn_name:    str,
        entry_addr: str,
        matches:    list[MatchResult],
    ) -> FunctionScore:
        total        = 0.0
        call_scores  = []
        sink_count   = 0
        source_count = 0
        seen_sinks:  list[str] = []

        for r in matches:
            score, breakdown = self._score_call(r)
            total += score
            call_scores.append({
                "fn"        : r.fn_name,
                "seq"       : r.op_seq,
                "score"     : round(score, 2),
                "breakdown" : breakdown,
                "kind"      : r.kind.name,
            })
            if r.is_sink:
                sink_count += 1
                if r.sink_type and r.sink_type not in seen_sinks:
                    seen_sinks.append(r.sink_type)
            if r.is_external_source():
                source_count += 1

        top_sink = next((s for s in _SINK_PRIORITY if s in seen_sinks), "")

        return FunctionScore(
            fn_name       = fn_name,
            entry_addr    = entry_addr,
            total_score   = round(total, 2),
            call_scores   = call_scores,
            sink_count    = sink_count,
            source_count  = source_count,
            top_sink_type = top_sink,
        )

    def _score_call(self, r: MatchResult) -> tuple[float, dict]:
        """Compute vuln score for one MatchResult. Returns (score, breakdown)."""
        breakdown: dict = {}

        base = r.vuln_score
        breakdown["base_vuln_score"] = base
        score = base

        if r.is_sink:
            score += _W["is_sink"]
            breakdown["is_sink"] = _W["is_sink"]
            bonus = _W["sink_bonus"].get(r.sink_type, 0)
            if bonus:
                score += bonus
                breakdown[f"sink_bonus({r.sink_type})"] = bonus
            if not r.bounded and r.writes_memory_at != -1:
                score += _W["unbounded_write"]
                breakdown["unbounded_write"] = _W["unbounded_write"]

        if r.is_external_source():
            score += _W["external_input"]
            breakdown["external_input"] = _W["external_input"]

        if r.return_tainted:
            score += _W["return_tainted"]
            breakdown["return_tainted"] = _W["return_tainted"]

        conf = {
            MatchKind.LIBRARY_MATCH    : r.confidence,
            MatchKind.STRUCTURAL_MATCH : r.confidence * _W["structural_mult"],
            MatchKind.NO_MATCH         : r.confidence * _W["no_match_mult"],
        }.get(r.kind, r.confidence)
        score *= conf
        breakdown["conf_mult"] = round(conf, 2)

        return round(score, 2), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    store   = PatternStore("pattern_store.db")
    matcher = PatternMatcher(store)

    sep = "─" * 62
    print(f"\n{sep}")
    print("Pattern Matcher tests")
    print(f"{sep}\n")

    # ── Test 1: strcpy — classic overflow ─────────────────────────────
    print("Test 1: strcpy CALL op\n")
    strcpy_op = {
        "op":  "CALL",
        "seq": 5,
        "addr": "0x401150",
        "output": None,
        "inputs": [
            {"name": "strcpy",  "size": 0},   # fn ptr
            {"name": "VAR_2",   "size": 8},   # dst
            {"name": "VAR_0",   "size": 8},   # src
        ],
    }
    r = matcher.match(strcpy_op)
    assert r is not None
    assert r.fn_name    == "strcpy"
    assert r.is_sink    == True
    assert r.sink_type  == "buffer_overflow"
    assert r.bounded    == False
    assert r.kind       == MatchKind.LIBRARY_MATCH
    assert r.written_memory_var() == ["VAR_2"]   # dst gets written
    assert r.source_var()         == "VAR_0"     # src is read
    print(f"  {r.summary()}")
    print(f"  writes_memory_at var : {r.written_memory_var()}")
    print(f"  source var           : {r.source_var()}")
    print(f"  is_sink              : {r.is_sink}")
    print(f"  kind                 : {r.kind.name}\n")

    # ── Test 2: recv — external input ─────────────────────────────────
    print("Test 2: recv CALL op\n")
    recv_op = {
        "op":  "CALL",
        "seq": 2,
        "addr": "0x401140",
        "output": {"name": "VAR_3", "size": 8},
        "inputs": [
            {"name": "recv",          "size": 0},
            {"name": "VAR_1",         "size": 8},   # sockfd
            {"name": "VAR_2",         "size": 8},   # buf
            {"name": "const(0x400)",  "size": 4},   # len
            {"name": "const(0x0)",    "size": 4},   # flags
        ],
    }
    r2 = matcher.match(recv_op)
    assert r2 is not None
    assert r2.is_external_source()   == True
    assert r2.bounded                == True
    assert r2.return_tainted         == True
    assert r2.kind                   == MatchKind.LIBRARY_MATCH
    print(f"  {r2.summary()}")
    print(f"  external_input vars  : {r2.tainted_arg_vars()}")
    print(f"  written_memory_var   : {r2.written_memory_var()}")
    print(f"  return_var (tainted) : {r2.return_var}\n")

    # ── Test 3: memcpy — bounded copy ─────────────────────────────────
    print("Test 3: memcpy CALL op\n")
    memcpy_op = {
        "op":  "CALL",
        "seq": 8,
        "addr": "0x401160",
        "output": {"name": "VAR_5", "size": 8},
        "inputs": [
            {"name": "memcpy",       "size": 0},
            {"name": "VAR_3",        "size": 8},   # dst
            {"name": "VAR_2",        "size": 8},   # src
            {"name": "const(0x40)",  "size": 4},   # size = 64
        ],
    }
    r3 = matcher.match(memcpy_op)
    assert r3 is not None
    assert r3.bounded   == True
    assert r3.is_sink   == False   # bounded — not automatically a sink
    assert r3.size_vars() == ["const(0x40)"]
    print(f"  {r3.summary()}")
    print(f"  bounded              : {r3.bounded}")
    print(f"  size_vars            : {r3.size_vars()}\n")

    # ── Test 4: system — command injection ────────────────────────────
    print("Test 4: system CALL op\n")
    system_op = {
        "op":  "CALL",
        "seq": 12,
        "addr": "0x401170",
        "output": {"name": "VAR_6", "size": 4},
        "inputs": [
            {"name": "system",  "size": 0},
            {"name": "VAR_0",   "size": 8},   # cmd string
        ],
    }
    r4 = matcher.match(system_op)
    assert r4 is not None
    assert r4.is_sink   == True
    assert r4.sink_type == "command_injection"
    print(f"  {r4.summary()}")
    print(f"  taint_arg (cmd var)  : {r4.arg_vars[r4.taint_arg] if r4.taint_arg >= 0 else 'n/a'}\n")

    # ── Test 5: unknown address — pattern fingerprint match ───────────
    print("Test 5: unknown address — pattern fingerprint match\n")
    unknown_op = {
        "op":  "CALL",
        "seq": 20,
        "addr": "0x401200",
        "output": {"name": "VAR_7", "size": 8},
        "inputs": [
            {"name": "ram(0x401020)", "size": 0},   # stripped address
            {"name": "VAR_2",         "size": 8},   # arg0 — pointer
            {"name": "VAR_0",         "size": 8},   # arg1 — pointer
            {"name": "const(0x40)",   "size": 4},   # arg2 — size
        ],
    }
    r5 = matcher.match(unknown_op)
    assert r5 is not None
    # [8,8,4] + 8B return → pattern fingerprint resolves to memcpy
    # kind is STRUCTURAL_MATCH (resolved via pattern DB) or NO_MATCH (no GROQ_API_KEY)
    assert r5.kind in (MatchKind.STRUCTURAL_MATCH, MatchKind.NO_MATCH)
    print(f"  {r5.summary()}")
    print(f"  kind          : {r5.kind.name}")
    print(f"  resolved name : {r5.fn_name}")
    print(f"  return tainted: {r5.return_tainted}\n")

    # ── Test 5b: truly unknown pattern — no fingerprint match ─────────
    print("Test 5b: truly unknown pattern (unique arg signature)\n")
    truly_unknown_op = {
        "op":  "CALL",
        "seq": 21,
        "addr": "0x401300",
        "output": None,
        "inputs": [
            {"name": "ram(0x401099)", "size": 0},   # stripped
            {"name": "VAR_9",         "size": 4},   # int
            {"name": "VAR_10",        "size": 2},   # short — rare combo
            {"name": "VAR_11",        "size": 1},   # byte
            {"name": "VAR_12",        "size": 4},   # int
            {"name": "VAR_13",        "size": 4},   # int
        ],
    }
    r5b = matcher.match(truly_unknown_op)
    assert r5b is not None
    # [4,2,1,4,4] void — no library function has this exact shape
    # pattern lookup will miss, Groq will miss (no key), so NO_MATCH
    assert r5b.kind        == MatchKind.NO_MATCH
    assert r5b.return_tainted == True   # conservative default
    print(f"  {r5b.summary()}")
    print(f"  kind                 : {r5b.kind.name}")
    print(f"  conservative return  : {r5b.return_tainted}\n")
    print("Test 6: symbol name cleanup (PLT format)\n")
    plt_op = {
        "op":  "CALL",
        "seq": 3,
        "addr": "0x401130",
        "output": None,
        "inputs": [
            {"name": "<strcpy>@plt",  "size": 0},   # Ghidra PLT format
            {"name": "VAR_2",         "size": 8},
            {"name": "VAR_0",         "size": 8},
        ],
    }
    r6 = matcher.match(plt_op)
    assert r6 is not None
    assert r6.fn_name  == "strcpy"    # correctly stripped
    assert r6.is_sink  == True
    print(f"  raw name  : '<strcpy>@plt'")
    print(f"  resolved  : '{r6.fn_name}'")
    print(f"  matched   : {r6.kind.name}\n")

    # ── Test 7: match_all on a function ───────────────────────────────
    print("Test 7: match_all — scan all ops in a function\n")
    func_ops = [
        {"op": "LOAD",   "seq": 0, "output": {"name":"VAR_1","size":8}, "inputs":[]},
        recv_op,
        {"op": "INT_ADD","seq": 4, "output": {"name":"VAR_4","size":4}, "inputs":[]},
        strcpy_op,
        {"op": "RETURN", "seq": 9, "output": None, "inputs":[]},
    ]
    matches = matcher.match_all(func_ops)
    assert len(matches) == 2   # recv + strcpy, not LOAD/INT_ADD/RETURN
    print(f"  ops in function : {len(func_ops)}")
    print(f"  CALL matches    : {len(matches)}")
    for m in matches:
        print(f"    seq={m.op_seq}  {m.summary()}")

    print(f"\n{sep}")
    print("All pattern matcher tests passed ✓")
    print(f"{sep}")

    # ── Test 8: FunctionRanker top-50 ─────────────────────────────────
    print(f"\n{sep}")
    print("Test 8: FunctionRanker — top-50 ranking\n")

    ranker = FunctionRanker(matcher)

    functions = {
        "main": {
            "entry": "0x401000",
            "ops": [recv_op, strcpy_op],      # external source → unbounded sink
        },
        "handle_request": {
            "entry": "0x401200",
            "ops": [recv_op, memcpy_op],      # source → bounded copy
        },
        "run_command": {
            "entry": "0x401400",
            "ops": [system_op],               # command injection only
        },
        "helper": {
            "entry": "0x401600",
            "ops": [memcpy_op],               # bounded copy only
        },
    }

    ranking = ranker.rank(functions, top_n=50)
    ranker.print_ranking(ranking)

    assert ranking[0].fn_name  == "main",   "main should rank highest"
    assert ranking[-1].fn_name == "helper", "helper should rank lowest"

    print("  Per-call breakdown for top function:")
    for cs in ranking[0].call_scores:
        print(f"    seq={cs['seq']}  {cs['fn']}  score={cs['score']}  kind={cs['kind']}")
        for k, v in cs["breakdown"].items():
            print(f"      {k}: {v}")

    print(f"\n{sep}")
    print("All tests passed ✓")
    print(f"{sep}")
    store.close()