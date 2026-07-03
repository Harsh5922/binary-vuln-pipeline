"""
interprocedural.py

Summary-based inter-procedural taint analysis.

Problem solved
--------------
The intra-procedural taint engine stops at every CALL boundary.  When an
internal helper function propagates tainted data from argument to return
value, the engine has no way to know this — the call is marked NO_MATCH
and taint either dies (false negative) or is naively conserved (false positive).

Solution
--------
For each internal function, compute a FuncSummary that records:
  - which argument positions, when tainted, cause the return value to be tainted
  - which argument positions, when tainted, cause pointed-to memory to be tainted

Summaries are built in call-graph order (callees before callers) so that when
a function is summarized, its callees' summaries are already available.  The
TaintEngine then looks up the callee's summary at each NO_MATCH CALL site
instead of falling back to the conservative generic propagation.

Usage
-----
    from interprocedural import SummaryDatabase
    from pattern_matcher  import PatternMatcher
    from taint_engine     import TaintEngine

    db = SummaryDatabase(matcher)
    db.build(funcs)                          # funcs = list of function dicts

    engine = TaintEngine(matcher, summary_db=db)
    results = engine.analyze_all(funcs)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pattern_matcher import PatternMatcher

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FuncSummary:
    """
    Taint-transfer summary for one internal function.

    arg_effects maps argument index → frozenset of effect strings:
      "return"   — the function's return value becomes tainted
      "mem:N"    — memory pointed to by argument N becomes tainted

    frees_fields is a frozenset of (arg_index, field_offset) tuples meaning:
      the function frees the struct field at byte-offset `field_offset` inside
      argument `arg_index`.  Callers use this to update freed_fields in
      TaintState so that post-call accesses through the same field are flagged.

    Example: png_image_free_function(image) frees image->opaque (offset 0x228)
      frees_fields = frozenset({(0, 0x228)})
    """
    func_name:    str
    arg_effects:  dict[int, frozenset[str]]
    frees_fields: frozenset[tuple[int, int]] = frozenset()
    # Which arg indices receive external data transitively (fread/recv/etc.)
    # Used by TaintEngine to seed external taint when processing this function.
    externally_tainted_args: frozenset[int] = frozenset()

    def has_effects(self) -> bool:
        return bool(self.arg_effects) or bool(self.frees_fields)

    def __repr__(self) -> str:
        return f"FuncSummary({self.func_name!r}, {dict(self.arg_effects)}, frees={set(self.frees_fields)})"


# ─────────────────────────────────────────────────────────────────────────────
# Call-graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_call_graph(funcs: list[dict]) -> dict[str, list[str]]:
    """Build caller → [internal callee] adjacency list."""
    known = {f["name"] for f in funcs}
    return {
        f["name"]: [cs for cs in f.get("call_sites", []) if cs in known]
        for f in funcs
    }


def _topological_sort(call_graph: dict[str, list[str]]) -> list[str]:
    """
    Return node names ordered so callees appear before callers.
    Uses Kahn's algorithm; functions in cycles are appended at the end
    in arbitrary order (they will receive conservative summaries).
    """
    all_nodes = set(call_graph)

    # in_degree[caller] = number of its internal callees not yet processed
    in_degree: dict[str, int] = {n: 0 for n in all_nodes}
    # reverse[callee] = callers that depend on this callee being summarized first
    reverse:   dict[str, list[str]] = {n: [] for n in all_nodes}

    for caller, callees in call_graph.items():
        for callee in callees:
            if callee in all_nodes:
                in_degree[caller] += 1
                reverse[callee].append(caller)

    queue = deque(n for n in all_nodes if in_degree[n] == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for caller in reverse.get(node, []):
            in_degree[caller] -= 1
            if in_degree[caller] == 0:
                queue.append(caller)

    # Append cycle members in arbitrary order
    processed = set(order)
    order.extend(n for n in all_nodes if n not in processed)

    return order


# ─────────────────────────────────────────────────────────────────────────────
# Parameter and return-value extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_params(ops: list[dict]) -> list[str]:
    """
    Heuristically identify function parameter varnodes.

    Scans ops before the first CALL and collects 8-byte non-constant
    non-ram varnodes in order of first appearance.  These are the
    pointer-sized parameters that could carry taint into the function.
    """
    seen:   set[str]  = set()
    params: list[str] = []

    for op in ops:
        if op.get("op") in ("CALL", "CALLIND"):
            break
        for inp in (op.get("inputs") or []):
            if not isinstance(inp, dict):
                continue
            name = inp.get("name", "")
            size = inp.get("size", 0)
            if (
                name
                and size == 8
                and name not in seen
                and not name.startswith("const(")
                and not name.startswith("ram(")
            ):
                params.append(name)
                seen.add(name)

    return params


def _find_return_var(ops: list[dict]) -> Optional[str]:
    """
    Find the variable returned by the function (last RETURN op, first input).
    """
    for op in reversed(ops):
        if op.get("op") == "RETURN":
            inputs = op.get("inputs") or []
            if inputs and isinstance(inputs[0], dict):
                name = inputs[0].get("name", "")
                if name and not name.startswith("const("):
                    return name
    return None


# Ops that produce a varnode that is semantically "the same pointer" as one of
# their inputs — used to build the param-alias map for summary computation.
_ALIAS_OPS = frozenset({
    "COPY", "CAST",
    "INT_ZEXT", "INT_SEXT",
    "PTRADD", "PTRSUB",     # pointer arithmetic: still derived from the same base
})


def _compute_param_aliases(ops: list[dict], params: list[str]) -> dict[str, int]:
    """
    Build a mapping varnode_name → param_index for any varnode that is
    transitively derived from a param via COPY / CAST / pointer arithmetic.

    Used in summary computation: when mem[VAR_dst] is tainted and VAR_dst is
    an alias of param[0], we know param[0]'s memory was written.

    Example:
        params = ["VAR_h0", "VAR_h1"]
        COPY VAR_h_dst ← VAR_h0    → alias["VAR_h_dst"] = 0
        PTRADD VAR_h_ptr ← VAR_h_dst, const(8) → alias["VAR_h_ptr"] = 0
    """
    alias: dict[str, int] = {p: i for i, p in enumerate(params)}

    for op in ops:
        if op.get("op") not in _ALIAS_OPS:
            continue
        out = op.get("output")
        if not out:
            continue
        out_name = _vn(out)
        if not out_name or out_name in alias:
            continue
        # If any input is a known alias, propagate the param index
        for inp in (op.get("inputs") or []):
            inp_name = _vn(inp)
            if inp_name in alias:
                alias[out_name] = alias[inp_name]
                break

    return alias


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight taint propagation (summary computation only)
# ─────────────────────────────────────────────────────────────────────────────

# Duplicated from taint_engine to avoid circular imports.
# These cover all ops that propagate taint through data flow.
_CALL_OPS = frozenset({"CALL", "CALLIND"})
_ARITH_OPS = frozenset({
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
    "CAST", "COPY", "PIECE", "SUBPIECE", "MULTIEQUAL",
})
_SKIP_OPS = frozenset({"INDIRECT"})


_FREE_CALL_NAMES = frozenset({"free", "png_free", "g_free", "cfree", "png_free_data"})
# For png_free(png_ptr, ptr): the freed argument is at index 1, not 0.
_FREE_ARG_INDEX: dict[str, int] = {"png_free": 1, "png_free_data": 2}


def _const_int(inp: dict) -> Optional[int]:
    """Parse a constant varnode name like 'const(0x228)' → integer, or None."""
    name = inp.get("name", "")
    if not name.startswith("const(") or inp.get("space") != "const":
        return None
    try:
        return int(name[6:-1], 16)
    except (ValueError, IndexError):
        return None


def build_addr_to_name(funcs: list[dict]) -> dict[str, str]:
    """
    Build a map from normalized hex address → function name.

    Reads the ``entry_addr`` field of each function dict (format: ``'0010b614'``
    — 8 hex chars, no ``0x`` prefix, leading zeros) and normalizes it by stripping
    leading zeros so that it matches the constant emitted by Ghidra's P-code for
    ``PTRSUB(const(0x0), const(0x10b614))`` → key ``'10b614'``.
    """
    result: dict[str, str] = {}
    for f in funcs:
        entry = f.get("entry_addr", "")
        if entry:
            normalized = entry.lstrip("0") or "0"
            result[normalized.lower()] = f["name"]
    return result


def _collect_all_params_ordered(ops: list[dict]) -> list[str]:
    """
    Return true function-parameter varnodes ordered by first appearance.

    A true parameter is a variable that is used as an input in some op but
    never produced as an output in the same function.  Unlike ``_extract_params``
    (which stops before the first CALL and misses parameters that only appear
    late in the body), this scan covers the entire op list.

    Only register-space, 4- or 8-byte varnodes are included — stack vars,
    unique vars, and external function-name refs are excluded.
    """
    defined: set[str] = set()
    for op in ops:
        out = op.get("output")
        if out and isinstance(out, dict):
            n = out.get("name", "")
            if n:
                defined.add(n)

    seen:    set[str]  = set()
    ordered: list[str] = []
    for op in ops:
        for inp in (op.get("inputs") or []):
            if not isinstance(inp, dict):
                continue
            name  = inp.get("name", "")
            size  = inp.get("size", 0)
            space = inp.get("space", "")
            if (
                name
                and name not in seen
                and name not in defined
                and size in (4, 8)
                and space == "register"
                and not name.startswith("const(")
                and not name.startswith("ram(")
            ):
                ordered.append(name)
                seen.add(name)

    return ordered


def _compute_frees_fields(
    ops:              list[dict],
    params:           list[str],
    addr_to_name:     Optional[dict[str, str]]            = None,
    known_summaries:  Optional[dict[str, "FuncSummary"]]  = None,
) -> frozenset[tuple[int, int]]:
    """
    Scan a function's ops to find which struct fields of its arguments are freed.

    Returns frozenset of (param_index, field_offset) tuples.  Each tuple means:
    "this function frees the pointer stored at byte offset `field_offset` inside
    the struct pointed to by argument `param_index`."

    Patterns detected:
      Direct free:
        PTRSUB  VAR_addr  ← VAR_base  const(offset)
        LOAD    VAR_ptr   ← VAR_addr
        CALL    free      ← VAR_ptr   (or png_free(_, VAR_ptr))

      Indirect free via CALLIND with a resolved function pointer:
        PTRSUB  VAR_fn    ← const(0x0)  const(ADDR)      (fn-ptr constant)
        CALLIND VAR_ret   ← VAR_fn  VAR_arg ...
        → if the resolved function has frees_fields, propagate them.

    addr_to_name  : normalized-hex-address → function name (from build_addr_to_name)
    known_summaries : already-computed FuncSummary objects for callee lookup
    """
    param_alias = _compute_param_aliases(ops, params)
    # Maps: var → (base_var, byte_offset)
    ptr_offsets:    dict[str, tuple[str, int]] = {}
    field_load_src: dict[str, tuple[str, int]] = {}
    freed: set[tuple[int, int]] = set()
    # fn_ptr_vars: maps a varnode to the function name it points to
    # (populated when PTRSUB(const(0), const(ADDR)) is seen)
    fn_ptr_vars: dict[str, str] = {}

    for op in ops:
        op_type = op.get("op", "")
        inputs  = op.get("inputs") or []
        out_v   = _vn(op.get("output") or {})

        if op_type == "PTRSUB" and out_v and len(inputs) >= 2:
            base_v = _vn(inputs[0])
            cval   = _const_int(inputs[1]) if isinstance(inputs[1], dict) else None
            if base_v and cval is not None:
                ptr_offsets[out_v] = (base_v, cval)
                # Function-pointer constant: PTRSUB(const(0x0), const(ADDR))
                b0 = _const_int(inputs[0]) if isinstance(inputs[0], dict) else None
                if b0 == 0 and addr_to_name and cval > 0:
                    hex_addr = hex(cval)[2:].lower()
                    fn_name = addr_to_name.get(hex_addr)
                    if fn_name:
                        fn_ptr_vars[out_v] = fn_name

        elif op_type in ("COPY", "CAST") and out_v and inputs:
            src = _vn(inputs[0])
            if src and src in fn_ptr_vars:
                fn_ptr_vars[out_v] = fn_ptr_vars[src]

        elif op_type == "LOAD" and out_v:
            addr_v = _vn(inputs[1]) if len(inputs) >= 2 else (_vn(inputs[0]) if inputs else "")
            if addr_v and addr_v in ptr_offsets:
                field_load_src[out_v] = ptr_offsets[addr_v]

        elif op_type in ("CALL", "CALLIND") and inputs:
            fn_name_v = _vn(inputs[0])
            arg_vars  = [_vn(i) for i in inputs[1:]]

            # For CALLIND: try to resolve the callee from fn_ptr_vars.
            if op_type == "CALLIND":
                resolved = fn_ptr_vars.get(fn_name_v)
                if resolved:
                    fn_name_v = resolved
                else:
                    continue   # unresolvable indirect call — skip

            # Case 1: callee is a known free() function.
            if fn_name_v in _FREE_CALL_NAMES:
                freed_arg_idx = _FREE_ARG_INDEX.get(fn_name_v, 0)
                if freed_arg_idx >= len(arg_vars):
                    continue
                freed_var = arg_vars[freed_arg_idx]
                if freed_var not in field_load_src:
                    continue
                base_var, offset = field_load_src[freed_var]
                param_idx = param_alias.get(base_var)
                if param_idx is not None:
                    freed.add((param_idx, offset))

            # Case 2: callee has a known frees_fields summary — propagate.
            elif known_summaries and fn_name_v in known_summaries:
                callee_summary = known_summaries[fn_name_v]
                for (callee_param_idx, field_offset) in callee_summary.frees_fields:
                    if callee_param_idx >= len(arg_vars):
                        continue
                    freed_var = arg_vars[callee_param_idx]
                    # Map freed_var back to one of our own params.
                    param_idx = param_alias.get(freed_var)
                    if param_idx is None:
                        # Try transitively via field_load_src / ptr_offsets
                        if freed_var in field_load_src:
                            base_var, _ = field_load_src[freed_var]
                            param_idx = param_alias.get(base_var)
                    if param_idx is not None:
                        freed.add((param_idx, field_offset))

    return frozenset(freed)


def _vn(inp) -> str:
    """Extract varnode name from a dict or return empty string."""
    return inp.get("name", "") if isinstance(inp, dict) else ""


def _mini_analyze(
    ops:      list[dict],
    seed_var: str,
    matcher:  "PatternMatcher",
    db:       "SummaryDatabase",
) -> tuple[set[str], set[str]]:
    """
    Lightweight single-seed taint propagation for summary computation.

    Seeds taint from seed_var and propagates through all ops.
    Returns (tainted_vars, tainted_mem_vars).

    No VulnCandidate generation — this is purely a dataflow probe.
    Uses the partially-built summary database for inter-proc calls so
    that already-summarized callees contribute accurate propagation.
    """
    var_taint: dict[str, bool] = {}
    mem_taint: dict[str, bool] = {}

    def is_v(v: str) -> bool:
        return bool(v) and var_taint.get(v, False)

    def is_m(v: str) -> bool:
        return bool(v) and mem_taint.get(v, False)

    def tv(v: str) -> None:
        if v:
            var_taint[v] = True

    def tm(v: str) -> None:
        if v:
            mem_taint[v] = True

    # Seed
    tv(seed_var)

    for op in ops:
        op_type = op.get("op", "")

        if op_type in _SKIP_OPS:
            continue

        # ── CALL / CALLIND ────────────────────────────────────────────
        if op_type in _CALL_OPS:
            inputs     = op.get("inputs") or []
            out        = op.get("output")
            ret_var    = _vn(out) if out else ""
            fn_name    = _vn(inputs[0]) if inputs else ""
            arg_vars   = [_vn(i) for i in inputs[1:]]
            arg_sizes  = [i.get("size", 0) if isinstance(i, dict) else 0 for i in inputs[1:]]
            any_tainted = any(is_v(v) or is_m(v) for v in arg_vars)

            if not any_tainted:
                continue

            # Try pattern_matcher first (known library rules)
            result = matcher.match(op)
            if result is not None and result.kind.name != "NO_MATCH":
                if result.return_tainted and ret_var:
                    tv(ret_var)
                for src in (result.tainted_arg_vars() or []):
                    tv(src)
                for dst in (result.written_memory_var() or []):
                    src = result.source_var()
                    if src and is_v(src):
                        tm(dst)
                if result.is_external_source():
                    for dv in (result.written_memory_var() or []):
                        tm(dv)
                continue

            # Try inter-proc summary for internal functions
            summary = db.get(fn_name)
            if summary:
                for arg_idx, effects in summary.arg_effects.items():
                    if arg_idx < len(arg_vars) and is_v(arg_vars[arg_idx]):
                        if "return" in effects and ret_var:
                            tv(ret_var)
                        for eff in effects:
                            if eff.startswith("mem:"):
                                dest = int(eff[4:])
                                if dest < len(arg_vars):
                                    tm(arg_vars[dest])
                continue

            # Conservative fallback for truly unknown calls
            if ret_var:
                tv(ret_var)
            for v, sz in zip(arg_vars, arg_sizes):
                if sz == 8 and is_v(v):
                    tm(v)

        # ── STORE ─────────────────────────────────────────────────────
        elif op_type == "STORE":
            inputs = op.get("inputs") or []
            if len(inputs) >= 3:
                addr_v  = _vn(inputs[1])
                val_v   = _vn(inputs[2])
            elif len(inputs) == 2:
                addr_v  = _vn(inputs[0])
                val_v   = _vn(inputs[1])
            else:
                continue
            if is_v(val_v):
                tm(addr_v)

        # ── LOAD ──────────────────────────────────────────────────────
        elif op_type == "LOAD":
            inputs    = op.get("inputs") or []
            out       = op.get("output")
            out_var   = _vn(out) if out else ""
            addr_v    = _vn(inputs[1]) if len(inputs) >= 2 else _vn(inputs[0]) if inputs else ""
            if out_var and (is_m(addr_v) or is_v(addr_v)):
                tv(out_var)

        # ── Arithmetic / data-flow ────────────────────────────────────
        elif op_type in _ARITH_OPS:
            out     = op.get("output")
            out_var = _vn(out) if out else ""
            if not out_var:
                continue
            inputs = op.get("inputs") or []
            if any(is_v(_vn(i)) for i in inputs if isinstance(i, dict)):
                tv(out_var)

    return (
        {v for v, t in var_taint.items() if t},
        {v for v, t in mem_taint.items() if t},
    )


def _specialize_frees_fields(
    fn_name:         str,
    arg_vars:        list[str],
    fn_ptr_args:     dict[int, str],           # caller arg_idx → fn_name
    addr_to_name:    dict[str, str],
    known_summaries: dict[str, "FuncSummary"],
) -> frozenset[tuple[int, int]]:
    """
    Attempt to discover frees_fields for a call with concrete function-pointer args.

    This handles wrapper patterns like:
        png_safe_execute(image, fn_ptr, arg)
    where ``fn_ptr`` is a known function (e.g. ``png_image_free_function``) that
    frees fields of its first argument.  When png_safe_execute does
    ``CALLIND(fn_ptr, arg)``, we can determine that ``arg``'s fields get freed.

    Algorithm (heuristic — works for single-dispatch wrappers):
      1. Look up the known frees_fields of each concrete fn_ptr argument.
      2. To find which arg of the WRAPPER becomes arg[0] of the concrete fn,
         look in known_summaries for the wrapper's own frees_fields when a
         callee with the given signature is passed.  (We only have the fn_ptr
         argument's own summary here, not the wrapper's internal ops.)
      3. Emit (caller_arg_idx, field_offset) pairs for propagation.

    Returns a frozenset of (caller_arg_idx, field_offset) pairs.
    """
    result: set[tuple[int, int]] = set()

    for caller_fn_ptr_idx, concrete_fn_name in fn_ptr_args.items():
        summary = known_summaries.get(concrete_fn_name)
        if not summary or not summary.frees_fields:
            continue

        # For each (concrete_param_idx, field_offset) in the concrete fn's frees_fields,
        # map concrete_param_idx → which arg of the wrapper was passed as that param.
        #
        # The wrapper calls: CALLIND(fn_ptr, wrapper_arg0, wrapper_arg1, ...)
        # The arg at position concrete_param_idx inside the wrapper's CALLIND
        # maps to some wrapper arg.  We don't have the wrapper's ops here, so
        # we use a heuristic: the wrapper passes its non-fn_ptr arguments in
        # order to the callee.  The most common pattern is that wrapper arg[2]
        # becomes callee arg[0] (fn_ptr is arg[1], actual data is arg[2]).
        #
        # For a 3-arg wrapper like png_safe_execute(image, fn, arg):
        #   - fn_ptr_arg_idx = 1  (fn)
        #   - remaining args in order: arg[0]=image, arg[2]=arg
        #   - callee arg[0] → wrapper arg[2] (first non-fn_ptr arg that follows fn)
        #
        # Build ordered list of non-fn_ptr arg indices.
        non_fn_indices = [i for i in range(len(arg_vars)) if i not in fn_ptr_args]

        for (concrete_param_idx, field_offset) in summary.frees_fields:
            if concrete_param_idx < len(non_fn_indices):
                caller_arg_idx = non_fn_indices[concrete_param_idx]
                result.add((caller_arg_idx, field_offset))

    return frozenset(result)


# ─────────────────────────────────────────────────────────────────────────────
# Summary database
# ─────────────────────────────────────────────────────────────────────────────

class SummaryDatabase:
    """
    Builds and stores per-function taint-transfer summaries.

    Call build() once with all functions to be analyzed.
    Then pass the database to TaintEngine via summary_db=db.

    The database is queried at CALL sites that the PatternMatcher
    cannot resolve (NO_MATCH).  If the callee has a summary, its
    effects are applied precisely rather than using the coarse
    conservative fallback.
    """

    def __init__(self, matcher: "PatternMatcher"):
        self._matcher      = matcher
        self._summaries:   dict[str, FuncSummary] = {}
        self._addr_to_name: dict[str, str] = {}  # populated by build()

    # ── Public API ────────────────────────────────────────────────────

    def build(self, funcs: list[dict]) -> None:
        """
        Compute summaries for all functions in topological call-graph order.
        Callees are always processed before their callers.
        """
        name_to_func         = {f["name"]: f for f in funcs}
        self._addr_to_name   = build_addr_to_name(funcs)
        call_graph           = _build_call_graph(funcs)
        order                = _topological_sort(call_graph)

        cycles = sum(
            1 for name in order
            if name in name_to_func and any(
                name in call_graph.get(callee, [])
                for callee in call_graph.get(name, [])
            )
        )
        if cycles:
            log.debug("Inter-proc: %d functions are in call cycles (summaries may be conservative)", cycles)

        for func_name in order:
            func = name_to_func.get(func_name)
            if func is None:
                continue
            summary = self._compute_summary(func)
            if summary and summary.has_effects():
                self._summaries[func_name] = summary
                log.debug("Summary built: %s", summary)

        # ── External taint propagation pass ──────────────────────────────────
        # After building all summaries, propagate "external taint reaches this
        # arg" through the call graph so the main TaintEngine can seed external
        # taint correctly for functions like png_combine_row that receive
        # file/network data transitively through 3+ call hops.
        #
        # Algorithm:
        #   1. Seed: any function whose RETURN is marked external_source by the
        #      pattern matcher (fread, recv, etc.) — mark callers' arg as external.
        #   2. Propagate: if function F passes its arg N to callee G's arg M, and
        #      G's arg M is externally tainted, then F's arg N is also external.
        #   3. Repeat until no new propagation.

        # Step 1: local analysis — for each function B that calls a known I/O source
        # (fread, png_read_data, recv, TIFFGetField, …), find which LOCAL vars in B
        # receive external data, run mini-taint from those seeds within B, then
        # mark each callee of B that receives a tainted var at arg position ai.
        #
        # This handles the direct case:
        #   png_read_row calls png_read_data(png_ptr, buf, size) → buf is external
        #   png_read_row then calls png_combine_row(png_ptr, buf, display)
        #   → png_combine_row's arg 1 is externally tainted

        externally_tainted_by_func: dict[str, set[int]] = {}  # func → {arg_idx}

        for func_name_b, func_b in name_to_func.items():
            ops_b = func_b.get("ops", [])
            local_ext_seeds: set[str] = set()

            for op in ops_b:
                if op.get("op") not in ("CALL", "CALLIND"):
                    continue
                result = self._matcher.match(op)
                if result is None or not result.is_external_source():
                    continue
                # Collect vars in B that receive external data from this I/O call
                for v in (result.tainted_arg_vars() or []):
                    if v:
                        local_ext_seeds.add(v)
                for v in (result.written_memory_var() or []):
                    if v:
                        local_ext_seeds.add(v)
                if result.return_tainted and result.return_var:
                    local_ext_seeds.add(result.return_var)

            if not local_ext_seeds:
                continue

            # Mark B as having internal external taint (for upward propagation in Step 2)
            if func_name_b not in externally_tainted_by_func:
                externally_tainted_by_func[func_name_b] = set()
            externally_tainted_by_func[func_name_b].add(-1)

            # Mini-taint: propagate from seeds through B's ops
            all_ext_tainted: set[str] = set()
            for seed in local_ext_seeds:
                tainted, _ = _mini_analyze(ops_b, seed, self._matcher, self)
                all_ext_tainted |= tainted

            # Mark callees of B that receive externally tainted vars at specific arg positions
            for op in ops_b:
                if op.get("op") not in ("CALL", "CALLIND"):
                    continue
                inputs = op.get("inputs") or []
                if not inputs:
                    continue
                callee_h = (inputs[0].get("name", "")
                            if isinstance(inputs[0], dict) else str(inputs[0]))
                arg_vars_h = [
                    (inp.get("name", "") if isinstance(inp, dict) else "")
                    for inp in inputs[1:]
                ]
                for ai, av in enumerate(arg_vars_h):
                    if av and av in all_ext_tainted:
                        if callee_h not in externally_tainted_by_func:
                            externally_tainted_by_func[callee_h] = set()
                        externally_tainted_by_func[callee_h].add(ai)

        # Step 2: propagate through callers — if callee has external taint,
        # callers that pass args to it get those args marked as external too
        # Simple fixed-point: repeat until stable
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for func_name_c, func_c in name_to_func.items():
                ops_c = func_c.get("ops", [])
                for op in ops_c:
                    if op.get("op") not in ("CALL", "CALLIND"):
                        continue
                    inputs = op.get("inputs") or []
                    if not inputs:
                        continue
                    callee = (inputs[0].get("name","") if isinstance(inputs[0], dict)
                              else str(inputs[0]))
                    # If callee has external taint, mark this caller too
                    if callee in externally_tainted_by_func:
                        if func_name_c not in externally_tainted_by_func:
                            externally_tainted_by_func[func_name_c] = set()
                            changed = True
                        # Find which args of func_c are passed to callee
                        arg_vars_c = [
                            (inp.get("name","") if isinstance(inp, dict) else "")
                            for inp in inputs[1:]
                        ]
                        for ai, av in enumerate(arg_vars_c):
                            if av and av.startswith("VAR_"):
                                # Check if this var is a param of func_c
                                # (simple heuristic: early VAR with no prior def)
                                if func_name_c not in externally_tainted_by_func:
                                    externally_tainted_by_func[func_name_c] = set()
                                if ai not in externally_tainted_by_func[func_name_c]:
                                    externally_tainted_by_func[func_name_c].add(ai)
                                    changed = True

        # Step 3: update summaries with externally_tainted_args
        updated = 0
        for func_name_u, ext_args in externally_tainted_by_func.items():
            useful_args = frozenset(a for a in ext_args if a >= 0)
            if not useful_args:
                continue
            if func_name_u in self._summaries:
                old = self._summaries[func_name_u]
                self._summaries[func_name_u] = FuncSummary(
                    func_name             = old.func_name,
                    arg_effects           = old.arg_effects,
                    frees_fields          = old.frees_fields,
                    externally_tainted_args = useful_args,
                )
            else:
                self._summaries[func_name_u] = FuncSummary(
                    func_name             = func_name_u,
                    arg_effects           = {},
                    frees_fields          = frozenset(),
                    externally_tainted_args = useful_args,
                )
            updated += 1

        log.debug("External taint propagation: %d functions updated (%d iterations)",
                  updated, iterations)
        # ── End external taint propagation pass ───────────────────────────────

        log.info(
            "Inter-proc summaries: %d / %d functions with propagation effects",
            len(self._summaries), len(funcs),
        )

    def get(self, func_name: str) -> Optional[FuncSummary]:
        """Return the summary for func_name, or None if not found."""
        return self._summaries.get(func_name)

    def apply_at_call_site(
        self,
        summary:    FuncSummary,
        arg_vars:   list[str],
        return_var: Optional[str],
        state,
        fn_name:    str,
    ) -> bool:
        """
        Apply summary taint effects at a call site.

        For each argument that is tainted in state, apply the effects
        recorded in the summary (taint return value, taint memory of args).

        Returns True if any new taint was propagated.
        """
        propagated = False

        for arg_idx, effects in summary.arg_effects.items():
            if arg_idx >= len(arg_vars):
                continue
            src_var = arg_vars[arg_idx]
            if not state.is_tainted(src_var):
                continue

            for effect in effects:
                if effect == "return" and return_var:
                    state.taint_var(
                        return_var,
                        reason   = f"interprocedural:{fn_name}:return",
                        from_var = src_var,
                    )
                    propagated = True

                elif effect.startswith("mem:"):
                    dest_idx = int(effect[4:])
                    if dest_idx < len(arg_vars):
                        state.taint_mem(
                            arg_vars[dest_idx],
                            reason   = f"interprocedural:{fn_name}:mem",
                            from_var = src_var,
                        )
                        propagated = True

        return propagated

    @property
    def summary_count(self) -> int:
        return len(self._summaries)

    @property
    def addr_to_name(self) -> dict[str, str]:
        """Normalized-hex-address → function name.  Available after build()."""
        return self._addr_to_name

    # ── Internal ──────────────────────────────────────────────────────

    def _compute_summary(self, func: dict) -> Optional[FuncSummary]:
        """
        Probe the function once per argument: seed that arg alone and
        observe what becomes tainted in the outputs (return value, memory).

        Uses a param-alias map so that COPY/PTRADD chains are tracked:
        if param[0] is copied into VAR_dst and mem[VAR_dst] gets tainted,
        that correctly records "mem:0" as an effect of whatever arg taints VAR_dst.
        """
        ops    = func.get("ops") or []
        params = _extract_params(ops)
        if not params:
            return None

        ret_var      = _find_return_var(ops)
        param_alias  = _compute_param_aliases(ops, params)  # varname → param index
        arg_effects: dict[int, set[str]] = {}

        for i, param_name in enumerate(params):
            tainted_vars, tainted_mem = _mini_analyze(
                ops, param_name, self._matcher, self
            )

            effects: set[str] = set()

            if ret_var and ret_var in tainted_vars:
                effects.add("return")

            # Check which param's memory got tainted.
            # A varnode in tainted_mem may be a COPY/alias of a param —
            # param_alias resolves this so mem:N is recorded correctly.
            for var in tainted_mem:
                param_idx = param_alias.get(var)
                if param_idx is not None:
                    effects.add(f"mem:{param_idx}")

            if effects:
                arg_effects[i] = effects

        frees_fields = _compute_frees_fields(
            ops, params,
            addr_to_name = getattr(self, "_addr_to_name", None),
        )

        if not arg_effects and not frees_fields:
            return None

        return FuncSummary(
            func_name    = func["name"],
            arg_effects  = {k: frozenset(v) for k, v in arg_effects.items()},
            frees_fields = frees_fields,
        )