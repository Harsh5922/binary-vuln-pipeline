"""
stage3_backward_slicer.py  —  Backward Slicing (Research Contribution)
=======================================================================
Complements Stage 3B's forward taint with backward slicing from suspect sinks.

Motivation (user proposal):
  "I would make it Bi-directional. Example:
   malloc() → Candidate → Walk backward → Can attacker control this? → YES → Candidate.
   Many commercial analyzers already combine forward and backward slicing."

Algorithm:
  P-code SSA has the property that each variable has EXACTLY ONE defining op.
  So backward slicing is deterministic:

    slice(target_var):
      defining_op = op_that_outputs(target_var)
      if no defining_op → root (function parameter) → attacker-reachable
      if CALL → check source confidence of the callee
      else    → recurse on all non-constant inputs

  This continues until we reach:
    - A parameter (no defining op) → potentially attacker-controlled (conf 0.55)
    - A known external-source CALL → use its Source Confidence
    - Max depth exceeded           → unknown (conf 0.0)

Usage
-----
    from stage3_backward_slicer import BackwardSlicer
    from stage3_source_analysis  import SourceAnalyzer

    slicer = BackwardSlicer(SourceAnalyzer())
    result = slicer.slice(target_var="VAR_5", ops=func_ops)
    if result.max_conf >= 0.50:
        # attacker can likely influence this value — emit candidate
        ...

Result fields:
    target_var     — the variable we sliced backward from
    root_sources   — param names / "call:fn_name" at slice boundary
    max_conf       — highest source confidence reached (key decision signal)
    path_vars      — all variables in the backward slice
    ops_in_slice   — ops traversed
    external_fns   — external source functions encountered in slice
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SliceResult:
    """Result of a backward data-flow slice."""
    target_var:   str
    root_sources: list[str]   # param vars / "call:fn_name" at boundary
    max_conf:     float       # highest source confidence found in slice
    path_vars:    list[str]   # all variables visited (for audit)
    ops_in_slice: int         # number of ops traversed
    external_fns: list[str]   # external source functions encountered


# Parameters are potentially attacker-controlled when the binary is a
# library or daemon. We assign 0.55 as a "moderate suspicion" level —
# enough to trigger backward-enhanced recall but not auto-confirm.
_PARAM_CONFIDENCE = 0.55


class BackwardSlicer:
    """
    Backward SSA data-flow slicer.

    Takes a `SourceAnalyzer` to look up confidence when a CALL is encountered
    at the slice boundary.

    Thread-safe (stateless instance, all state in slice() local vars).
    """

    def __init__(self, source_analyzer, max_depth: int = 20):
        """
        Parameters
        ----------
        source_analyzer : SourceAnalyzer instance
        max_depth       : maximum backward hops before giving up
        """
        self._src  = source_analyzer
        self._max  = max_depth

    def slice(self, target_var: str, ops: list[dict]) -> SliceResult:
        """
        Compute backward data-flow slice from `target_var`.

        Returns SliceResult with max source confidence reachable from any
        input that defines the value of `target_var`.
        """
        if not target_var or not ops:
            return SliceResult(target_var, [], 0.0, [], 0, [])

        # Build SSA def-map: output_var_name → op
        definer: dict[str, dict] = {}
        for op in ops:
            out = op.get("output")
            if out and isinstance(out, dict):
                oname = out.get("name", "")
                if oname:
                    definer[oname] = op

        # BFS backward
        visited:      set[str]             = set()
        queue:        list[tuple[str, int]] = [(target_var, 0)]
        root_sources: list[str]            = []
        path_vars:    list[str]            = []
        external_fns: list[str]            = []
        max_conf      = 0.0
        ops_in_slice  = 0

        while queue:
            var, depth = queue.pop(0)
            if var in visited:
                continue
            visited.add(var)
            path_vars.append(var)

            defining_op = definer.get(var)

            # ── No defining op → function parameter ──────────────────────
            if defining_op is None:
                root_sources.append(var)
                max_conf = max(max_conf, _PARAM_CONFIDENCE)
                continue

            if depth >= self._max:
                continue   # depth cap — stop expanding this branch

            op_type = defining_op.get("op", "")
            ops_in_slice += 1

            # ── CALL / CALLIND → check source confidence ──────────────────
            if op_type in ("CALL", "CALLIND"):
                inputs  = defining_op.get("inputs") or []
                fn_name = inputs[0].get("name", "") if inputs else ""
                fn_conf = self._src.get_confidence(fn_name)
                if fn_conf > 0.0:
                    external_fns.append(fn_name)
                    max_conf = max(max_conf, fn_conf)
                    root_sources.append(f"call:{fn_name}")
                else:
                    # Unknown external call — recurse through its args
                    for inp in inputs[1:]:
                        if isinstance(inp, dict):
                            iname = inp.get("name", "")
                            if iname and not _is_const(iname):
                                queue.append((iname, depth + 1))
                continue

            # ── Regular op → recurse on non-constant inputs ───────────────
            for inp in (defining_op.get("inputs") or []):
                if isinstance(inp, dict):
                    iname = inp.get("name", "")
                    if iname and not _is_const(iname):
                        queue.append((iname, depth + 1))

        return SliceResult(
            target_var   = target_var,
            root_sources = root_sources,
            max_conf     = max_conf,
            path_vars    = path_vars,
            ops_in_slice = ops_in_slice,
            external_fns = external_fns,
        )


def _is_const(name: str) -> bool:
    """Return True if the P-code variable name represents a constant."""
    return name.startswith("const(") or name.startswith("ram(")
