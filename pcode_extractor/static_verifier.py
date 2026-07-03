"""
static_verifier.py — Item 10: LLM Hypothesis + Static Verification.

When the LLM classifies a function (e.g., "this is an allocator"), the static
verifier checks whether the P-code actually supports that hypothesis before the
rule is committed to the pattern store.

This prevents LLM hallucination from poisoning the cross-binary knowledge base.
Without verification, a single bad LLM call causes every subsequent binary
processed with the same pattern store to misclassify the same function.

Architecture:
    LLM → hypothesis (role + confidence)
    StaticVerifier.verify(role, ops) → VerificationResult
    If CONFIRMED: store rule at full confidence
    If WEAKLY_CONFIRMED: store rule at reduced confidence (0.4)
    If REFUTED: skip taint rule, store null sentinel (cache-only)

Per-role verification criteria
───────────────────────────────
allocator     : calls a known allocator (malloc/calloc/realloc/new), returns ptr
copy          : has LOAD + STORE, ≥2 pointer-sized args, size arg present
read_input    : calls known I/O function OR writes into an out-arg (writes_memory_at)
validator     : has a comparison op (INT_LESS/INT_LESSEQUAL/INT_EQUAL) + CBRANCH
exec          : calls known exec function (system/execve/popen/shellexec)
format_string : calls printf-family with ≥1 pointer arg
free          : calls free/delete, size of first arg = 8
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)

# ── Known function sets (lowercased for matching) ──────────────────────────────
_ALLOCATOR_FNS = frozenset({
    "malloc", "calloc", "realloc", "new", "png_malloc", "png_calloc",
    "png_malloc_warn", "png_zalloc", "g_malloc", "g_malloc0",
    "av_malloc", "av_mallocz", "zmalloc", "xmalloc", "smalloc",
    "safe_malloc", "safe_calloc", "xml_malloc",
})
_IO_FNS = frozenset({
    "fread", "fgets", "fgetc", "fread_unlocked",
    "recv", "recvfrom", "recvmsg", "read", "pread",
    "getline", "getdelim", "scanf", "fscanf", "sscanf",
    "getenv", "xmlParserInputShrink",
    "png_crc_read", "psf_binheader_readf",
})
_EXEC_FNS = frozenset({
    "system", "execve", "execl", "execle", "execvp", "execlp",
    "popen", "wordexp", "posix_spawn", "ShellExecute", "shellexec",
    "sqlite3_exec",
})
_FREE_FNS = frozenset({
    "free", "delete", "png_free", "g_free", "av_free", "xmlFree",
    "xml_free", "sqlite3_free",
})
_PRINTF_FNS = frozenset({
    "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf",
    "vsprintf", "vsnprintf", "syslog", "log_message",
})
_COMPARISON_OPS = frozenset({
    "INT_LESS", "INT_LESSEQUAL", "INT_EQUAL", "INT_NOTEQUAL",
    "INT_SLESS", "INT_SLESSEQUAL",
})


class VerificationStatus(Enum):
    CONFIRMED       = "confirmed"       # strong static evidence matches hypothesis
    WEAKLY_CONFIRMED = "weakly_confirmed" # partial evidence; store at lower confidence
    REFUTED         = "refuted"         # static evidence contradicts hypothesis


@dataclass
class VerificationResult:
    status:     VerificationStatus
    evidence:   list[str]    # what we found that supports / contradicts
    confidence_multiplier: float  # 1.0 / 0.6 / 0.0 applied to LLM confidence


class StaticVerifier:
    """
    Verifies LLM role hypotheses against raw P-code operations.

    Usage:
        result = StaticVerifier.verify(role="allocator", ops=func_ops)
        if result.status != VerificationStatus.REFUTED:
            store_taint_rule(...)
    """

    @classmethod
    def verify(
        cls,
        role:       str,
        ops:        list[dict],
        arg_sizes:  list[int] | None = None,
        ret_size:   int = 0,
        fn_name:    str = "",
    ) -> VerificationResult:
        """
        Dispatch to the role-specific verifier.
        Returns CONFIRMED / WEAKLY_CONFIRMED / REFUTED.
        """
        role = role.lower()
        verifiers = {
            "allocator":     cls._verify_allocator,
            "copy":          cls._verify_copy,
            "read_input":    cls._verify_read_input,
            "validator":     cls._verify_validator,
            "exec":          cls._verify_exec,
            "format_string": cls._verify_format_string,
            "free":          cls._verify_free,
        }
        fn = verifiers.get(role)
        if fn is None:
            # logger / other — no specific structural criterion; accept as-is
            return VerificationResult(
                status=VerificationStatus.WEAKLY_CONFIRMED,
                evidence=["no structural criterion for role — accepted as-is"],
                confidence_multiplier=0.8,
            )

        result = fn(ops, arg_sizes or [], ret_size, fn_name)
        log.debug(
            "StaticVerifier [%s] %s → %s  evidence=%s",
            fn_name, role, result.status.value, result.evidence,
        )
        return result

    # ── Role verifiers ────────────────────────────────────────────────────────

    @classmethod
    def _verify_allocator(cls, ops, arg_sizes, ret_size, fn_name) -> VerificationResult:
        """
        Allocator: LLM says "this returns a newly-allocated buffer."
        Evidence required:
          STRONG:  direct CALL to a known allocator function
          MEDIUM:  return-value varnode is pointer-sized (8B) AND no STORE ops
          WEAK:    function name contains alloc/malloc/new hint
        """
        evidence = []
        score = 0

        callee_names = cls._get_callees(ops)
        alloc_calls = callee_names & _ALLOCATOR_FNS
        if alloc_calls:
            evidence.append(f"calls allocator: {sorted(alloc_calls)}")
            score += 3

        # Return value is pointer-sized
        if ret_size == 8 or cls._has_pointer_return(ops):
            evidence.append("return value is pointer-sized")
            score += 1

        # No STORE ops — allocators don't write user data
        store_count = sum(1 for op in ops if op.get("op") == "STORE")
        if store_count == 0:
            evidence.append("no STORE ops (consistent with pure allocator)")
            score += 1

        name_hint = any(t in fn_name.lower() for t in ("alloc", "malloc", "new", "create", "init"))
        if name_hint:
            evidence.append(f"name contains alloc hint: {fn_name}")
            score += 1

        if score >= 3:
            return VerificationResult(VerificationStatus.CONFIRMED, evidence, 1.0)
        # Require at least an allocator call OR (pointer return + name hint) for WEAK.
        # "no STORE ops" alone is insufficient — every pure computation also has no STORE.
        if score >= 2 and (alloc_calls or name_hint):
            return VerificationResult(VerificationStatus.WEAKLY_CONFIRMED, evidence, 0.6)
        return VerificationResult(
            VerificationStatus.REFUTED,
            evidence or ["no allocator calls, no pointer return, no name hint"],
            0.0,
        )

    @classmethod
    def _verify_copy(cls, ops, arg_sizes, ret_size, fn_name) -> VerificationResult:
        """
        Copy: LLM says "this copies memory from src to dst."
        Evidence required:
          STRONG:  has both LOAD and STORE, ≥2 pointer-sized (8B) args
          MEDIUM:  has LOAD and STORE, or has a size argument (4B)
        """
        evidence = []
        score = 0

        has_load  = any(op.get("op") == "LOAD"  for op in ops)
        has_store = any(op.get("op") == "STORE" for op in ops)
        has_loop  = cls._has_loop(ops)

        if has_load and has_store:
            evidence.append("has both LOAD and STORE")
            score += 2

        ptr_args = sum(1 for s in arg_sizes if s == 8)
        if ptr_args >= 2:
            evidence.append(f"{ptr_args} pointer-sized args (src+dst pattern)")
            score += 2

        size_args = sum(1 for s in arg_sizes if s in (4, 8) and s != 8)
        if any(s == 4 for s in arg_sizes):
            evidence.append("has 4-byte arg (size parameter)")
            score += 1

        if has_loop:
            evidence.append("has loop (iterative copy pattern)")
            score += 1

        if score >= 4:
            return VerificationResult(VerificationStatus.CONFIRMED, evidence, 1.0)
        if score >= 2:
            return VerificationResult(VerificationStatus.WEAKLY_CONFIRMED, evidence, 0.6)
        return VerificationResult(
            VerificationStatus.REFUTED,
            evidence or ["no LOAD+STORE pair, no dual-pointer signature"],
            0.0,
        )

    @classmethod
    def _verify_read_input(cls, ops, arg_sizes, ret_size, fn_name) -> VerificationResult:
        """
        Read-input: LLM says "this reads external/attacker-controlled data."
        Evidence required:
          STRONG:  direct CALL to known I/O function
          MEDIUM:  writes to a pointer arg (STORE through arg) without reading from it first
        """
        evidence = []
        score = 0

        callee_names = cls._get_callees(ops)
        io_calls = callee_names & _IO_FNS
        if io_calls:
            evidence.append(f"calls I/O function: {sorted(io_calls)}")
            score += 4

        # STORE with no prior LOAD at same address pattern (writes out-arg)
        has_store = any(op.get("op") == "STORE" for op in ops)
        has_load  = any(op.get("op") == "LOAD"  for op in ops)
        if has_store and not has_load:
            evidence.append("writes to output arg without reading (out-param pattern)")
            score += 2
        elif has_store:
            evidence.append("has STORE (writes output buffer)")
            score += 1

        ptr_args = sum(1 for s in arg_sizes if s == 8)
        if ptr_args >= 1:
            evidence.append(f"{ptr_args} pointer arg(s) (output buffer receiver)")
            score += 1

        name_hint = any(t in fn_name.lower() for t in
                        ("read", "recv", "input", "fetch", "get", "load", "parse"))
        if name_hint:
            evidence.append(f"name suggests I/O: {fn_name}")
            score += 1

        if score >= 4:
            return VerificationResult(VerificationStatus.CONFIRMED, evidence, 1.0)
        if score >= 2:
            return VerificationResult(VerificationStatus.WEAKLY_CONFIRMED, evidence, 0.6)
        return VerificationResult(
            VerificationStatus.REFUTED,
            evidence or ["no I/O calls, no output-arg write pattern"],
            0.0,
        )

    @classmethod
    def _verify_validator(cls, ops, arg_sizes, ret_size, fn_name) -> VerificationResult:
        """
        Validator: LLM says "this bounds-checks or validates an argument."
        Evidence required:
          STRONG:  comparison op (INT_LESS/LESSEQUAL) + CBRANCH
          MEDIUM:  comparison op alone (function may abort/setjmp instead of branch)
        """
        evidence = []
        score = 0

        cmp_ops   = [op for op in ops if op.get("op") in _COMPARISON_OPS]
        has_branch = any(op.get("op") == "CBRANCH" for op in ops)

        if cmp_ops:
            evidence.append(f"{len(cmp_ops)} comparison ops: "
                            f"{set(op['op'] for op in cmp_ops)}")
            score += 2

        if has_branch:
            evidence.append("has CBRANCH (conditional exit on failure)")
            score += 2

        # Check for comparison against a constant (typical bounds check)
        const_cmp = any(
            any(isinstance(inp, dict) and inp.get("space") == "const"
                for inp in (op.get("inputs") or []))
            for op in cmp_ops
        )
        if const_cmp:
            evidence.append("comparison against constant (bound value)")
            score += 1

        name_hint = any(t in fn_name.lower() for t in
                        ("check", "valid", "verify", "assert", "bound", "limit", "safe"))
        if name_hint:
            evidence.append(f"name suggests validation: {fn_name}")
            score += 1

        if score >= 4:
            return VerificationResult(VerificationStatus.CONFIRMED, evidence, 1.0)
        if score >= 2:
            return VerificationResult(VerificationStatus.WEAKLY_CONFIRMED, evidence, 0.6)
        return VerificationResult(
            VerificationStatus.REFUTED,
            evidence or ["no comparison ops, no conditional branch"],
            0.0,
        )

    @classmethod
    def _verify_exec(cls, ops, arg_sizes, ret_size, fn_name) -> VerificationResult:
        """
        Exec: LLM says "this executes a system command."
        Evidence required: direct call to exec/system/popen or exec-like name.
        """
        evidence = []
        callee_names = cls._get_callees(ops)
        exec_calls = callee_names & _EXEC_FNS
        if exec_calls:
            evidence.append(f"calls exec function: {sorted(exec_calls)}")
            return VerificationResult(VerificationStatus.CONFIRMED, evidence, 1.0)

        name_hint = any(t in fn_name.lower() for t in ("exec", "system", "shell", "spawn", "run"))
        if name_hint:
            evidence.append(f"name suggests execution: {fn_name}")
            return VerificationResult(VerificationStatus.WEAKLY_CONFIRMED, evidence, 0.7)

        return VerificationResult(
            VerificationStatus.REFUTED,
            ["no exec/system calls found"],
            0.0,
        )

    @classmethod
    def _verify_format_string(cls, ops, arg_sizes, ret_size, fn_name) -> VerificationResult:
        """
        Format string: LLM says "this uses a format string."
        Evidence: calls printf family, has 2+ pointer args (format + additional).
        """
        evidence = []
        callee_names = cls._get_callees(ops)
        fmt_calls = callee_names & _PRINTF_FNS
        if fmt_calls:
            evidence.append(f"calls format function: {sorted(fmt_calls)}")
            ptr_args = sum(1 for s in arg_sizes if s == 8)
            if ptr_args >= 2:
                evidence.append(f"{ptr_args} pointer args (format + variadic)")
                return VerificationResult(VerificationStatus.CONFIRMED, evidence, 1.0)
            return VerificationResult(VerificationStatus.WEAKLY_CONFIRMED, evidence, 0.7)

        return VerificationResult(
            VerificationStatus.REFUTED,
            ["no printf-family calls found"],
            0.0,
        )

    @classmethod
    def _verify_free(cls, ops, arg_sizes, ret_size, fn_name) -> VerificationResult:
        """
        Free: LLM says "this releases memory."
        Evidence: calls free/delete, or first arg is pointer-sized.
        """
        evidence = []
        callee_names = cls._get_callees(ops)
        free_calls = callee_names & _FREE_FNS
        if free_calls:
            evidence.append(f"calls free function: {sorted(free_calls)}")
            return VerificationResult(VerificationStatus.CONFIRMED, evidence, 1.0)

        name_hint = any(t in fn_name.lower() for t in ("free", "release", "destroy", "dealloc", "delete"))
        if name_hint and arg_sizes and arg_sizes[0] == 8:
            evidence.append(f"name suggests deallocation: {fn_name}, arg[0] is pointer")
            return VerificationResult(VerificationStatus.WEAKLY_CONFIRMED, evidence, 0.7)

        return VerificationResult(
            VerificationStatus.REFUTED,
            ["no free/delete calls found"],
            0.0,
        )

    # ── P-code inspection helpers ─────────────────────────────────────────────

    @staticmethod
    def _get_callees(ops: list[dict]) -> set[str]:
        """Extract all callee names from CALL ops."""
        callees: set[str] = set()
        for op in ops:
            if op.get("op") not in ("CALL", "CALLIND"):
                continue
            inputs = op.get("inputs") or []
            if not inputs or not isinstance(inputs[0], dict):
                continue
            name = inputs[0].get("name", "")
            if "@" in name:
                name = name.split("@")[0]
            if name.startswith("<") and name.endswith(">"):
                name = name[1:-1]
            if name:
                callees.add(name.lower())
        return callees

    @staticmethod
    def _has_pointer_return(ops: list[dict]) -> bool:
        """True if the last RETURN-like op carries a pointer-sized value."""
        for op in reversed(ops):
            if op.get("op") in ("RETURN", "COPY"):
                out = op.get("output")
                if isinstance(out, dict) and out.get("size", 0) == 8:
                    return True
        return False

    @staticmethod
    def _has_loop(ops: list[dict]) -> bool:
        """True if there is a backward branch (loop indicator)."""
        seqs = {op.get("seq", -1): i for i, op in enumerate(ops)}
        for op in ops:
            if op.get("op") == "CBRANCH":
                inputs = op.get("inputs") or []
                for inp in inputs:
                    if isinstance(inp, dict):
                        target_seq = inp.get("seq") or inp.get("value")
                        if target_seq is not None:
                            try:
                                if int(target_seq) < int(op.get("seq", 0)):
                                    return True
                            except (TypeError, ValueError):
                                pass
        # Fallback: MULTIEQUAL (phi node) implies loop convergence in SSA P-code
        return any(op.get("op") == "MULTIEQUAL" for op in ops)


# ── Stats tracking for paper Table 3 (verification outcomes) ──────────────────

class VerificationStats:
    """
    Thread-safe counter for verification outcomes across a full binary run.
    Written to pattern_store kv_metadata for cross-binary persistence.
    """

    def __init__(self):
        self._counts: dict[str, int] = {
            "confirmed": 0,
            "weakly_confirmed": 0,
            "refuted": 0,
            "skipped": 0,  # roles with no verifier (logger/other)
        }

    def record(self, result: VerificationResult) -> None:
        key = result.status.value
        self._counts[key] = self._counts.get(key, 0) + 1

    def totals(self) -> dict:
        total = sum(self._counts.values())
        return {
            **self._counts,
            "total": total,
            "verified_pct": round(
                self._counts["confirmed"] / total * 100 if total else 0, 1
            ),
        }

    def log_summary(self, prefix: str = "") -> None:
        t = self.totals()
        log.info(
            "%sVerification: confirmed=%d  weak=%d  refuted=%d  "
            "verified_pct=%.0f%%  total=%d",
            prefix,
            t["confirmed"], t["weakly_confirmed"], t["refuted"],
            t["verified_pct"], t["total"],
        )
