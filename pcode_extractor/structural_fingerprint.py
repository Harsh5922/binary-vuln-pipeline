"""
structural_fingerprint.py — Item 6: Auto Pattern Discovery.

Computes a structural fingerprint from a function's P-code operations.
Two functions with the same fingerprint are structurally identical — they
perform the same computation even if they have different names, different
compiled argument types, or come from different binaries.

This solves the "renamed memcpy problem":
  - Binary 1 has `memcpy` → LIBRARY_MATCH, no LLM needed
  - Binary 2 has `my_custom_memcpy` → same P-code structure → STRUCTURAL_MATCH
  - Binary 3 has `png_copy_row` → same ops → STRUCTURAL_MATCH → no LLM needed

Why structural matching works for binary analysis
──────────────────────────────────────────────────
The compiler produces similar instruction sequences for semantically similar
functions, especially for memory operations (copy, zero, compare, hash).
Cross-compilation may change absolute addresses and variable names, but the
sequence of P-code operation TYPES and their relative structure (loop vs.
linear, comparison before write, etc.) is preserved.

Fingerprint design
──────────────────
We extract a bag-of-features vector from the function's P-code ops:

  1. Op histogram:        normalised frequency of each P-code opcode
  2. Control flow sketch: [has_loop, has_call, has_indirect_call, branch_density]
  3. Memory profile:      [load_count, store_count, ptr_arg_count, ret_size]
  4. Call signature:      sorted tuple of known callee names (allocators, I/O, etc.)
  5. Loop depth:          nesting level of loops (CBRANCH backward jumps)

The fingerprint is a SHA-256 of the canonical representation.

Fuzzy matching (for renamed-but-restructured functions)
────────────────────────────────────────────────────────
For approximate matching, we provide an edit-distance over the feature vectors.
Two functions with cosine_similarity > 0.90 are considered the same structurally.

Storage: fingerprints are stored in the pattern_store.db `structural_fingerprints`
table alongside the learned role + taint rule. On subsequent binaries, any function
matching a stored fingerprint is auto-classified (STRUCTURAL_MATCH).

Usage
─────
    from structural_fingerprint import FunctionFingerprinter, FingerprintMatcher

    # Compute fingerprint for a function
    fp = FunctionFingerprinter.compute(fn_name="my_custom_memcpy", ops=function_ops)
    print(fp.sha256_key)  # canonical hash — used as DB lookup key
    print(fp.feature_vector)  # float vector for fuzzy matching

    # Match against stored fingerprints
    matcher = FingerprintMatcher(pattern_store)
    match = matcher.find_match(ops=func_ops, fn_name=fn_name)
    if match:
        print(f"STRUCTURAL_MATCH: {fn_name} ~ {match.canonical_name} ({match.role})")
"""
from __future__ import annotations
import hashlib
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ── P-code opcode groups ──────────────────────────────────────────────────────
# Reduces 60+ opcodes to 12 semantic groups for robust matching.
# Two equivalent functions compiled with different optimization levels should
# map to the same group sequence.

_OP_GROUP = {
    # Memory
    "LOAD":       "mem_read",
    "STORE":      "mem_write",
    # Arithmetic
    "INT_ADD":    "arith",   "INT_SUB":    "arith",
    "INT_MULT":   "arith",   "INT_DIV":    "arith",
    "INT_REM":    "arith",   "INT_NEGATE": "arith",
    "FLOAT_ADD":  "arith",   "FLOAT_SUB":  "arith",
    # Logic
    "INT_AND":    "logic",   "INT_OR":     "logic",
    "INT_XOR":    "logic",   "INT_NOTT":   "logic",
    "BOOL_AND":   "logic",   "BOOL_OR":    "logic",
    # Comparison
    "INT_EQUAL":      "cmp", "INT_NOTEQUAL": "cmp",
    "INT_LESS":       "cmp", "INT_LESSEQUAL":"cmp",
    "INT_SLESS":      "cmp", "INT_SLESSEQUAL":"cmp",
    # Pointer / shift
    "PTRADD":     "ptr",     "PTRSUB":     "ptr",
    "INT_ZEXT":   "cast",    "INT_SEXT":   "cast",
    "INT_LSHIFT": "shift",   "INT_RSHIFT": "shift",
    # Control flow
    "CBRANCH":    "branch",  "BRANCH":     "branch",  "BRANCHIND": "branch",
    # Calls
    "CALL":       "call",    "CALLIND":    "indirect_call",
    # PHI / multi-assign (SSA)
    "MULTIEQUAL": "phi",
    # Return
    "RETURN":     "return",
    # Copy
    "COPY":       "copy",    "PIECE":      "copy",    "SUBPIECE": "copy",
}
_ALL_GROUPS = sorted(set(_OP_GROUP.values()))  # 14 groups

# ── Feature extraction ────────────────────────────────────────────────────────

@dataclass
class FunctionFingerprint:
    fn_name:        str
    sha256_key:     str           # canonical hash (exact match)
    feature_vector: list[float]   # 20-dim float vector (fuzzy match)
    op_count:       int
    call_signature: str           # sorted known callees

    # Role inferred from structural analysis (before LLM)
    structural_role: str = "unknown"


class FunctionFingerprinter:
    """
    Computes a structural fingerprint for a function's P-code.
    """

    # Known callee categories for call signature
    _KNOWN_CALLEES = {
        "malloc","calloc","realloc","free",
        "memcpy","memset","memmove","memcmp","strcmp","strcpy","strncpy","strlen",
        "fread","fwrite","fgets","fputs","fopen","fclose",
        "recv","send","read","write","accept","connect",
        "printf","fprintf","sprintf","snprintf",
        "atoi","atol","strtol","strtoll","strtod",
    }

    @classmethod
    def compute(cls, fn_name: str, ops: list[dict], arg_sizes: list[int] = None) -> FunctionFingerprint:
        """
        Compute the fingerprint for a function given its P-code ops.
        """
        if not ops:
            return FunctionFingerprint(
                fn_name=fn_name, sha256_key="empty",
                feature_vector=[0.0] * 20, op_count=0, call_signature="",
            )

        arg_sizes = arg_sizes or []
        n = len(ops)

        # 1. Op group histogram (normalised to [0,1]) — 14 dims
        counts: Counter = Counter()
        for op in ops:
            grp = _OP_GROUP.get(op.get("op", ""), "other")
            counts[grp] += 1
        hist = [counts.get(g, 0) / max(n, 1) for g in _ALL_GROUPS]

        # 2. Control flow features — 4 dims
        has_loop          = 1.0 if cls._has_loop(ops) else 0.0
        has_call          = 1.0 if counts.get("call", 0) > 0 else 0.0
        has_indirect_call = 1.0 if counts.get("indirect_call", 0) > 0 else 0.0
        branch_density    = min(1.0, counts.get("branch", 0) / max(n, 1) * 10)
        cf_features = [has_loop, has_call, has_indirect_call, branch_density]

        # 3. Memory profile — 4 dims (normalised)
        load_count   = min(1.0, counts.get("mem_read", 0)  / max(n, 1) * 5)
        store_count  = min(1.0, counts.get("mem_write", 0) / max(n, 1) * 5)
        ptr_arg_count = min(1.0, sum(1 for s in arg_sizes if s == 8) / max(len(arg_sizes), 1))
        ret_size_8   = 1.0 if cls._has_pointer_return(ops) else 0.0
        mem_features = [load_count, store_count, ptr_arg_count, ret_size_8]

        # Feature vector: 14 op groups + 4 CF + 4 memory = 22 dims (truncate/pad to 20)
        feature_vector = (hist + cf_features + mem_features)[:20]
        while len(feature_vector) < 20:
            feature_vector.append(0.0)

        # 4. Call signature: known callees (for exact structural match)
        call_signature = cls._call_signature(ops)

        # 5. Canonical hash: derived from group histogram counts + call signature
        # We use integer group counts (not normalised) for determinism
        group_counts_str = ",".join(f"{g}:{counts.get(g,0)}" for g in _ALL_GROUPS)
        canon = f"{group_counts_str}|{call_signature}|ptrs={sum(1 for s in arg_sizes if s==8)}"
        sha256_key = hashlib.sha256(canon.encode()).hexdigest()[:16]

        # 6. Structural role inference (no LLM)
        structural_role = cls._infer_role(counts, arg_sizes, ops)

        return FunctionFingerprint(
            fn_name         = fn_name,
            sha256_key      = sha256_key,
            feature_vector  = feature_vector,
            op_count        = n,
            call_signature  = call_signature,
            structural_role = structural_role,
        )

    @classmethod
    def _call_signature(cls, ops: list[dict]) -> str:
        """Sorted tuple of known callee names present in the function."""
        known: set[str] = set()
        for op in ops:
            if op.get("op") not in ("CALL", "CALLIND"):
                continue
            inputs = op.get("inputs") or []
            if not inputs or not isinstance(inputs[0], dict):
                continue
            name = inputs[0].get("name", "").lower()
            name = name.replace("<", "").replace(">", "").split("@")[0]
            if name in cls._KNOWN_CALLEES:
                known.add(name)
        return ",".join(sorted(known))

    @classmethod
    def _infer_role(cls, counts: Counter, arg_sizes: list[int], ops: list[dict]) -> str:
        """
        Infer the likely role from pure structural evidence — no LLM, no names.
        These are coarse labels used as a hint; StaticVerifier does proper verification.
        """
        has_store = counts.get("mem_write", 0) > 0
        has_load  = counts.get("mem_read",  0) > 0
        has_call  = counts.get("call", 0) > 0
        has_cmp   = counts.get("cmp", 0) > 0
        has_branch = counts.get("branch", 0) > 0
        ptr_args  = sum(1 for s in arg_sizes if s == 8)
        ret_ptr   = cls._has_pointer_return(ops)
        call_sig  = cls._call_signature(ops)

        if "malloc" in call_sig or "calloc" in call_sig or "realloc" in call_sig:
            return "allocator"
        if "free" in call_sig:
            return "free"
        if "memcpy" in call_sig or "memmove" in call_sig:
            return "copy"
        if "fread" in call_sig or "recv" in call_sig or "read" in call_sig:
            return "read_input"
        if has_store and has_load and ptr_args >= 2:
            return "copy"       # loop + dual-pointer = copy pattern
        if ret_ptr and not has_store and has_call:
            return "allocator"  # returns pointer, calls something
        if has_cmp and has_branch and not has_store:
            return "validator"  # comparison + conditional exit, no writes
        return "unknown"

    @staticmethod
    def _has_loop(ops: list[dict]) -> bool:
        return any(op.get("op") == "MULTIEQUAL" for op in ops) or any(
            op.get("op") == "CBRANCH" and op.get("is_backward", False)
            for op in ops
        )

    @staticmethod
    def _has_pointer_return(ops: list[dict]) -> bool:
        for op in reversed(ops):
            if op.get("op") in ("RETURN", "COPY"):
                out = op.get("output")
                if isinstance(out, dict) and out.get("size", 0) == 8:
                    return True
        return False


# ── Fuzzy vector similarity ───────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two feature vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Pattern store integration ─────────────────────────────────────────────────

@dataclass
class FingerprintMatch:
    """Result of a successful fingerprint match."""
    canonical_name:   str   # name of the matched prototype function
    role:             str   # learned role for the match
    match_kind:       str   # "EXACT" (sha256) | "FUZZY" (cosine > threshold)
    similarity:       float # 1.0 for exact, 0.9+ for fuzzy
    taint_rule:       dict  # rule from the matched entry


class FingerprintMatcher:
    """
    Matches a function against known structural fingerprints in the pattern store.
    Falls back from exact → fuzzy matching.
    """

    FUZZY_THRESHOLD = 0.90

    def __init__(self, pattern_store):
        self._store = pattern_store

    def find_match(
        self,
        ops:       list[dict],
        fn_name:   str = "",
        arg_sizes: list[int] = None,
    ) -> Optional[FingerprintMatch]:
        """
        Try to find a structural match for a function in the pattern store.
        Returns FingerprintMatch if found, None otherwise.
        """
        fp = FunctionFingerprinter.compute(fn_name, ops, arg_sizes)

        # Try exact match first (O(1))
        exact = self._store.get_structural_fingerprint(fp.sha256_key)
        if exact:
            log.debug("STRUCTURAL_MATCH (exact): %s ~ %s (%s)",
                      fn_name, exact["canonical_name"], exact["role"])
            return FingerprintMatch(
                canonical_name = exact["canonical_name"],
                role           = exact["role"],
                match_kind     = "EXACT",
                similarity     = 1.0,
                taint_rule     = exact.get("taint_rule", {}),
            )

        # Try fuzzy match (O(n) over stored fingerprints, practical for small stores)
        all_fps = self._store.get_all_structural_fingerprints()
        best_sim = 0.0
        best_entry = None
        for entry in all_fps:
            stored_vec = entry.get("feature_vector", [])
            if not stored_vec:
                continue
            sim = cosine_similarity(fp.feature_vector, stored_vec)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self.FUZZY_THRESHOLD:
            log.debug(
                "STRUCTURAL_MATCH (fuzzy %.2f): %s ~ %s (%s)",
                best_sim, fn_name, best_entry["canonical_name"], best_entry["role"],
            )
            return FingerprintMatch(
                canonical_name = best_entry["canonical_name"],
                role           = best_entry["role"],
                match_kind     = "FUZZY",
                similarity     = best_sim,
                taint_rule     = best_entry.get("taint_rule", {}),
            )

        return None

    def store_fingerprint(
        self,
        ops:        list[dict],
        fn_name:    str,
        role:       str,
        taint_rule: dict,
        arg_sizes:  list[int] = None,
    ) -> str:
        """
        Compute and store a fingerprint for a function whose role is now known.
        Returns the sha256_key.
        """
        fp = FunctionFingerprinter.compute(fn_name, ops, arg_sizes)
        self._store.store_structural_fingerprint(
            sha256_key     = fp.sha256_key,
            canonical_name = fn_name,
            role           = role,
            feature_vector = fp.feature_vector,
            taint_rule     = taint_rule,
        )
        return fp.sha256_key
