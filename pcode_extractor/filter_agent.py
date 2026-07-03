"""
filter_agent.py — Stage 2: Multi-Dimensional Function Ranking

Architecture (6 sub-stages):
  2.1  Fast Feature Extraction   — pure static, no LLM, extract 25+ granular signals
  2.2  Semantic Role Inference   — name heuristics + structural patterns + pattern_store
  2.3  Multi-Dimensional Scoring — five independent [0,1] scores per function
  2.4  Graph Context Propagation — spread scores through the call graph (2-hop)
  2.5  Diversity Selection       — category-quota selection (parser/validator/arith/…)
  2.6  Adaptive Neighborhood     — include callers/callees of high-scoring functions

Public API (unchanged from previous version — pipeline.py compatibility):
  agent = FunctionFilterAgent.from_jsonl(path, budget=300, min_score=0.15)
  agent.save_ranked(out_path)
  agent.stats()   → {"kept":…, "total":…, "reduction_pct":…, …}

Scoring formula:
  FinalScore = 0.20·Exposure + 0.20·Validation + 0.20·Arithmetic
             + 0.15·Graph   + 0.10·Semantic   + 0.05·PatternStore
             + 0.05·Memory  + 0.05·DangerousCalls
"""

from __future__ import annotations

import enum
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Optional pattern_matcher integration ─────────────────────────────────────
try:
    from pattern_store   import PatternStore
    from pattern_matcher import PatternMatcher, FunctionRanker, MatchKind
    _PATTERN_MATCHER_AVAILABLE = True
except ImportError:
    _PATTERN_MATCHER_AVAILABLE = False
    log.debug("pattern_matcher not found — heuristic-only scoring")

# ── Canonical dangerous-import list (shared with extractor.py) ───────────────
try:
    from extractor import DANGEROUS_IMPORTS
except ImportError:
    DANGEROUS_IMPORTS = frozenset({
        "strcpy", "strncpy", "strcat", "strncat",
        "sprintf", "vsprintf", "snprintf", "vsnprintf", "gets",
        "memcpy", "memmove", "memset", "mempcpy", "wmemcpy",
        "recv", "recvfrom", "read", "fread", "fgets",
        "scanf", "sscanf", "fscanf", "getenv",
        "ReadFile", "WSARecv",
        "malloc", "calloc", "realloc", "free", "alloca",
        "system", "popen", "execl", "execle", "execlp",
        "execv", "execve", "execvp",
        "printf", "fprintf", "wprintf",
        "OPENSSL_malloc", "OPENSSL_realloc", "OPENSSL_zalloc",
        "CRYPTO_malloc", "CRYPTO_realloc", "OPENSSL_memdup",
        "BUF_MEM_grow", "BIO_read", "SSL_read", "EVP_DecodeUpdate",
        "sqlite3Malloc", "sqlite3MallocZero", "sqlite3Realloc",
        "sqlite3DbMalloc", "sqlite3DbMallocRaw", "sqlite3DbRealloc",
        "sqlite3_malloc", "sqlite3_malloc64", "sqlite3_realloc",
        "sqlite3StrAccumEnlarge",
        "luaM_realloc_", "luaM_malloc_", "luaL_prepbuffsize",
        "_TIFFmalloc", "_TIFFcalloc", "_TIFFrealloc",
        "_TIFFCheckMalloc", "_TIFFCheckRealloc",
        "TIFFGetField", "TIFFReadRawTile", "TIFFReadRawStrip",
        "xmlMalloc", "xmlMallocAtomic", "xmlRealloc",
        "xmlStrdup", "xmlStrndup", "xmlBufGrow", "xmlBufAdd",
        "xmlGetProp", "xmlNodeGetContent",
        "Stream::getChar",
    })

# Input sources: functions that bring attacker data in
TAINT_SOURCE_IMPORTS: frozenset[str] = frozenset({
    "recv", "recvfrom", "read", "fread", "fgets",
    "scanf", "sscanf", "fscanf", "getenv",
    "ReadFile", "WSARecv", "gets",
})

# Pure dangerous sinks (not allocators — those need context to judge)
TAINT_SINK_IMPORTS: frozenset[str] = frozenset({
    "strcpy", "strcat", "sprintf", "vsprintf", "gets",
    "memcpy", "memmove", "mempcpy", "wmemcpy",
    "strncat", "strncpy", "snprintf", "vsnprintf",
    "system", "popen", "execl", "execle", "execlp",
    "execv", "execve", "execvp",
    "printf", "fprintf", "wprintf",
})


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.1 — Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionInfo:
    """Mirror of extractor.FunctionInfo — populated from JSONL."""
    name:           str
    entry_addr:     str
    prototype:      str
    ops:            list[dict]
    call_sites:     list[str]
    flags:          dict[str, bool]
    size_bytes:     int
    decompile_time: float = 0.0


@dataclass
class FunctionFeatures:
    """
    Stage 2.1 output: granular static-analysis features per function.
    All counts come directly from P-code ops — no heuristics, no LLM.
    """
    # Structural
    instruction_count: int = 0
    basic_block_count: int = 0   # approximated as CBRANCH/BRANCH count + 1
    cyclomatic:        int = 0   # CBRANCH count + 1 (McCabe approximation)

    # Memory
    load_count:  int  = 0
    store_count: int  = 0
    heap_alloc:  bool = False   # calls malloc/realloc family
    stack_alloc: bool = False   # large constant written to STORE

    # Arithmetic — split by type (each tells a different story)
    int_mult:    int = 0   # width * height, size * count → overflow risk
    int_add:     int = 0
    int_sub:     int = 0
    shift_count: int = 0   # INT_LEFT / INT_RIGHT / INT_SRIGHT → scale factor
    cast_count:  int = 0   # INT_ZEXT / INT_SEXT / INT_TRUNC → truncation risk
    ptradd:      int = 0   # PTRADD → pointer arithmetic
    ptrsub:      int = 0   # PTRSUB → pointer difference

    # Validation — first-class citizens
    int_less:       int = 0   # INT_LESS / INT_LESSEQUAL
    int_equal:      int = 0   # INT_EQUAL / INT_NOTEQUAL
    cbranch:        int = 0   # conditional branches

    # Input detection
    input_calls:    list[str] = field(default_factory=list)   # taint source calls

    # Calls
    total_calls:    int = 0
    indirect_calls: int = 0
    sink_calls:     list[str] = field(default_factory=list)   # direct sink calls

    # Unchecked arithmetic pattern: arith result flows to STORE without CBRANCH check
    unchecked_arith_store: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.2 — Semantic Role
# ─────────────────────────────────────────────────────────────────────────────

class SemanticRole(enum.Enum):
    PARSER       = "parser"
    VALIDATOR    = "validator"
    DECODER      = "decoder"
    ALLOCATOR    = "allocator"
    WRAPPER      = "wrapper"
    ITERATOR     = "iterator"
    ERROR_HANDLER= "error_handler"
    SERIALIZER   = "serializer"
    CHECKSUM     = "checksum"
    STATE_MACHINE= "state_machine"
    MATH         = "math"
    DISPATCHER   = "dispatcher"
    UNKNOWN      = "unknown"


@dataclass
class RoleAssignment:
    role:       SemanticRole
    confidence: float          # [0, 1]
    source:     str            # "name", "structural", "pattern_store"


# Name-pattern → role mapping (order matters: more specific patterns first)
_ROLE_NAME_PATTERNS: list[tuple[SemanticRole, tuple[str, ...]]] = [
    (SemanticRole.ERROR_HANDLER, (
        "error", "err_", "_err", "warning", "warn_", "fatal",
        "abort_", "_abort", "panic", "cleanup", "_cleanup", "exception",
    )),
    (SemanticRole.CHECKSUM, (
        "crc", "hash", "checksum", "digest", "hmac",
        "sha", "md5", "adler", "fletcher",
    )),
    (SemanticRole.DECODER, (
        "decode", "decrypt", "decompress", "inflate", "unpack",
        "base64", "unescape", "urldecode", "deobfuscat",
    )),
    (SemanticRole.VALIDATOR, (
        "check_", "_check", "validate", "verify", "assert_",
        "guard", "bound", "limit_", "_limit", "sanity",
        "ensure", "is_valid", "clamp", "is_safe",
    )),
    (SemanticRole.PARSER, (
        "parse", "parser", "lex", "scan_", "_scan", "tokenize",
        "_parse", "read_", "_read", "load_", "_load",
        "select", "query", "stmt", "handle_", "_handle",
        "process_", "_process",
    )),
    (SemanticRole.SERIALIZER, (
        "write_", "_write", "encode", "serialize", "pack_",
        "_pack", "emit_", "_emit", "output_", "format_", "print_",
    )),
    (SemanticRole.ALLOCATOR, (
        "alloc", "malloc", "new_", "create_", "_new", "_create",
        "init_", "_init", "setup_", "construct",
    )),
    (SemanticRole.ITERATOR, (
        "foreach", "iter_", "_iter", "_next", "next_",
        "walk_", "_walk", "traverse", "visit",
    )),
    (SemanticRole.STATE_MACHINE, (
        "state_", "_state", "fsm", "transition", "switch_state",
        "process_event", "dispatch_event",
    )),
    (SemanticRole.DISPATCHER, (
        "dispatch", "route_", "_route", "callback",
        "event_", "_event", "signal_", "handler",
    )),
    (SemanticRole.MATH, (
        "sqrt", "pow_", "_pow", "log_", "_log", "_math",
        "sin_", "cos_", "mul_", "div_", "mod_",
    )),
    (SemanticRole.WRAPPER, (
        "wrapper", "_wrapper", "wrap_", "_wrap",
    )),
]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.3 — Five-Score Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FiveScores:
    """
    Five independent [0,1] scores per function.
    Final score = weighted sum per the paper's formula.
    """
    exposure:        float = 0.0   # attacker data reachability
    validation:      float = 0.0   # bounds-checking / conditional logic
    arithmetic:      float = 0.0   # arithmetic risk (overflow patterns)
    graph:           float = 0.0   # graph centrality / importance
    semantic:        float = 0.0   # learned role / cross-binary knowledge
    pattern_store:   float = 0.0   # structural pattern matches
    memory:          float = 0.0   # memory operation density
    dangerous_calls: float = 0.0   # direct/transitive calls to known sinks

    @property
    def total(self) -> float:
        return round(
            0.20 * self.exposure
          + 0.20 * self.validation
          + 0.20 * self.arithmetic
          + 0.15 * self.graph
          + 0.10 * self.semantic
          + 0.05 * self.pattern_store
          + 0.05 * self.memory
          + 0.05 * self.dangerous_calls,
            4,
        )

    def to_reasons(self) -> list[tuple[str, float]]:
        """Convert to (label, weight) pairs for backwards-compat with Stage 3/4."""
        return [
            ("exposure",        round(0.20 * self.exposure,        4)),
            ("validation",      round(0.20 * self.validation,      4)),
            ("arithmetic",      round(0.20 * self.arithmetic,      4)),
            ("graph",           round(0.15 * self.graph,           4)),
            ("semantic",        round(0.10 * self.semantic,        4)),
            ("pattern_store",   round(0.05 * self.pattern_store,   4)),
            ("memory",          round(0.05 * self.memory,          4)),
            ("dangerous_calls", round(0.05 * self.dangerous_calls, 4)),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Filter Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    func:             FunctionInfo
    score:            float
    scores:           FiveScores
    role:             SemanticRole        = SemanticRole.UNKNOWN
    role_confidence:  float              = 0.0
    role_source:      str               = "none"
    reasons:          list[tuple[str, float]] = field(default_factory=list)
    discarded:        bool              = False
    discard_reason:   str               = ""
    reachability_score: float           = 1.0
    expanded:         bool              = False   # added via neighborhood expansion

    def explain(self) -> str:
        if self.discarded:
            return (f"  DISCARDED  {self.func.name}  "
                    f"@ {self.func.entry_addr}  reason: {self.discard_reason}")
        lines = [
            f"  [{self.score:.3f}]  {self.func.name}  "
            f"@ {self.func.entry_addr}  role={self.role.value}({self.role_confidence:.2f})"
        ]
        for label, w in sorted(self.reasons, key=lambda x: -x[1]):
            if w > 0:
                lines.append(f"    +{w:.3f}  {label}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.1 — Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

_ALLOC_NAMES: frozenset[str] = frozenset({
    "malloc", "calloc", "realloc", "alloca",
    "sqlite3Malloc", "sqlite3MallocZero", "sqlite3Realloc",
    "sqlite3DbMalloc", "sqlite3DbMallocRaw", "sqlite3DbRealloc",
    "sqlite3_malloc", "sqlite3_malloc64", "sqlite3_realloc",
    "_TIFFmalloc", "_TIFFcalloc", "_TIFFrealloc",
    "xmlMalloc", "xmlMallocAtomic", "xmlRealloc",
    "luaM_realloc_", "luaM_malloc_",
    "OPENSSL_malloc", "OPENSSL_realloc", "CRYPTO_malloc",
})


class FeatureExtractor:
    """Stage 2.1: Extract granular static features from P-code ops."""

    @staticmethod
    def extract(func: FunctionInfo) -> FunctionFeatures:
        ops    = func.ops or []
        ft     = FunctionFeatures(instruction_count=len(ops))
        flags  = func.flags or {}

        # Accumulate op-type counts in a single pass
        op_counts: Counter = Counter(op.get("op", "") for op in ops)

        # Structural
        branches = (op_counts["CBRANCH"] + op_counts["BRANCH"]
                    + op_counts["BRANCHIND"] + op_counts["RETURN"])
        ft.basic_block_count = max(1, branches)
        ft.cyclomatic        = op_counts["CBRANCH"] + 1

        # Memory
        ft.load_count  = op_counts["LOAD"]
        ft.store_count = op_counts["STORE"]

        # Arithmetic — each op type counted separately
        ft.int_mult    = op_counts["INT_MULT"]
        ft.int_add     = op_counts["INT_ADD"]
        ft.int_sub     = op_counts["INT_SUB"]
        ft.shift_count = (op_counts["INT_LEFT"]
                         + op_counts["INT_RIGHT"]
                         + op_counts["INT_SRIGHT"])
        ft.cast_count  = (op_counts["INT_ZEXT"]
                         + op_counts["INT_SEXT"]
                         + op_counts["INT_TRUNC"])
        ft.ptradd      = op_counts["PTRADD"]
        ft.ptrsub      = op_counts["PTRSUB"]

        # Validation — first-class
        ft.int_less    = op_counts["INT_LESS"] + op_counts["INT_LESSEQUAL"]
        ft.int_equal   = op_counts["INT_EQUAL"] + op_counts["INT_NOTEQUAL"]
        ft.cbranch     = op_counts["CBRANCH"]

        # Calls
        sources_lower = {s.lower() for s in TAINT_SOURCE_IMPORTS}
        sinks_lower   = {s.lower() for s in TAINT_SINK_IMPORTS}
        alloc_lower   = {s.lower() for s in _ALLOC_NAMES}

        for op in ops:
            op_name = op.get("op", "")
            if op_name in ("CALL", "CALLIND"):
                ft.total_calls += 1
                if op_name == "CALLIND":
                    ft.indirect_calls += 1
                inputs = op.get("inputs") or []
                if inputs:
                    target = inputs[0].get("name", "").lower() if inputs else ""
                    if any(s in target for s in sources_lower):
                        ft.input_calls.append(target)
                    if any(s in target for s in sinks_lower):
                        ft.sink_calls.append(target)
                    if any(s in target for s in alloc_lower):
                        ft.heap_alloc = True

        # Stack allocation: large constant in STORE
        for op in ops:
            if op.get("op") != "STORE":
                continue
            for inp in (op.get("inputs") or []):
                if isinstance(inp, dict):
                    name = inp.get("name", "")
                    if name.startswith("const(0x"):
                        try:
                            if int(name[8:-1], 16) >= 512:
                                ft.stack_alloc = True
                                break
                        except ValueError:
                            pass

        # Unchecked arithmetic → STORE pattern (integer overflow to buffer write)
        arith_outputs: set[str] = set()
        checked_vars:  set[str] = set()
        for op in ops:
            mnemonic = op.get("op", "")
            out      = op.get("output")
            out_name = out.get("name", "") if isinstance(out, dict) else ""
            inputs   = op.get("inputs") or []
            in_names = {i.get("name", "") for i in inputs if isinstance(i, dict)}

            if mnemonic in ("INT_ADD", "INT_SUB", "INT_MULT") and out_name:
                arith_outputs.add(out_name)
            elif mnemonic == "CBRANCH":
                checked_vars.update(in_names)
            elif mnemonic == "STORE":
                if any(n in arith_outputs and n not in checked_vars for n in in_names):
                    ft.unchecked_arith_store = True

        return ft


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.2 — Semantic Role Classifier
# ─────────────────────────────────────────────────────────────────────────────

class RoleClassifier:
    """
    Assigns a SemanticRole with confidence from three sources (no LLM):
      1. Name heuristics   — fast pattern matching on function name
      2. Structural signals — feature-based inference when name is ambiguous
      3. pattern_store      — cross-binary learned roles from previous Stage 2.5 runs
    """

    @classmethod
    def classify(
        cls,
        func:     FunctionInfo,
        features: FunctionFeatures,
        store=None,
    ) -> RoleAssignment:
        name = func.name.lower()

        # 1 — Name heuristics (highest priority)
        for role, patterns in _ROLE_NAME_PATTERNS:
            for pat in patterns:
                if pat in name:
                    # Confidence based on how specific the match is
                    conf = 0.90 if len(pat) >= 6 else 0.75
                    return RoleAssignment(role, conf, "name")

        # 2 — pattern_store lookup (cross-binary learned roles)
        if store is not None:
            try:
                prev = store.get_learned_summary(func.name, [])
                if prev and not prev.get("null_result"):
                    role_str = prev.get("likely_role", "")
                    role_map = {
                        "parser": SemanticRole.PARSER,
                        "validator": SemanticRole.VALIDATOR,
                        "decoder": SemanticRole.DECODER,
                        "allocator": SemanticRole.ALLOCATOR,
                        "serializer": SemanticRole.SERIALIZER,
                        "iterator": SemanticRole.ITERATOR,
                        "dispatcher": SemanticRole.DISPATCHER,
                        "error_handler": SemanticRole.ERROR_HANDLER,
                        "checksum": SemanticRole.CHECKSUM,
                        "state_machine": SemanticRole.STATE_MACHINE,
                        "math": SemanticRole.MATH,
                        "wrapper": SemanticRole.WRAPPER,
                    }
                    if role_str in role_map:
                        return RoleAssignment(role_map[role_str], 0.80, "pattern_store")
            except Exception:
                pass

        # 3 — Structural inference (fall-through for unknown names)
        return cls._structural_classify(features)

    @staticmethod
    def _structural_classify(features: FunctionFeatures) -> RoleAssignment:
        """Infer role from feature counts when name patterns don't match."""
        # Error handler: very few loads/stores, few calls, low arithmetic
        if (features.load_count <= 2 and features.store_count <= 2
                and features.total_calls <= 3 and features.int_mult == 0
                and features.cbranch <= 3):
            return RoleAssignment(SemanticRole.ERROR_HANDLER, 0.40, "structural")

        # Validator: many CBRANCH + INT_LESS relative to size, few stores
        total_ops = max(features.instruction_count, 1)
        if (features.cbranch >= 3
                and features.int_less >= 2
                and features.store_count / total_ops < 0.05):
            return RoleAssignment(SemanticRole.VALIDATOR, 0.55, "structural")

        # Parser: has loops (via has_loop flag approximated by CBRANCH) + loads
        if (features.cbranch >= 4
                and features.load_count >= 5
                and features.total_calls >= 2):
            return RoleAssignment(SemanticRole.PARSER, 0.50, "structural")

        # Math: heavy arithmetic, few memory ops, few calls
        total_arith = (features.int_mult + features.int_add + features.int_sub
                      + features.shift_count)
        if total_arith >= 10 and features.load_count <= 3 and features.total_calls <= 2:
            return RoleAssignment(SemanticRole.MATH, 0.45, "structural")

        # Allocator: calls heap allocator
        if features.heap_alloc and features.total_calls <= 4:
            return RoleAssignment(SemanticRole.ALLOCATOR, 0.60, "structural")

        return RoleAssignment(SemanticRole.UNKNOWN, 0.0, "none")


# ─────────────────────────────────────────────────────────────────────────────
# Call Graph (expanded from previous version)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallGraph:
    callees: dict[str, set[str]] = field(default_factory=dict)
    callers: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, funcs: list[FunctionInfo]) -> "CallGraph":
        cg = cls()

        def _norm_addr(raw: str) -> str:
            s = raw.lower().strip()
            if s.startswith("ram(") and s.endswith(")"):
                s = s[4:-1]
            s = s.lstrip("0x") or "0"
            s = s.lstrip("0") or "0"
            return s

        addr_to_name: dict[str, str] = {}
        for func in funcs:
            addr_to_name[_norm_addr(func.entry_addr)] = func.name

        for func in funcs:
            edges: set[str] = set(func.call_sites)
            for op in (func.ops or []):
                if op.get("op") not in ("CALL", "CALLIND"):
                    continue
                inputs = op.get("inputs") or []
                if not inputs:
                    continue
                target = inputs[0].get("name", "")
                if not target:
                    continue
                if target.startswith("ram(") or target.startswith("0x"):
                    resolved = addr_to_name.get(_norm_addr(target), "")
                    if resolved:
                        edges.add(resolved)
                else:
                    edges.add(target)
            cg.callees[func.name] = edges
            for callee in edges:
                cg.callers.setdefault(callee, set()).add(func.name)

        return cg

    def callers_of(self, name: str) -> set[str]:
        return self.callers.get(name, set())

    def callees_of(self, name: str) -> set[str]:
        return self.callees.get(name, set())

    def calls_dangerous_transitively(
        self, name: str, dangerous: frozenset[str], depth: int = 2
    ) -> bool:
        visited: set[str] = set()
        queue = [(name, 0)]
        while queue:
            node, d = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for callee in self.callees.get(node, set()):
                if any(imp in callee.lower() for imp in {d.lower() for d in dangerous}):
                    return True
                if d < depth:
                    queue.append((callee, d + 1))
        return False

    def source_distance(
        self, name: str, sources: frozenset[str], max_hops: int = 4
    ) -> int:
        sources_lower = {s.lower() for s in sources}
        visited: set[str] = set()
        queue = [(name, 0)]
        while queue:
            node, d = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            if node.lower() in sources_lower or any(
                s in node.lower() for s in sources_lower
            ):
                return d
            if d < max_hops:
                for callee in self.callees.get(node, set()):
                    queue.append((callee, d + 1))
        return max_hops + 1

    def reachability_bfs(self, external_sources: frozenset[str]) -> set[str]:
        """
        Strategy 1: BFS from read-source functions (standalone programs).
        Strategy 2: write-path exclusion fallback (library binaries with no recv/read).
        """
        all_known = set(self.callees.keys()) | set(self.callers.keys())
        total     = max(len(all_known), 1)

        _READ_PREFIXES = (
            "parse_", "read_", "handle_", "decode_", "load_",
            "recv_",  "fetch_", "input_",  "unpack_", "deserializ",
        )
        _READ_INFIXES = (
            "_handle_", "_read_", "_parse_", "_decode_", "_load_",
            "_recv_",   "_fetch_", "_input_", "_unpack_",
        )

        def _is_read_source(name: str) -> bool:
            n = name.lower()
            return (n in {s.lower() for s in external_sources}
                    or any(n.startswith(p) for p in _READ_PREFIXES)
                    or any(tok in n for tok in _READ_INFIXES))

        reachable: set[str] = set()
        queue: list[str] = []
        for func_name in self.callees:
            if _is_read_source(func_name) or any(
                _is_read_source(c) for c in self.callees.get(func_name, set())
            ):
                if func_name not in reachable:
                    reachable.add(func_name)
                    queue.append(func_name)
        while queue:
            node = queue.pop()
            for caller in self.callers.get(node, set()):
                if caller not in reachable:
                    reachable.add(caller)
                    queue.append(caller)

        if len(reachable) / total < 0.05:
            _WRITE_TOKENS = frozenset({
                "_write_", "_write", "write_", "_encode_", "_encode", "encode_",
                "_serialize", "serialize_", "_output_", "output_", "_emit_", "emit_",
                "_deflate", "deflate_", "_flush", "flush_",
            })
            _WRITE_SINKS = frozenset({
                "fwrite", "write", "send", "sendto", "sendmsg",
                "puts", "putchar", "fputc", "fputs", "WriteFile", "WSASend",
            })
            _ERROR_TOKENS = frozenset({
                "error", "warning", "assert", "abort", "panic", "fatal",
                "longjmp", "setjmp", "exception", "signal",
            })

            def _is_write_path(fn: str) -> bool:
                n = fn.lower()
                return any(tok in n for tok in _WRITE_TOKENS)

            def _calls_write_sink(fn: str, depth: int = 6) -> bool:
                vis: set[str] = set()
                q = [(fn, 0)]
                while q:
                    node, d = q.pop()
                    if node in vis:
                        continue
                    vis.add(node)
                    for callee in self.callees.get(node, set()):
                        cl = callee.lower()
                        if any(tok in cl for tok in _ERROR_TOKENS):
                            continue
                        if any(s in cl for s in _WRITE_SINKS):
                            return True
                        if _is_write_path(callee):
                            return True
                        if d < depth:
                            q.append((callee, d + 1))
                return False

            write_only = {
                fn for fn in all_known
                if _is_write_path(fn) or _calls_write_sink(fn)
            }
            reachable = all_known - write_only

        return reachable

    def compute_graph_scores(
        self,
        dangerous: frozenset[str] = DANGEROUS_IMPORTS,
        alpha: float = 0.85,
        n_iter: int = 25,
    ) -> dict[str, float]:
        """
        Stage 2.3 — Graph Importance Score via Personalized PageRank (PPR).

        Seeds: functions that directly call a known dangerous sink/source.
        Propagation: upward through the call graph (callee → caller direction).

        A function scores high if it is on a call path leading INTO a dangerous
        operation — even if it doesn't call one directly.  This catches complex
        parser functions like sqlite3Select whose vulnerability is multiple hops
        from a memcpy/strcpy call.

        Returns func_name → [0, 1] PPR score.
        """
        all_nodes = set(self.callees.keys()) | set(self.callers.keys())
        if not all_nodes:
            return {}

        dangerous_lower = {d.lower() for d in dangerous}

        # Seeds: functions that directly call at least one dangerous import
        seeds: set[str] = set()
        for fn in all_nodes:
            for callee in self.callees.get(fn, set()):
                if any(imp in callee.lower() for imp in dangerous_lower):
                    seeds.add(fn)
                    break

        if not seeds:
            # Fall back to fan-in score if no seeds found
            fan_ins = {fn: len(self.callers.get(fn, set())) for fn in all_nodes}
            max_fi  = max(fan_ins.values(), default=1)
            return {fn: round(fi / max_fi, 4) for fn, fi in fan_ins.items()}

        # PPR personalization vector (uniform over seeds)
        seed_weight  = 1.0 / len(seeds)
        personalize  = {fn: (seed_weight if fn in seeds else 0.0) for fn in all_nodes}
        rank         = {fn: 1.0 / len(all_nodes) for fn in all_nodes}

        # Iterative PPR: score flows FROM callee TO caller
        # (if B is dangerous, A which calls B gets a boost)
        for _ in range(n_iter):
            new_rank: dict[str, float] = {}
            for node in all_nodes:
                incoming = 0.0
                for callee in self.callees.get(node, set()) & all_nodes:
                    n_callers = max(len(self.callers.get(callee, set()) & all_nodes), 1)
                    incoming += rank[callee] / n_callers
                new_rank[node] = alpha * incoming + (1.0 - alpha) * personalize[node]
            rank = new_rank

        max_r = max(rank.values(), default=1.0)
        if max_r == 0.0:
            return {fn: 0.0 for fn in all_nodes}
        return {fn: round(r / max_r, 4) for fn, r in rank.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.3 — Five-Score Scorer
# ─────────────────────────────────────────────────────────────────────────────

class FiveScoreScorer:
    """Computes the five independent scores for one function."""

    def __init__(
        self,
        cg:          CallGraph,
        reachable:   set[str],
        graph_scores: dict[str, float],
        matcher=None,
        store=None,
    ):
        self.cg           = cg
        self.reachable    = reachable
        self.graph_scores = graph_scores
        self.matcher      = matcher
        self.store        = store

        self._dangerous_lower = {d.lower() for d in DANGEROUS_IMPORTS}
        self._source_lower    = {s.lower() for s in TAINT_SOURCE_IMPORTS}
        self._sink_lower      = {s.lower() for s in TAINT_SINK_IMPORTS}

    def score(
        self,
        func:     FunctionInfo,
        features: FunctionFeatures,
        role:     RoleAssignment,
    ) -> FiveScores:
        name = func.name
        s    = FiveScores()

        # ── 1. Exposure Score [0,1] ──────────────────────────────────────────
        # Measures how likely attacker-controlled data reaches this function.
        if name in self.reachable:
            s.exposure += 0.35
        hop = self.cg.source_distance(name, TAINT_SOURCE_IMPORTS, max_hops=4)
        if hop <= 1:
            s.exposure += 0.50
        elif hop <= 2:
            s.exposure += 0.35
        elif hop <= 3:
            s.exposure += 0.20
        elif hop <= 4:
            s.exposure += 0.10
        if features.input_calls:
            s.exposure += 0.20
        # Semantic role boost: parsers and decoders are input-processing
        if role.role in (SemanticRole.PARSER, SemanticRole.DECODER):
            s.exposure += 0.20 * role.confidence
        elif role.role == SemanticRole.DISPATCHER:
            s.exposure += 0.10 * role.confidence
        s.exposure = min(s.exposure, 1.0)

        # ── 2. Arithmetic Risk Score [0,1] ───────────────────────────────────
        # INT_MULT is the strongest signal — width*height, count*size overflows.
        s.arithmetic += min(features.int_mult * 0.25, 0.50)
        s.arithmetic += min(features.shift_count * 0.10, 0.20)
        s.arithmetic += min((features.ptradd + features.ptrsub) * 0.10, 0.20)
        s.arithmetic += min(features.cast_count * 0.08, 0.15)
        s.arithmetic += min((features.int_add + features.int_sub) * 0.02, 0.15)
        if features.unchecked_arith_store:
            s.arithmetic += 0.35
        # Size / width / height patterns in prototype hint at dimension arithmetic
        proto_lower = func.prototype.lower()
        if any(tok in proto_lower for tok in ("width", "height", "size", "len", "count", "num")):
            s.arithmetic += 0.10
        s.arithmetic = min(s.arithmetic, 1.0)

        # ── 3. Validation Score [0,1] ────────────────────────────────────────
        # Paradox: functions with MANY checks often contain the bugs (a check
        # that's wrong or missing). Validators are prime CVE loci.
        total_ops = max(features.instruction_count, 1)
        # Dense CBRANCH = complex conditional logic
        cbranch_density = features.cbranch / total_ops
        s.validation += min(cbranch_density * 10.0, 0.35)
        # Explicit boundary comparisons
        s.validation += min(features.int_less * 0.08, 0.25)
        # Role is validator
        if role.role == SemanticRole.VALIDATOR:
            s.validation += 0.35 * role.confidence
        # Parser with arithmetic = likely has bounds checks (and possible missing ones)
        if role.role == SemanticRole.PARSER and features.int_mult > 0:
            s.validation += 0.15
        # Unchecked arith: arith result reaches store WITHOUT a prior check
        if features.unchecked_arith_store:
            s.validation += 0.20
        s.validation = min(s.validation, 1.0)

        # ── 4. Graph Importance Score [0,1] ──────────────────────────────────
        # Security-critical functions are usually call-graph junctions.
        s.graph += self.graph_scores.get(name, 0.0) * 0.60
        if name in self.reachable:
            s.graph += 0.20
        # Transitively calls a dangerous sink → bridge function
        if self.cg.calls_dangerous_transitively(name, DANGEROUS_IMPORTS, depth=2):
            s.graph += 0.20
        # Large dispatcher penalty: many callees → orchestrator, not vuln site
        n_callees = len(self.cg.callees.get(name, set()))
        if n_callees > 30:
            s.graph *= 0.5
        s.graph = min(s.graph, 1.0)

        # ── 5. Semantic Score [0,1] ──────────────────────────────────────────
        # confidence² makes the score non-linear:
        #   conf=0.90 → 0.81  (strong name match → full boost)
        #   conf=0.55 → 0.30  (weak structural inference → modest boost)
        #   conf=0.30 → 0.09  (uncertain → almost no boost)
        # This avoids rewarding ambiguous role assignments as much as clear ones.
        conf_sq = role.confidence ** 2
        if role.role in (SemanticRole.PARSER, SemanticRole.VALIDATOR, SemanticRole.DECODER):
            s.semantic += 0.50 * conf_sq
        elif role.role in (SemanticRole.STATE_MACHINE, SemanticRole.DISPATCHER):
            s.semantic += 0.35 * conf_sq
        elif role.role not in (SemanticRole.UNKNOWN, SemanticRole.ERROR_HANDLER,
                                SemanticRole.CHECKSUM, SemanticRole.MATH):
            s.semantic += 0.20 * conf_sq
        # Cross-binary LLM confirmation from pattern_store
        if self.store is not None:
            try:
                prev = self.store.get_learned_summary(func.name, [])
                if (prev and not prev.get("null_result")
                        and prev.get("likely_role") not in (None, "other", "logger")):
                    s.semantic += 0.40
            except Exception:
                pass
        s.semantic = min(s.semantic, 1.0)

        # ── 6. Pattern Store Score [0,1] ─────────────────────────────────────
        if self.matcher is not None:
            ranker = FunctionRanker(self.matcher)
            for op in (func.ops or []):
                if op.get("op") not in ("CALL", "CALLIND"):
                    continue
                result = self.matcher.match(op)
                if result is None or result.kind == MatchKind.NO_MATCH:
                    continue
                call_score, _ = ranker._score_call(result)
                s.pattern_store += call_score / 20.0
        s.pattern_store = min(s.pattern_store, 1.0)

        # ── 7. Memory Score [0,1] ────────────────────────────────────────────
        total_ops_f = max(features.instruction_count, 1)
        s.memory += min(features.load_count  / total_ops_f * 3, 0.40)
        s.memory += min(features.store_count / total_ops_f * 3, 0.40)
        if features.heap_alloc:
            s.memory += 0.15
        if features.stack_alloc:
            s.memory += 0.10
        s.memory = min(s.memory, 1.0)

        # ── 8. Dangerous Calls Score [0,1] ───────────────────────────────────
        calls_lower = " ".join(func.call_sites).lower()
        if any(s_name in calls_lower for s_name in self._sink_lower):
            s.dangerous_calls += 0.70
        elif any(s_name in calls_lower for s_name in self._source_lower):
            s.dangerous_calls += 0.50
        if features.sink_calls:
            s.dangerous_calls += 0.20
        # Direct caller of any dangerous import (1-hop)
        for callee in self.cg.callees.get(name, set()):
            if any(imp in callee.lower() for imp in self._dangerous_lower):
                s.dangerous_calls += 0.30
                break
        s.dangerous_calls = min(s.dangerous_calls, 1.0)

        return s


# ─────────────────────────────────────────────────────────────────────────────
# Discard rules
# ─────────────────────────────────────────────────────────────────────────────

def _should_discard(func: FunctionInfo, features: FunctionFeatures) -> str:
    """Return discard reason string or empty string if the function should be kept."""
    if features.instruction_count < 5:
        return "too_few_ops"

    # Pure computation with absolutely no memory access, no calls, no branches:
    # skip ONLY if it's also not a validator (validators may have no memory ops)
    has_memory = features.load_count > 0 or features.store_count > 0
    has_calls  = features.total_calls > 0
    has_cond   = features.cbranch > 0
    has_dangerous = bool(func.flags.get("calls_dangerous_import"))

    if not has_memory and not has_calls and not has_dangerous and not has_cond:
        return "pure_computation_no_memory"

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.4 — Graph Context Propagation
# ─────────────────────────────────────────────────────────────────────────────

def _propagate_scores(
    results:  dict[str, FilterResult],
    cg:       CallGraph,
    n_hops:   int = 2,
    caller_w: float = 0.15,
    callee_w: float = 0.10,
    ops_cap:  int   = 500,
) -> None:
    """
    Stage 2.4: Spread scores through the call graph.
    - Callee's score propagates upward to caller  (caller_w per hop)
    - Caller's score propagates downward to callee (callee_w per hop)
    This is why a Parser scoring high boosts its downstream Validator.
    """
    for _ in range(n_hops):
        updates: dict[str, float] = {}

        for name, r in results.items():
            if r.discarded:
                continue
            n_ops = r.func.flags.get("size_bytes", 0) or len(r.func.ops)
            prop_cap = 2.0 if n_ops > ops_cap else 999.0

            # Upward: caller gains from its callees
            callee_boost = sum(
                results[callee].score * caller_w
                for callee in cg.callees.get(name, set())
                if callee in results and not results[callee].discarded
            )
            callee_boost = min(callee_boost, prop_cap)

            # Downward: callee gains from its callers (smaller weight)
            caller_boost = sum(
                results[caller].score * callee_w
                for caller in cg.callers.get(name, set())
                if caller in results and not results[caller].discarded
            )

            updates[name] = callee_boost + caller_boost

        for name, boost in updates.items():
            results[name].score = round(results[name].score + boost, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.5 — Diversity & Category Selection
# ─────────────────────────────────────────────────────────────────────────────

# Base quota fractions — adapted per-binary in _adaptive_quotas()
_BASE_QUOTA_FRACTIONS: dict[str, float] = {
    "parser":       0.233,   # 70/300
    "validator":    0.200,   # 60/300
    "arithmetic":   0.167,   # 50/300
    "memory":       0.133,   # 40/300
    "unknown":      0.133,   # 40/300
    "state_machine":0.067,   # 20/300
    "error_handler":0.067,   # 20/300
}


def _detect_binary_profile(funcs: list) -> dict[str, float]:
    """
    Detect the functional composition of a binary from function name patterns.
    Returns a fraction-per-category dict (sums to ≤1.0, rest is unlabeled).

    Used to adapt category quotas: a parser-heavy binary (libpng, libxml2)
    gets more parser budget; a database binary (sqlite3) more validator budget;
    a crypto library more arithmetic budget.
    """
    counts: dict[str, int] = {cat: 0 for cat in _BASE_QUOTA_FRACTIONS}
    for func in funcs:
        name = func.name.lower() if hasattr(func, "name") else func.get("name", "").lower()
        for role, patterns in _ROLE_NAME_PATTERNS:
            if any(p in name for p in patterns):
                cat = _ROLE_TO_CATEGORY.get(role, "unknown")
                counts[cat] = counts.get(cat, 0) + 1
                break

    total = max(sum(counts.values()), 1)
    return {cat: count / total for cat, count in counts.items()}


def _adaptive_quotas(budget: int, profile: dict[str, float]) -> dict[str, int]:
    """
    Scale base quota fractions by observed binary profile.
    A category that's 2× more common in this binary gets up to 1.5× its base quota.
    All quotas are renormalized to sum to exactly `budget`.
    """
    base = _BASE_QUOTA_FRACTIONS
    raw: dict[str, float] = {}
    for cat, base_frac in base.items():
        observed = profile.get(cat, 0.0)
        # Ratio of observed to base: if 2× common → boost by 1.5×, capped at 2.0×
        ratio   = min(observed / max(base_frac, 0.01), 2.0)
        boosted = max(ratio, 0.5)   # floor: never below half base
        raw[cat] = base_frac * boosted

    # Renormalize to sum to 1.0
    total_raw = sum(raw.values())
    normalized = {cat: v / total_raw for cat, v in raw.items()}

    # Convert to integer slot counts, ensure each category gets at least 1
    quotas = {cat: max(1, int(frac * budget)) for cat, frac in normalized.items()}

    # Fix rounding: distribute leftover slots to largest categories
    used   = sum(quotas.values())
    excess = budget - used
    if excess > 0:
        for cat in sorted(quotas, key=lambda c: -quotas[c]):
            quotas[cat] += 1
            excess -= 1
            if excess == 0:
                break

    return quotas

_ROLE_TO_CATEGORY: dict[SemanticRole, str] = {
    SemanticRole.PARSER:        "parser",
    SemanticRole.DECODER:       "parser",
    SemanticRole.VALIDATOR:     "validator",
    SemanticRole.STATE_MACHINE: "state_machine",
    SemanticRole.DISPATCHER:    "state_machine",
    SemanticRole.ERROR_HANDLER: "error_handler",
    SemanticRole.ALLOCATOR:     "memory",
    SemanticRole.WRAPPER:       "memory",
    SemanticRole.ITERATOR:      "memory",
    SemanticRole.SERIALIZER:    "memory",
    SemanticRole.CHECKSUM:      "unknown",
    SemanticRole.MATH:          "unknown",
    SemanticRole.UNKNOWN:       "unknown",
}

# Functions with dominant arithmetic score go to "arithmetic" bucket regardless of role.
# Handled in _assign_category below.


def _assign_category(r: FilterResult) -> str:
    """Assign a function to a selection category."""
    # Arithmetic bucket: arithmetic score is the dominant dimension
    if r.scores.arithmetic >= 0.60 and r.scores.arithmetic >= r.scores.exposure:
        return "arithmetic"
    # Role-based assignment
    cat = _ROLE_TO_CATEGORY.get(r.role, "unknown")
    if cat == "arithmetic":   # shouldn't reach here but guard anyway
        return "unknown"
    return cat


def _select_with_quotas(
    ranked:  list[FilterResult],
    budget:  int,
    quotas:  Optional[dict[str, int]] = None,
) -> list[FilterResult]:
    """
    Stage 2.5: Fill category buckets according to adaptive quotas, then fill
    remaining budget globally from highest score.

    25% of budget is always reserved for global-score fill so that high-scoring
    functions that overflow their category quota (e.g., parser-heavy libxml2)
    still get selected on global merit.
    """
    # Use provided adaptive quotas or fall back to base fractions
    if quotas is None:
        quotas = {cat: max(1, int(frac * budget))
                  for cat, frac in _BASE_QUOTA_FRACTIONS.items()}

    # Reserve 25% of budget for global-score fill; scale category quotas to 75%
    global_reserve = max(10, int(budget * 0.25))
    quota_budget   = budget - global_reserve

    buckets: dict[str, list[FilterResult]] = {cat: [] for cat in quotas}
    for r in ranked:
        cat = _assign_category(r)
        buckets[cat].append(r)

    selected: list[FilterResult] = []
    seen: set[str] = set()

    # Phase 1: fill category quotas up to quota_budget
    for cat, quota in quotas.items():
        # Scale each category's quota proportionally to fit within quota_budget
        scaled = max(1, int(quota * quota_budget / budget))
        for r in buckets[cat][:scaled]:
            if r.func.name not in seen:
                selected.append(r)
                seen.add(r.func.name)

    # Phase 2: always fill remaining slots from globally ranked list
    # This catches overflow from any saturated category (e.g., parsers in libxml2)
    remaining = sorted(
        [r for r in ranked if r.func.name not in seen],
        key=lambda x: x.score,
        reverse=True,
    )
    for r in remaining:
        if len(selected) >= budget:
            break
        selected.append(r)
        seen.add(r.func.name)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.6 — Adaptive Neighborhood Expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expansion_threshold(selected: list[FilterResult]) -> float:
    """
    Compute the expansion threshold from the score distribution instead of
    using a hardcoded constant.

    Strategy: find the "elbow" in the sorted score curve — the point where
    the marginal gain between consecutive ranks drops to less than half the
    average gap across all selected functions.  Functions above this elbow
    have meaningfully higher scores than the rest and are the natural seeds
    for neighborhood expansion.

    Falls back to the 75th-percentile score if no clear elbow is found.
    """
    scores = sorted([r.score for r in selected if not r.discarded], reverse=True)
    if len(scores) < 10:
        return scores[0] if scores else 0.70

    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.01

    # Walk from index 5 onward (skip the very top to avoid noise from outliers)
    for i in range(5, len(gaps)):
        if gaps[i] < avg_gap * 0.50:   # marginal gain drops below half average
            return scores[i]

    # No clear elbow → use 75th-percentile score
    return scores[len(scores) // 4]


def _expand_neighborhood(
    selected:        list[FilterResult],
    all_results:     dict[str, FilterResult],
    cg:              CallGraph,
    score_threshold: Optional[float] = None,   # None → compute from distribution
    hops:            int   = 2,
    max_expansion:   int   = 80,
) -> list[FilterResult]:
    """
    Stage 2.6: For every selected function with score ≥ threshold, include
    its callers and high-scoring callees within `hops` hops.

    This catches functions like selectExpander / sqlite3Select that are
    callers of an in-budget high-scoring function (multiSelect), but rank
    below budget on their own because their name patterns don't include
    obvious sink calls.
    """
    # Compute data-driven threshold from score distribution if not provided
    threshold = score_threshold if score_threshold is not None else _expansion_threshold(selected)
    log.debug("Neighborhood expansion threshold: %.3f", threshold)

    selected_names = {r.func.name for r in selected}
    added: dict[str, FilterResult] = {}

    # Seed: high-scoring selected functions (above the score elbow)
    seeds = [r for r in selected if r.score >= threshold]

    queue: list[tuple[str, int]] = [(r.func.name, hops) for r in seeds]
    visited: set[str] = set(selected_names)

    while queue and len(added) < max_expansion:
        name, hops_left = queue.pop(0)
        if hops_left <= 0:
            continue

        # Callers of this function (outward expansion — catches functions that
        # lead INTO the high-scoring function)
        for caller in cg.callers.get(name, set()):
            if len(added) >= max_expansion:
                break
            if caller in visited:
                continue
            r = all_results.get(caller)
            if r is None or r.discarded:
                continue
            visited.add(caller)
            r.expanded = True
            added[caller] = r
            if hops_left > 1:
                queue.append((caller, hops_left - 1))

        # Callees with significant score (downward, toward sinks)
        for callee in cg.callees.get(name, set()):
            if len(added) >= max_expansion:
                break
            if callee in visited:
                continue
            r = all_results.get(callee)
            if r is None or r.discarded or r.score < 0.30:
                continue
            visited.add(callee)
            r.expanded = True
            added[callee] = r

    return list(added.values())


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint helper (unchanged — Stage 3 may use it)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fingerprint(ops: list[dict]) -> str:
    import hashlib
    parts = [
        f"{op.get('op','?')}({(op.get('output') or {}).get('size', 0)})"
        for op in ops
    ]
    return hashlib.sha256(" ".join(parts).encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────────────────

class FunctionFilterAgent:
    """
    Orchestrates Stage 2.1–2.6.

    Parameters
    ----------
    funcs      : FunctionInfo list from pcode.jsonl
    budget     : base candidate count (default 300); adaptive expansion adds up
                 to 30% more
    min_score  : score floor below which no function is kept (default 0.15)
    store_path : path to pattern_store.db (pass "" to disable)
    """

    def __init__(
        self,
        funcs:      list[FunctionInfo],
        budget:     int   = 300,
        min_score:  float = 0.15,
        store_path: str   = "",
    ):
        self.funcs     = funcs
        self.budget    = budget
        self.min_score = min_score

        # Build call graph
        self.call_graph = CallGraph.build(funcs)

        # Optional pattern_store / matcher
        matcher       = None
        pattern_store = None
        if _PATTERN_MATCHER_AVAILABLE and store_path and Path(store_path).exists():
            try:
                pattern_store = PatternStore(store_path)
                matcher       = PatternMatcher(pattern_store)
                log.info("PatternMatcher loaded from %s", store_path)
            except Exception as exc:
                log.warning("PatternMatcher init failed: %s — heuristic-only", exc)

        self._matcher = matcher
        self._store   = pattern_store

        # Reachability BFS (once)
        self._reachable = self.call_graph.reachability_bfs(TAINT_SOURCE_IMPORTS)
        log.debug(
            "Reachability BFS: %d / %d functions reachable from external sources",
            len(self._reachable), len(funcs),
        )

        # Graph importance scores (once)
        self._graph_scores = self.call_graph.compute_graph_scores()

        self._scorer  = FiveScoreScorer(
            self.call_graph, self._reachable, self._graph_scores,
            matcher=matcher, store=pattern_store,
        )
        self._results:     Optional[list[FilterResult]]       = None
        self._results_map: dict[str, FilterResult]            = {}
        self._candidates:  Optional[list[FilterResult]]       = None

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_jsonl(
        cls,
        path:       str | Path,
        budget:     int   = 300,
        min_score:  float = 0.15,
        store_path: str   = "",
    ) -> "FunctionFilterAgent":
        """Load from pcode.jsonl written by extractor.to_jsonl()."""
        funcs: list[FunctionInfo] = []
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    funcs.append(FunctionInfo(
                        name           = d.get("name", f"func_{lineno}"),
                        entry_addr     = d.get("entry_addr", "0x0"),
                        prototype      = d.get("prototype", ""),
                        ops            = d.get("ops", []),
                        call_sites     = d.get("call_sites", []),
                        flags          = d.get("flags", {}),
                        size_bytes     = d.get("size_bytes", 0),
                        decompile_time = d.get("decompile_time", 0.0),
                    ))
                except (json.JSONDecodeError, KeyError) as exc:
                    log.warning("Line %d: %s", lineno, exc)
        log.info("Loaded %d functions from %s", len(funcs), path)
        return cls(funcs, budget=budget, min_score=min_score, store_path=store_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def rank(self) -> list[FilterResult]:
        """
        Run all six sub-stages and return all FilterResults sorted by score.
        Discarded functions are placed at the end (score = 0).
        Result is cached — call once.
        """
        if self._results is not None:
            return self._results

        results_map: dict[str, FilterResult] = {}

        # Stage 2.1 + 2.2 + 2.3: features → role → five scores
        for func in self.funcs:
            features = FeatureExtractor.extract(func)
            discard  = _should_discard(func, features)

            if discard:
                r = FilterResult(
                    func=func, score=0.0, scores=FiveScores(),
                    role=SemanticRole.UNKNOWN, role_confidence=0.0,
                    discarded=True, discard_reason=discard,
                    reasons=[],
                )
                results_map[func.name] = r
                continue

            role_assign = RoleClassifier.classify(func, features, self._store)
            five        = self._scorer.score(func, features, role_assign)
            score       = five.total

            r = FilterResult(
                func=func, score=score, scores=five,
                role=role_assign.role,
                role_confidence=role_assign.confidence,
                role_source=role_assign.source,
                reasons=five.to_reasons(),
                reachability_score=(1.0 if func.name in self._reachable else 0.0),
            )
            results_map[func.name] = r

        # Stage 2.4: graph propagation (mutates scores in-place)
        _propagate_scores(results_map, self.call_graph)

        # Sort: non-discarded by score desc, discarded at end
        active   = sorted(
            [r for r in results_map.values() if not r.discarded],
            key=lambda x: x.score, reverse=True,
        )
        discarded = [r for r in results_map.values() if r.discarded]
        self._results = active + discarded
        self._results_map = results_map
        return self._results

    def _select_candidates(self) -> list[FilterResult]:
        """
        Run Stage 2.5 (diversity selection) + Stage 2.6 (neighborhood expansion).
        Returns the final candidate list ordered by score. Result is cached.
        """
        if self._candidates is not None:
            return self._candidates

        ranked = [
            r for r in self.rank()
            if not r.discarded and r.score >= self.min_score
        ]

        # Stage 2.5 — adaptive category-quota selection
        profile = _detect_binary_profile(self.funcs)
        quotas  = _adaptive_quotas(self.budget, profile)
        log.debug("Adaptive quotas: %s", quotas)
        selected = _select_with_quotas(ranked, self.budget, quotas=quotas)

        # Stage 2.6 — neighborhood expansion with data-driven threshold
        max_expansion = max(10, int(self.budget * 0.30))
        extras = _expand_neighborhood(
            selected        = selected,
            all_results     = self._results_map,
            cg              = self.call_graph,
            score_threshold = None,   # auto-computed from score distribution
            hops            = 2,
            max_expansion   = max_expansion,
        )
        combined = selected + extras

        # Re-sort and deduplicate
        seen: set[str] = set()
        final: list[FilterResult] = []
        for r in sorted(combined, key=lambda x: x.score, reverse=True):
            if r.func.name not in seen:
                final.append(r)
                seen.add(r.func.name)

        self._candidates = final
        return final

    def save_ranked(self, output_path: str | Path) -> None:
        """Write the final candidate set to pcode_ranked.jsonl."""
        output_path = Path(output_path)
        to_save = self._select_candidates()

        with output_path.open("w", encoding="utf-8") as fh:
            for r in to_save:
                fh.write(json.dumps({
                    "name":               r.func.name,
                    "entry_addr":         r.func.entry_addr,
                    "prototype":          r.func.prototype,
                    "score":              round(r.score, 4),
                    "discarded":          r.discarded,
                    "discard_reason":     r.discard_reason,
                    "reasons":            r.reasons,
                    "call_sites":         r.func.call_sites,
                    "flags":              r.func.flags,
                    "op_count":           len(r.func.ops),
                    "ops":                r.func.ops,
                    "fingerprint":        _compute_fingerprint(r.func.ops),
                    "reachability_score": r.reachability_score,
                    "semantic_role":      r.role.value,
                    "role_confidence":    round(r.role_confidence, 3),
                    "expanded":           r.expanded,
                    "scores": {
                        "exposure":        round(r.scores.exposure,        3),
                        "validation":      round(r.scores.validation,      3),
                        "arithmetic":      round(r.scores.arithmetic,      3),
                        "graph":           round(r.scores.graph,           3),
                        "semantic":        round(r.scores.semantic,        3),
                        "pattern_store":   round(r.scores.pattern_store,   3),
                        "memory":          round(r.scores.memory,          3),
                        "dangerous_calls": round(r.scores.dangerous_calls, 3),
                    },
                }) + "\n")

        log.info("Saved %d candidates → %s", len(to_save), output_path)

    def stats(self) -> dict:
        """Evaluation-ready statistics (pipeline.py reads kept/total/reduction_pct)."""
        all_results = self.rank()
        candidates  = self._select_candidates()

        kept      = len(candidates)
        total     = len(all_results)
        discarded = sum(1 for r in all_results if r.discarded)
        expanded  = sum(1 for r in candidates if r.expanded)

        role_counts: dict[str, int] = {}
        for r in candidates:
            role_counts[r.role.value] = role_counts.get(r.role.value, 0) + 1

        score_dist = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8+": 0}
        for r in all_results:
            if r.score < 0.2:   score_dist["0.0–0.2"] += 1
            elif r.score < 0.4: score_dist["0.2–0.4"] += 1
            elif r.score < 0.6: score_dist["0.4–0.6"] += 1
            elif r.score < 0.8: score_dist["0.6–0.8"] += 1
            else:                score_dist["0.8+"]    += 1

        return {
            "total":          total,
            "kept":           kept,
            "discarded":      discarded,
            "expanded":       expanded,
            "budget":         self.budget,
            "reduction_pct":  f"{(1 - kept / max(total, 1)) * 100:.1f}%",
            "score_dist":     score_dist,
            "role_dist":      role_counts,
        }

    def top(self, n: Optional[int] = None) -> list[FunctionInfo]:
        """Return FunctionInfo list for evaluation tooling."""
        limit = n or self.budget
        return [r.func for r in self._select_candidates()[:limit]]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt= "%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python filter_agent.py <pcode.jsonl> [budget] [min_score] [store.db]")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    budget     = int(sys.argv[2])    if len(sys.argv) > 2 else 300
    min_score  = float(sys.argv[3])  if len(sys.argv) > 3 else 0.15
    store_path = sys.argv[4]         if len(sys.argv) > 4 else ""

    agent = FunctionFilterAgent.from_jsonl(
        jsonl_path, budget=budget, min_score=min_score, store_path=store_path,
    )

    sep = "─" * 62
    stats = agent.stats()

    print(f"\n{sep}")
    print(f"  Functions loaded   : {stats['total']}")
    print(f"  Discarded          : {stats['discarded']}")
    print(f"  Candidates kept    : {stats['kept']}  (base budget={budget})")
    print(f"  Neighborhood added : {stats['expanded']}")
    print(f"  Reduction          : {stats['reduction_pct']}")
    print(f"\n  Role distribution:")
    for role, count in sorted(stats['role_dist'].items(), key=lambda x: -x[1]):
        print(f"    {role:<20s} {count:>4}")
    print(f"\n  Score distribution:")
    for bucket, count in stats['score_dist'].items():
        print(f"    {bucket}   {count:>4}")
    print(f"{sep}\n")

    print("Top 20 candidates:")
    for r in agent.rank()[:20]:
        if not r.discarded:
            print(f"  {r.score:6.3f}  {r.role.value:<14s}  {r.func.name}")
