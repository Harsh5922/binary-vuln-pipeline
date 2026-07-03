"""
stage3_sink_verifier.py  —  Stage 3D: Sink Verification
=========================================================
Evidence-based confirmation: the arithmetic is the vulnerability, not the sink.

TODAY: malloc() → known sink → candidate.  (generates false positives)
THIS MODULE:
  malloc() → Was size computed via INT_MULT? → Can it overflow? → Candidate
             No arithmetic evidence         → Filtered out

Sink classes and their evidence requirements:

  ALLOCATOR  malloc / calloc / realloc / …
             Requires: INT_MULT on the size arg + source_conf > 0.40
             Rationale: allocation itself is not a bug; unchecked arithmetic is.

  COPY       memcpy / memmove / strcpy / …
             Requires: source_conf > 0.50 (external data being copied)
             Reduced confidence when destination is bounds-checked.

  FORMAT     printf / sprintf / snprintf / …
             Requires: source_conf > 0.70 (format string from external data)

  FREE       free / g_free / TIFFClose / …
             Requires: freed-pointer-reuse (UAF) evidence in the candidate.

  GENERIC    Any other sink
             Requires: source_conf > 0.30

Usage
-----
    verifier = SinkVerifier()
    sink_class = verifier.classify("_TIFFmalloc")    # → SinkClass.ALLOCATOR
    ok, conf, reason = verifier.verify(
        sink_fn     = "_TIFFmalloc",
        sink_type   = "integer_overflow",
        source_conf = 0.87,
        size_var    = "VAR_7",
        via_mult    = True,
        is_checked  = False,
    )
    # → (True, 0.826, "allocator_arithmetic_evidence")
"""

from __future__ import annotations

import enum


class SinkClass(enum.Enum):
    ALLOCATOR = "allocator"
    COPY      = "copy"
    FORMAT    = "format"
    FREE      = "free"
    GENERIC   = "generic"


# ─── Allocators ───────────────────────────────────────────────────────────────
_ALLOCATORS: frozenset[str] = frozenset({
    "malloc", "calloc", "realloc", "reallocarray", "valloc", "pvalloc",
    "mmap", "mmap64", "brk",
    # libpng
    "png_malloc", "png_malloc_warn", "png_calloc",
    "png_malloc_base", "png_realloc_array", "png_zalloc",
    # libtiff
    "_TIFFmalloc", "_TIFFrealloc", "_TIFFcalloc",
    # libxml2
    "xmlMalloc", "xmlRealloc", "xmlMallocAtomic",
    # GLib
    "g_malloc", "g_malloc0", "g_malloc_n", "g_realloc",
    # SQLite
    "sqlite3_malloc", "sqlite3_malloc64", "sqlite3_realloc",
    # Misc
    "xmalloc", "zmalloc", "emalloc",
    "HeapAlloc", "VirtualAlloc", "LocalAlloc",
    "operator new", "operator new[]",
})

# ─── Copy sinks ───────────────────────────────────────────────────────────────
_COPY_SINKS: frozenset[str] = frozenset({
    "memcpy", "memmove", "bcopy", "memcpy_s", "memmove_s",
    "strcpy", "strcat", "stpcpy", "strcpy_s", "strcat_s",
    "png_memcpy", "png_memset",
    "wmemcpy", "wmemmove",
})

# ─── Format-string sinks ──────────────────────────────────────────────────────
_FORMAT_SINKS: frozenset[str] = frozenset({
    "printf",   "fprintf",  "sprintf",  "snprintf",
    "vprintf",  "vfprintf", "vsprintf", "vsnprintf",
    "wprintf",  "fwprintf", "swprintf",
})

# ─── Free / release sinks (UAF-relevant) ─────────────────────────────────────
_FREE_FUNS: frozenset[str] = frozenset({
    "free", "g_free", "png_free", "xmlFree",
    "TIFFClose", "sqlite3_free", "_TIFFfree",
    "operator delete", "operator delete[]",
})


def _normalise(fn_name: str) -> str:
    """Strip namespace/template decorators for matching."""
    return fn_name.split("::")[-1].split("(")[0].strip().lower()


class SinkVerifier:
    """
    Stage 3D — classify each sink and apply evidence-based confirmation.

    The verify() method returns (should_emit, adjusted_confidence, reason_tag).

      should_emit         — True if evidence supports emitting the candidate
      adjusted_confidence — confidence adjusted for evidence quality
      reason_tag          — short key describing why the decision was made
    """

    def classify(self, fn_name: str) -> SinkClass:
        """Return the sink class for a function name."""
        n = _normalise(fn_name)
        # Check substrings to catch variant names (png_malloc_warn, _TIFFrealloc, …)
        for name in _ALLOCATORS:
            if _normalise(name) in n or n in _normalise(name):
                return SinkClass.ALLOCATOR
        for name in _COPY_SINKS:
            if _normalise(name) in n or n in _normalise(name):
                return SinkClass.COPY
        for name in _FORMAT_SINKS:
            if _normalise(name) in n or n in _normalise(name):
                return SinkClass.FORMAT
        for name in _FREE_FUNS:
            if _normalise(name) in n or n in _normalise(name):
                return SinkClass.FREE
        return SinkClass.GENERIC

    def verify(
        self,
        sink_fn:     str,
        sink_type:   str,          # proposed vuln_type from 3B
        source_conf: float,        # source confidence from 3A / 3B
        size_var:    str,          # the size / data variable name (for audit)
        via_mult:    bool,         # was the size derived from INT_MULT?
        is_checked:  bool,         # was size/data bounds-checked?
        is_freed:    bool = False, # UAF context: was pointer freed before?
    ) -> tuple[bool, float, str]:
        """
        Decide whether the candidate should be emitted.

        Returns
        -------
        (should_emit, adjusted_confidence, reason_tag)
        """
        sink_class = self.classify(sink_fn)

        # ── ALLOCATOR ─────────────────────────────────────────────────────
        # malloc() is not a vulnerability.  The ARITHMETIC is.
        if sink_class == SinkClass.ALLOCATOR:
            if is_checked:
                return False, 0.0, "allocator_size_checked"
            if not via_mult:
                return False, 0.0, "allocator_no_mult_evidence"
            if source_conf < 0.40:
                return False, 0.0, "allocator_source_conf_too_low"
            adj = min(source_conf * 0.95, 0.90)
            return True, adj, "allocator_arithmetic_evidence"

        # ── COPY ──────────────────────────────────────────────────────────
        # memcpy / strcpy: external data being copied is the risk.
        if sink_class == SinkClass.COPY:
            if source_conf < 0.50:
                return False, 0.0, "copy_source_conf_too_low"
            if is_checked:
                # Bounded copy — reduced severity, not dismissed
                return True, source_conf * 0.45, "copy_with_bounds_check"
            adj = min(source_conf * 0.90, 0.85)
            return True, adj, "copy_unbounded"

        # ── FORMAT ────────────────────────────────────────────────────────
        # External format string → high external confidence required.
        if sink_class == SinkClass.FORMAT:
            if source_conf < 0.70:
                return False, 0.0, "format_source_conf_too_low"
            adj = min(source_conf * 0.85, 0.80)
            return True, adj, "format_external_string"

        # ── FREE ──────────────────────────────────────────────────────────
        # Use-after-free: only if there is lifetime evidence.
        if sink_class == SinkClass.FREE:
            if is_freed:
                return True, 0.75, "double_free_evidence"
            return False, 0.0, "free_no_lifetime_evidence"

        # ── GENERIC ───────────────────────────────────────────────────────
        # Any taint-reaching-sink with reasonable confidence.
        if source_conf < 0.30:
            return False, 0.0, "generic_source_conf_too_low"
        adj = min(source_conf * 0.85, 0.80)
        return True, adj, "generic_taint_reach"

    def is_allocator(self, fn_name: str) -> bool:
        """Quick check: is this function an allocator?"""
        return self.classify(fn_name) == SinkClass.ALLOCATOR
