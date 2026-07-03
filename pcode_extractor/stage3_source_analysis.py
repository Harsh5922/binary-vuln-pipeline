"""
stage3_source_analysis.py  —  Stage 3A: Source Confidence Analysis
===================================================================
Classifies external source functions by SourceRole (not by magic floats).

Reviewer-defensible design:
  - Every source function is assigned a SourceRole (category)
  - SourceRole.base_conf is derived from the category definition
  - No hand-tuned per-function confidence numbers

The SourceRole categories and their rationales are in stage3_evidence.py.

Usage
-----
    from stage3_source_analysis import SourceAnalyzer
    from stage3_evidence        import SourceRole

    analyzer = SourceAnalyzer()
    role, fn = analyzer.classify_call("png_crc_read")
    # → (SourceRole.LIBRARY_READER, "png_crc_read")

    seeds = analyzer.analyze_seeds(ops)
    # → {"VAR_3": (SourceRole.LIBRARY_READER, "png_crc_read"), ...}
"""

from __future__ import annotations

from stage3_evidence import SourceRole


# ─── Direct I/O (SourceRole.DIRECT_IO) ───────────────────────────────────────
# Functions where the kernel delivers raw attacker bytes without library filtering.

_DIRECT_IO: frozenset[str] = frozenset({
    "recv", "recvfrom", "recvmsg",
    "read", "fread", "fgets", "gets",
    "scanf", "sscanf", "fscanf",
    "getline", "readline",
})

# ─── Network-wrapped I/O (SourceRole.NETWORK_WRAPPED) ────────────────────────
# Same semantics as DIRECT_IO but through TLS/BIO.

_NETWORK_WRAPPED: frozenset[str] = frozenset({
    "BIO_read", "SSL_read", "SSL_read_ex",
    "ReadFile", "WSARecv", "WSARecvFrom",
})

# ─── Library Readers (SourceRole.LIBRARY_READER) ─────────────────────────────
# Library reads from its own I/O buffer. The file format header has been parsed
# but field VALUES are attacker-controlled.

_LIBRARY_READERS: frozenset[str] = frozenset({
    # libpng
    "png_crc_read", "png_read_data", "png_get_uint_32",
    "png_get_uint_16", "png_read_chunk_header", "png_read_buffer",
    # libtiff
    "TIFFReadDirectory", "TIFFGetField", "TIFFGetFieldDefaulted",
    "TIFFReadRawStrip", "TIFFReadEncodedStrip",
    # libxml2
    "xmlGetProp", "xmlGetNsProp", "xmlNodeGetContent",
    "xmlTextReaderRead", "xmlTextReaderGetAttribute",
    # libsndfile
    "psf_fread", "psf_binheader_readf", "psf_get_filelen",
    # OpenSSL / TLS application data
    "EVP_DecryptUpdate", "EVP_DecodeUpdate",
})

# ─── Parser Callbacks (SourceRole.PARSER_CALLBACK) ───────────────────────────
# Library parser invokes a callback with attacker-supplied content.

_PARSER_CALLBACKS: frozenset[str] = frozenset({
    # libxml2
    "xmlReadFile", "xmlReadDoc", "xmlParseDoc",
    "xmlParseFile", "xmlInputReadCallback",
    # Generic
    "xmlTextReaderRead",
})

# ─── Database Input (SourceRole.DB_INPUT) ────────────────────────────────────
# SQL bound parameters — tokenizer processes query but not VALUES.

_DB_INPUTS: frozenset[str] = frozenset({
    "sqlite3_value_text",   "sqlite3_value_blob",
    "sqlite3_value_int",    "sqlite3_value_int64",
    "sqlite3_value_double", "sqlite3_value_bytes",
    "sqlite3_column_text",  "sqlite3_column_blob",
    "sqlite3_column_bytes", "sqlite3_column_int",
    "sqlite3_str_vappendf",
})

# ─── CLI Arguments (SourceRole.CLI_ARGUMENT) ─────────────────────────────────
# Process may require authentication before reaching this code.

_CLI_ARGUMENTS: frozenset[str] = frozenset({
    "getopt", "getopt_long", "getopt_long_only",
    "poptGetArg", "optarg",
})

# ─── Environment (SourceRole.ENVIRONMENT) ────────────────────────────────────
# Less commonly attacker-controlled; OS may sanitize.

_ENVIRONMENT: frozenset[str] = frozenset({
    "getenv", "secure_getenv",
    "cuserid", "getlogin", "getlogin_r",
})

# ─── NOT sources — prevent false seeding of allocators/utilities ──────────────
_NOT_SOURCES: frozenset[str] = frozenset({
    "malloc", "calloc", "realloc", "free",
    "memcpy", "memset", "memmove", "strlen",
    "printf", "fprintf", "sprintf",
    "png_malloc", "_TIFFmalloc", "xmlMalloc",
})

# ─── Name-pattern fallbacks ────────────────────────────────────────────────────
# Applied in order when the function is not in any named set.
# (name substring, role)
_NAME_PATTERNS: list[tuple[str, SourceRole]] = [
    ("_callback",  SourceRole.PARSER_CALLBACK),
    ("_handler",   SourceRole.PARSER_CALLBACK),
    ("_readf",     SourceRole.LIBRARY_READER),
    ("_reader",    SourceRole.LIBRARY_READER),
    ("_read_",     SourceRole.LIBRARY_READER),
    ("_parse_",    SourceRole.PARSER_CALLBACK),
    ("_decode_",   SourceRole.LIBRARY_READER),
    ("_input_",    SourceRole.LIBRARY_READER),
    ("_recv",      SourceRole.DIRECT_IO),
    ("_receive",   SourceRole.DIRECT_IO),
    ("_fetch_",    SourceRole.LIBRARY_READER),
]


class SourceAnalyzer:
    """
    Stage 3A — classify source functions by SourceRole and produce seed maps.

    Two outputs:
      1. classify_call(fn_name) → (SourceRole, fn_name)
         Used by the orchestrator and backward slicer for confidence estimation.

      2. analyze_seeds(ops) → {var_name: (SourceRole, fn_name)}
         Used by the orchestrator to inject 3A seeds into TaintEngine (3B).
    """

    def classify_call(self, fn_name: str) -> tuple[SourceRole, str]:
        """
        Classify a function call by SourceRole.

        Returns (SourceRole.UNKNOWN, fn_name) when not a recognized source.
        """
        if not fn_name:
            return SourceRole.UNKNOWN, fn_name

        bare = fn_name.split("::")[-1].split("(")[0].strip()
        if bare in _NOT_SOURCES:
            return SourceRole.UNKNOWN, bare

        # Exact set lookup (ordered by specificity)
        for fn_set, role in (
            (_DIRECT_IO,       SourceRole.DIRECT_IO),
            (_NETWORK_WRAPPED, SourceRole.NETWORK_WRAPPED),
            (_LIBRARY_READERS, SourceRole.LIBRARY_READER),
            (_PARSER_CALLBACKS,SourceRole.PARSER_CALLBACK),
            (_DB_INPUTS,       SourceRole.DB_INPUT),
            (_CLI_ARGUMENTS,   SourceRole.CLI_ARGUMENT),
            (_ENVIRONMENT,     SourceRole.ENVIRONMENT),
        ):
            if bare in fn_set:
                return role, bare

        # Pattern fallback (case-insensitive substring)
        lc = bare.lower()
        for pattern, role in _NAME_PATTERNS:
            if pattern in lc:
                return role, bare

        return SourceRole.UNKNOWN, bare

    def get_confidence(self, fn_name: str) -> float:
        """Convenience wrapper: return SourceRole.base_conf for fn_name."""
        role, _ = self.classify_call(fn_name)
        return role.base_conf

    def analyze_seeds(self, ops: list[dict]) -> dict[str, tuple[SourceRole, str]]:
        """
        Scan function ops for external-source CALL/CALLIND instructions.

        Returns {var_name: (SourceRole, source_fn_name)} for every variable
        seeded by an external source call.

        Two variables are seeded per qualifying call:
          - The CALL's return value (output variable)
          - Out-pointer args (size-8 pointer args that the callee writes to)
        """
        seeds: dict[str, tuple[SourceRole, str]] = {}

        for op in ops:
            if op.get("op") not in ("CALL", "CALLIND"):
                continue

            inputs  = op.get("inputs") or []
            fn_name = inputs[0].get("name", "") if inputs else ""
            role, bare = self.classify_call(fn_name)
            if role == SourceRole.UNKNOWN:
                continue

            # Seed the return value
            out = op.get("output") or {}
            out_name = out.get("name", "")
            if out_name:
                existing = seeds.get(out_name)
                if existing is None or role.base_conf > existing[0].base_conf:
                    seeds[out_name] = (role, bare)

            # Seed out-pointer arguments (heuristic: size-8 non-const pointer args)
            for inp in inputs[1:]:
                if not isinstance(inp, dict):
                    continue
                if inp.get("size", 0) != 8:
                    continue
                iname = inp.get("name", "")
                if (
                    iname
                    and not iname.startswith("const(")
                    and not iname.startswith("ram(")
                ):
                    existing = seeds.get(iname)
                    if existing is None or role.base_conf > existing[0].base_conf:
                        seeds[iname] = (role, bare)

        return seeds

    def seeds_as_conf(self, ops: list[dict]) -> dict[str, float]:
        """
        Compatibility shim: return {var: base_conf} for TaintEngine injection.
        TaintEngine stores float source_conf in TaintState.var_source_conf.
        """
        raw = self.analyze_seeds(ops)
        return {var: role.base_conf for var, (role, _) in raw.items()}
