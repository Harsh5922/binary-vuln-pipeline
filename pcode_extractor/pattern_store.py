"""
pattern_store.py

SQLite-backed store for vulnerability pattern rules.

Three tiers:
  Tier 1  library_patterns    — hardcoded rules for known C/C++ functions
  Tier 2  structural_patterns — LLM-inferred rules keyed by arg-size signature
  Tier 3  fingerprint_patterns — confirmed vuln shapes for cross-binary matching
  Bonus   learned_patterns     — Stage 2.5 semantic summaries (role classification)
  Bonus   learned_rules        — Stage 2.5 taint rules derived from summaries
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded library patterns — Tier 1
# Key format: "lib|fn_name"
# Rule fields match MatchResult in pattern_matcher.py
# ─────────────────────────────────────────────────────────────────────────────

def _rule(
    *,
    external_input=None,
    writes_memory_at=-1,
    reads_from=-1,
    return_tainted=False,
    bounded=False,
    size_arg=-1,
    frees_memory_at=-1,
    return_is_buffer=False,
    sink=False,
    sink_type="",
    taint_arg=-1,
    confidence=0.95,
    vuln_score=1.0,
    notes="",
):
    return {
        "external_input":   external_input or [],
        "writes_memory_at": writes_memory_at,
        "reads_from":       reads_from,
        "return_tainted":   return_tainted,
        "bounded":          bounded,
        "size_arg":         size_arg,
        "frees_memory_at":  frees_memory_at,
        "return_is_buffer": return_is_buffer,
        "sink":             sink,
        "sink_type":        sink_type,
        "taint_arg":        taint_arg,
        "confidence":       confidence,
        "vuln_score":       vuln_score,
        "source":           "hardcoded",
        "notes":            notes,
    }


_LIBRARY_PATTERNS: dict[str, dict] = {

    # ── External input sources ────────────────────────────────────────────────
    # These bring attacker-controlled data into the program.

    "lib|fread": _rule(
        external_input=[0], writes_memory_at=0, return_tainted=True,
        bounded=True, size_arg=[1, 2],
        vuln_score=4, notes="fread(buf, size, count, stream) — external file input",
    ),
    "lib|read": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=4, notes="read(fd, buf, count) — external fd input",
    ),
    "lib|recv": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=5, notes="recv(fd, buf, len, flags) — socket input",
    ),
    "lib|recvfrom": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=5, notes="recvfrom — socket input with address",
    ),
    "lib|recvmsg": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=False, size_arg=-1,
        vuln_score=5, notes="recvmsg — socket message input",
    ),
    "lib|fgets": _rule(
        external_input=[0], writes_memory_at=0, return_tainted=True,
        bounded=True, size_arg=1,
        vuln_score=3, notes="fgets(buf, size, stream) — external input, bounded",
    ),
    "lib|gets": _rule(
        external_input=[0], writes_memory_at=0, return_tainted=True,
        bounded=False, sink=True, sink_type="buffer_overflow", taint_arg=0,
        vuln_score=9, notes="gets — UNBOUNDED external input, always dangerous",
    ),
    "lib|scanf": _rule(
        external_input=[], writes_memory_at="all_ptr_args", return_tainted=False,
        bounded=False,
        vuln_score=4, notes="scanf — external stdin, writes to pointer args",
    ),
    "lib|fscanf": _rule(
        external_input=[], writes_memory_at="all_ptr_args", return_tainted=False,
        bounded=False,
        vuln_score=4, notes="fscanf — external file, writes to pointer args",
    ),
    "lib|sscanf": _rule(
        external_input=[], writes_memory_at="all_ptr_args", return_tainted=False,
        bounded=False,
        vuln_score=3, notes="sscanf — string input, writes to pointer args",
    ),
    "lib|getenv": _rule(
        return_tainted=True,
        vuln_score=3, notes="getenv — environment variable (attacker-controlled in some contexts)",
    ),
    "lib|ReadFile": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=False,
        bounded=True, size_arg=2,
        vuln_score=4, notes="Win32 ReadFile — external file input",
    ),
    "lib|WSARecv": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=False,
        bounded=True, size_arg=2,
        vuln_score=5, notes="Winsock WSARecv — socket input",
    ),
    "lib|BIO_read": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=4, notes="OpenSSL BIO_read — network/file input",
    ),

    # ── Unbounded string copy / concat — classic sinks ────────────────────────
    "lib|strcpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=False, sink=True, sink_type="buffer_overflow", taint_arg=1,
        vuln_score=8, notes="strcpy(dst, src) — unbounded copy",
    ),
    "lib|strcat": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=False, sink=True, sink_type="buffer_overflow", taint_arg=1,
        vuln_score=7, notes="strcat(dst, src) — unbounded concat",
    ),
    "lib|sprintf": _rule(
        writes_memory_at=0,
        bounded=False, sink=True, sink_type="buffer_overflow", taint_arg=0,
        vuln_score=7, notes="sprintf — unbounded format string write to buffer",
    ),
    "lib|vsprintf": _rule(
        writes_memory_at=0,
        bounded=False, sink=True, sink_type="buffer_overflow", taint_arg=0,
        vuln_score=7, notes="vsprintf — unbounded format string write to buffer",
    ),
    "lib|wcscpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=False, sink=True, sink_type="buffer_overflow", taint_arg=1,
        vuln_score=7, notes="wcscpy — wide char unbounded copy",
    ),
    "lib|wcscat": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=False, sink=True, sink_type="buffer_overflow", taint_arg=1,
        vuln_score=7, notes="wcscat — wide char unbounded concat",
    ),

    # ── Bounded string copy / concat ──────────────────────────────────────────
    "lib|strncpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2,
        vuln_score=3, notes="strncpy(dst, src, n) — bounded copy",
    ),
    "lib|strncat": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2,
        vuln_score=3, notes="strncat(dst, src, n) — bounded concat",
    ),
    "lib|snprintf": _rule(
        writes_memory_at=0,
        bounded=True, size_arg=1,
        vuln_score=2, notes="snprintf(buf, size, fmt, ...) — bounded format write",
    ),
    "lib|vsnprintf": _rule(
        writes_memory_at=0,
        bounded=True, size_arg=1,
        vuln_score=2, notes="vsnprintf — bounded format write",
    ),
    "lib|strlcpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2,
        vuln_score=2, notes="strlcpy — OpenBSD bounded copy",
    ),
    "lib|strlcat": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2,
        vuln_score=2, notes="strlcat — OpenBSD bounded concat",
    ),
    "lib|wcsncat": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2,
        vuln_score=3, notes="wcsncat — wide char bounded concat",
    ),
    "lib|wcsncpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2,
        vuln_score=3, notes="wcsncpy — wide char bounded copy",
    ),

    # ── Memory copy / move ────────────────────────────────────────────────────
    "lib|memcpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2, sink=True, sink_type="buffer_overflow", taint_arg=2,
        vuln_score=5, notes="memcpy(dst, src, n) — bounded by size arg",
    ),
    "lib|memmove": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2, sink=True, sink_type="buffer_overflow", taint_arg=2,
        vuln_score=5, notes="memmove — safe overlap, bounded by size arg",
    ),
    "lib|mempcpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2, return_tainted=True,
        vuln_score=4, notes="mempcpy — like memcpy, returns end pointer",
    ),
    "lib|memcpy_s": _rule(
        writes_memory_at=0, reads_from=2,
        bounded=True, size_arg=1,
        vuln_score=3, notes="memcpy_s(dst, dstsize, src, count) — safe version",
    ),
    "lib|memmove_s": _rule(
        writes_memory_at=0, reads_from=2,
        bounded=True, size_arg=1,
        vuln_score=3, notes="memmove_s — safe memmove",
    ),
    "lib|bcopy": _rule(
        writes_memory_at=1, reads_from=0,
        bounded=True, size_arg=2, sink=True, sink_type="buffer_overflow", taint_arg=2,
        vuln_score=5, notes="bcopy(src, dst, n) — arg order reversed vs memcpy",
    ),
    "lib|png_memcpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2, sink=True, sink_type="buffer_overflow", taint_arg=2,
        vuln_score=5, notes="png_memcpy — libpng wrapper for memcpy",
    ),
    "lib|memset": _rule(
        writes_memory_at=0,
        bounded=True, size_arg=2,
        vuln_score=1, notes="memset(buf, byte, n) — memory initialization",
    ),

    # ── Format string sinks ───────────────────────────────────────────────────
    "lib|printf": _rule(
        sink=True, sink_type="format_string", taint_arg=0,
        vuln_score=6, notes="printf(fmt, ...) — format string sink",
    ),
    "lib|fprintf": _rule(
        sink=True, sink_type="format_string", taint_arg=1,
        vuln_score=6, notes="fprintf(stream, fmt, ...) — format string sink",
    ),
    "lib|vprintf": _rule(
        sink=True, sink_type="format_string", taint_arg=0,
        vuln_score=6, notes="vprintf — format string sink",
    ),
    "lib|vfprintf": _rule(
        sink=True, sink_type="format_string", taint_arg=1,
        vuln_score=6, notes="vfprintf — format string sink",
    ),
    "lib|syslog": _rule(
        sink=True, sink_type="format_string", taint_arg=1,
        vuln_score=5, notes="syslog(priority, fmt, ...) — format string to syslog",
    ),
    "lib|err": _rule(
        sink=True, sink_type="format_string", taint_arg=1,
        vuln_score=4, notes="err(status, fmt, ...) — format string to stderr",
    ),
    "lib|warn": _rule(
        sink=True, sink_type="format_string", taint_arg=0,
        vuln_score=4, notes="warn(fmt, ...) — format string to stderr",
    ),

    # ── Command execution sinks ───────────────────────────────────────────────
    "lib|system": _rule(
        sink=True, sink_type="command_injection", taint_arg=0,
        vuln_score=10, notes="system(cmd) — executes shell command",
    ),
    "lib|popen": _rule(
        sink=True, sink_type="command_injection", taint_arg=0,
        return_tainted=True,
        vuln_score=9, notes="popen(cmd, mode) — opens pipe to shell command",
    ),
    "lib|execve": _rule(
        sink=True, sink_type="command_injection", taint_arg=0,
        vuln_score=10, notes="execve(path, argv, envp) — exec",
    ),
    "lib|execl": _rule(
        sink=True, sink_type="command_injection", taint_arg=0,
        vuln_score=10, notes="execl — exec family",
    ),
    "lib|execlp": _rule(
        sink=True, sink_type="command_injection", taint_arg=0,
        vuln_score=10, notes="execlp — exec with PATH search",
    ),
    "lib|execv": _rule(
        sink=True, sink_type="command_injection", taint_arg=0,
        vuln_score=10, notes="execv — exec with argv",
    ),
    "lib|execvp": _rule(
        sink=True, sink_type="command_injection", taint_arg=0,
        vuln_score=10, notes="execvp — exec with PATH + argv",
    ),
    "lib|ShellExecute": _rule(
        sink=True, sink_type="command_injection", taint_arg=2,
        vuln_score=9, notes="Win32 ShellExecute",
    ),
    "lib|CreateProcess": _rule(
        sink=True, sink_type="command_injection", taint_arg=1,
        vuln_score=9, notes="Win32 CreateProcess",
    ),

    # ── Allocators ────────────────────────────────────────────────────────────
    "lib|malloc": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=0, size_arg=0,
        vuln_score=3, notes="malloc(size) — heap allocation, size may overflow",
    ),
    "lib|calloc": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=-1, size_arg=[0, 1],
        vuln_score=4, notes="calloc(count, size) — heap alloc, count*size may overflow",
    ),
    "lib|realloc": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=1, size_arg=1,
        vuln_score=4, notes="realloc(ptr, size) — heap resize",
    ),
    "lib|reallocarray": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=-1, size_arg=[1, 2],
        vuln_score=4, notes="reallocarray(ptr, count, size) — safe realloc",
    ),
    "lib|valloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="valloc — page-aligned malloc",
    ),
    "lib|pvalloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="pvalloc — rounded page-aligned malloc",
    ),
    "lib|mmap": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="mmap(addr, length, ...) — memory map",
    ),
    "lib|mmap64": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="mmap64 — 64-bit mmap",
    ),
    "lib|HeapAlloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=2,
        vuln_score=3, notes="Win32 HeapAlloc",
    ),
    "lib|VirtualAlloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="Win32 VirtualAlloc",
    ),
    "lib|LocalAlloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="Win32 LocalAlloc",
    ),
    "lib|GlobalAlloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="Win32 GlobalAlloc",
    ),
    "lib|xmalloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="xmalloc — checked malloc",
    ),
    "lib|zmalloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="zmalloc — zeroing malloc",
    ),
    "lib|smalloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="smalloc — safe malloc",
    ),
    "lib|emalloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="emalloc — error-checking malloc",
    ),
    "lib|operator new": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="C++ operator new",
    ),
    "lib|operator new[]": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="C++ operator new[]",
    ),

    # ── libpng allocators ─────────────────────────────────────────────────────
    "lib|png_malloc": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=1, size_arg=1,
        vuln_score=3, notes="png_malloc(png_ptr, size) — arg1=size",
    ),
    "lib|png_malloc_warn": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=1, size_arg=1,
        vuln_score=3, notes="png_malloc_warn(png_ptr, size)",
    ),
    "lib|png_calloc": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=1, size_arg=1,
        vuln_score=3, notes="png_calloc(png_ptr, size)",
    ),
    "lib|png_malloc_base": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="png_malloc_base — internal libpng allocator",
    ),
    "lib|png_realloc_array": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=2, size_arg=[2, 3],
        vuln_score=4, notes="png_realloc_array(png_ptr, old_ptr, old_count, add_count, element_size)",
    ),
    "lib|png_zalloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="png_zalloc — zlib allocation hook for libpng",
    ),

    # ── GLib allocators ───────────────────────────────────────────────────────
    "lib|g_malloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="GLib g_malloc",
    ),
    "lib|g_malloc0": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="GLib g_malloc0 — zeroing malloc",
    ),
    "lib|g_malloc_n": _rule(
        return_tainted=False, return_is_buffer=True,
        sink=True, sink_type="integer_overflow", taint_arg=-1, size_arg=[0, 1],
        vuln_score=4, notes="GLib g_malloc_n(count, size) — count*size may overflow",
    ),
    "lib|g_realloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="GLib g_realloc",
    ),
    "lib|g_try_malloc": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="GLib g_try_malloc — non-aborting malloc",
    ),
    "lib|g_new": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="GLib g_new macro — typed allocation",
    ),
    "lib|g_new0": _rule(
        return_tainted=False, return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="GLib g_new0 macro — zeroing typed allocation",
    ),

    # ── Free functions ────────────────────────────────────────────────────────
    "lib|free": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="free(ptr) — heap deallocation",
    ),
    "lib|cfree": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="cfree — legacy free",
    ),
    "lib|png_free": _rule(
        frees_memory_at=1,
        vuln_score=2, notes="png_free(png_ptr, ptr) — arg1 is the freed ptr",
    ),
    "lib|png_free_data": _rule(
        frees_memory_at=2,
        vuln_score=2, notes="png_free_data(png_ptr, info_ptr, mask, num) — arg2 freed",
    ),
    "lib|g_free": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="GLib g_free(ptr)",
    ),

    # ── libpng I/O — external data sources ───────────────────────────────────
    "lib|png_read_data": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=False,
        bounded=True, size_arg=2,
        vuln_score=4, notes="png_read_data(png_ptr, buf, size) — reads from PNG stream",
    ),
    "lib|png_read_row": _rule(
        writes_memory_at=1, reads_from=-1,
        bounded=False,
        vuln_score=3, notes="png_read_row — reads one image row",
    ),
    "lib|png_read_rows": _rule(
        writes_memory_at=1, reads_from=-1,
        bounded=True, size_arg=3,
        vuln_score=3, notes="png_read_rows — reads multiple image rows",
    ),
    "lib|png_safe_execute": _rule(
        sink=True, sink_type="null_dereference", taint_arg=1,
        vuln_score=4, notes="png_safe_execute(image, fn_ptr, arg) — CALLIND via fn_ptr; NULL check",
    ),

    # ── libsndfile ────────────────────────────────────────────────────────────
    "lib|psf_binheader_readf": _rule(
        writes_memory_at="all_ptr_args",
        bounded=False,
        vuln_score=4, notes="libsndfile: reads binary header fields into pointer args",
    ),
    "lib|psf_fread": _rule(
        external_input=[0], writes_memory_at=0, return_tainted=True,
        bounded=True, size_arg=[1, 2],
        vuln_score=4, notes="libsndfile psf_fread — file input",
    ),

    # ── libtiff ───────────────────────────────────────────────────────────────
    "lib|TIFFGetField": _rule(
        writes_memory_at="all_ptr_args", return_tainted=False,
        bounded=False,
        vuln_score=4, notes="TIFFGetField — reads TIFF tag value into pointer args",
    ),
    "lib|TIFFReadRawTile": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=4, notes="TIFFReadRawTile — reads raw tile data",
    ),
    "lib|TIFFReadEncodedTile": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=4, notes="TIFFReadEncodedTile",
    ),
    "lib|TIFFReadRawStrip": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=4, notes="TIFFReadRawStrip",
    ),
    "lib|TIFFReadEncodedStrip": _rule(
        external_input=[1], writes_memory_at=1, return_tainted=True,
        bounded=True, size_arg=2,
        vuln_score=4, notes="TIFFReadEncodedStrip",
    ),
    "lib|_TIFFmalloc": _rule(
        return_is_buffer=True, size_arg=0,
        sink=True, sink_type="integer_overflow", taint_arg=0,
        vuln_score=3, notes="libtiff _TIFFmalloc(size)",
    ),
    "lib|_TIFFrealloc": _rule(
        return_is_buffer=True, size_arg=1,
        sink=True, sink_type="integer_overflow", taint_arg=1,
        vuln_score=3, notes="libtiff _TIFFrealloc(ptr, size)",
    ),
    "lib|_TIFFfree": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="libtiff _TIFFfree(ptr)",
    ),
    "lib|_TIFFmemcpy": _rule(
        writes_memory_at=0, reads_from=1,
        bounded=True, size_arg=2, sink=True, sink_type="buffer_overflow", taint_arg=2,
        vuln_score=5, notes="libtiff _TIFFmemcpy — memcpy wrapper",
    ),

    # ── libxml2 ───────────────────────────────────────────────────────────────
    "lib|xmlGetProp": _rule(
        return_tainted=True,
        vuln_score=3, notes="xmlGetProp — returns XML attribute as tainted string",
    ),
    "lib|xmlNodeGetContent": _rule(
        return_tainted=True,
        vuln_score=3, notes="xmlNodeGetContent — returns XML text content",
    ),
    "lib|xmlStrndup": _rule(
        return_tainted=True, return_is_buffer=True,
        vuln_score=3, notes="xmlStrndup — XML string duplicate",
    ),
    "lib|xmlMalloc": _rule(
        return_is_buffer=True, size_arg=0,
        sink=True, sink_type="integer_overflow", taint_arg=0,
        vuln_score=3, notes="libxml2 xmlMalloc",
    ),
    "lib|xmlMallocAtomic": _rule(
        return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="libxml2 xmlMallocAtomic",
    ),
    "lib|xmlRealloc": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="libxml2 xmlRealloc",
    ),
    "lib|xmlFree": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="libxml2 xmlFree",
    ),
    "lib|xmlMemMalloc": _rule(
        return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="libxml2 xmlMemMalloc",
    ),
    "lib|xmlMemRealloc": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="libxml2 xmlMemRealloc",
    ),
    "lib|xmlMemFree": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="libxml2 xmlMemFree",
    ),

    # ── libxml2 I/O sources ───────────────────────────────────────────────────
    # These write raw bytes into a plain char* buffer (not into opaque structs),
    # so external_input propagation stays contained to the actual data buffer.
    "lib|xmlParserInputBufferRead": _rule(
        external_input=[0], writes_memory_at=0,
        vuln_score=4, notes="xmlParserInputBufferRead — fills input buffer from source",
    ),
    "lib|xmlParserInputBufferPush": _rule(
        external_input=[2], writes_memory_at=2,
        vuln_score=4, notes="xmlParserInputBufferPush(in, len, buf) — pushes raw bytes into parser",
    ),
    "lib|xmlParseChunk": _rule(
        external_input=[1], writes_memory_at=1,
        vuln_score=4, notes="xmlParseChunk(ctxt, chunk, size, terminate) — feeds raw bytes to parser",
    ),

    # ── PHP stream I/O sources ─────────────────────────────────────────────────
    "lib|php_stream_read": _rule(
        external_input=[1], writes_memory_at=1,
        vuln_score=4, notes="php_stream_read(stream, buf, count) — reads external data into buf",
    ),
    "lib|_php_stream_read": _rule(
        external_input=[1], writes_memory_at=1,
        vuln_score=4, notes="_php_stream_read(stream, buf, count) — internal PHP stream read",
    ),
    "lib|php_stream_gets": _rule(
        external_input=[1], writes_memory_at=1,
        vuln_score=4, notes="php_stream_gets(stream, buf, maxlen) — reads line into buf",
    ),
    "lib|php_stream_fill_read_buffer": _rule(
        external_input=[0], writes_memory_at=0,
        vuln_score=3, notes="php_stream_fill_read_buffer — fills stream read buffer",
    ),
    "lib|zend_stream_read": _rule(
        external_input=[1], writes_memory_at=1,
        vuln_score=4, notes="zend_stream_read(stream, buf, count) — Zend stream read",
    ),

    # ── SQLite3 ───────────────────────────────────────────────────────────────
    "lib|sqlite3_value_text": _rule(
        return_tainted=True,
        vuln_score=3, notes="sqlite3_value_text — returns user-supplied SQL value as string",
    ),
    "lib|sqlite3_value_blob": _rule(
        return_tainted=True,
        vuln_score=3, notes="sqlite3_value_blob — returns user blob",
    ),
    "lib|sqlite3_value_int": _rule(
        return_tainted=True,
        vuln_score=2, notes="sqlite3_value_int — returns user integer",
    ),
    "lib|sqlite3_value_int64": _rule(
        return_tainted=True,
        vuln_score=2, notes="sqlite3_value_int64 — returns user 64-bit int",
    ),
    "lib|sqlite3_column_text": _rule(
        return_tainted=True,
        vuln_score=3, notes="sqlite3_column_text — query column value",
    ),
    "lib|sqlite3_column_blob": _rule(
        return_tainted=True,
        vuln_score=3, notes="sqlite3_column_blob — query column blob",
    ),
    "lib|sqlite3_column_int": _rule(
        return_tainted=True,
        vuln_score=2, notes="sqlite3_column_int",
    ),
    "lib|sqlite3_column_int64": _rule(
        return_tainted=True,
        vuln_score=2, notes="sqlite3_column_int64",
    ),
    "lib|sqlite3Malloc": _rule(
        return_is_buffer=True, size_arg=0,
        sink=True, sink_type="integer_overflow", taint_arg=0,
        vuln_score=3, notes="SQLite internal malloc",
    ),
    "lib|sqlite3Realloc": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="SQLite internal realloc",
    ),
    "lib|sqlite3MallocZero": _rule(
        return_is_buffer=True, size_arg=0,
        vuln_score=3, notes="SQLite zeroing malloc",
    ),
    "lib|sqlite3DbMalloc": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="SQLite db-context malloc",
    ),
    "lib|sqlite3DbMallocZero": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="SQLite db-context zeroing malloc",
    ),
    "lib|sqlite3DbRealloc": _rule(
        return_is_buffer=True, size_arg=2,
        vuln_score=3, notes="SQLite db-context realloc",
    ),
    "lib|sqlite3_free": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="sqlite3_free",
    ),

    # ── Lua ───────────────────────────────────────────────────────────────────
    "lib|lua_tostring": _rule(
        return_tainted=True,
        vuln_score=3, notes="lua_tostring — Lua value as C string (user-controlled)",
    ),
    "lib|lua_tolstring": _rule(
        return_tainted=True,
        vuln_score=3, notes="lua_tolstring — with length output",
    ),
    "lib|lua_tointeger": _rule(
        return_tainted=True,
        vuln_score=2, notes="lua_tointeger — user integer from Lua",
    ),
    "lib|lua_tonumber": _rule(
        return_tainted=True,
        vuln_score=2, notes="lua_tonumber — user number from Lua",
    ),
    "lib|luaM_malloc": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="Lua internal malloc",
    ),
    "lib|luaM_realloc": _rule(
        return_is_buffer=True, size_arg=2,
        vuln_score=3, notes="Lua internal realloc",
    ),

    # ── OpenSSL ───────────────────────────────────────────────────────────────
    "lib|OPENSSL_malloc": _rule(
        return_is_buffer=True, size_arg=0,
        sink=True, sink_type="integer_overflow", taint_arg=0,
        vuln_score=3, notes="OpenSSL OPENSSL_malloc",
    ),
    "lib|OPENSSL_realloc": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="OpenSSL OPENSSL_realloc",
    ),
    "lib|OPENSSL_free": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="OpenSSL OPENSSL_free",
    ),
    "lib|CRYPTO_malloc": _rule(
        return_is_buffer=True, size_arg=0,
        sink=True, sink_type="integer_overflow", taint_arg=0,
        vuln_score=3, notes="OpenSSL CRYPTO_malloc",
    ),
    "lib|CRYPTO_realloc": _rule(
        return_is_buffer=True, size_arg=1,
        vuln_score=3, notes="OpenSSL CRYPTO_realloc",
    ),
    "lib|CRYPTO_free": _rule(
        frees_memory_at=0,
        vuln_score=2, notes="OpenSSL CRYPTO_free",
    ),

    # ── String length / comparison ────────────────────────────────────────────
    "lib|strlen": _rule(
        return_tainted=True,
        vuln_score=1, notes="strlen — return tainted when input tainted",
    ),
    "lib|strnlen": _rule(
        return_tainted=True, bounded=True, size_arg=1,
        vuln_score=1, notes="strnlen — bounded strlen",
    ),
    "lib|wcslen": _rule(
        return_tainted=True,
        vuln_score=1, notes="wcslen — wide char strlen",
    ),
    "lib|strcmp": _rule(
        return_tainted=False,
        vuln_score=1, notes="strcmp — comparison, return not tainted",
    ),
    "lib|strncmp": _rule(
        return_tainted=False, bounded=True, size_arg=2,
        vuln_score=1, notes="strncmp — bounded comparison",
    ),
    "lib|memcmp": _rule(
        return_tainted=False, bounded=True, size_arg=2,
        vuln_score=1, notes="memcmp",
    ),

    # ── File I/O ──────────────────────────────────────────────────────────────
    "lib|fopen": _rule(
        return_tainted=False, return_is_buffer=False,
        vuln_score=1, notes="fopen — file handle, not directly tainted",
    ),
    "lib|fclose": _rule(
        vuln_score=1, notes="fclose",
    ),
    "lib|fwrite": _rule(
        reads_from=0, bounded=True, size_arg=[1, 2],
        vuln_score=1, notes="fwrite(buf, size, count, stream) — writes from buf",
    ),
    "lib|write": _rule(
        reads_from=1, bounded=True, size_arg=2,
        vuln_score=1, notes="write(fd, buf, count) — writes from buf",
    ),
    "lib|send": _rule(
        reads_from=1, bounded=True, size_arg=2,
        vuln_score=1, notes="send(fd, buf, len, flags) — sends from buf",
    ),

    # ── Misc ──────────────────────────────────────────────────────────────────
    "lib|atoi": _rule(
        return_tainted=True,
        vuln_score=2, notes="atoi — string to int (no error checking)",
    ),
    "lib|atol": _rule(
        return_tainted=True,
        vuln_score=2, notes="atol — string to long",
    ),
    "lib|atoll": _rule(
        return_tainted=True,
        vuln_score=2, notes="atoll — string to long long",
    ),
    "lib|strtol": _rule(
        return_tainted=True,
        vuln_score=2, notes="strtol — string to long with base",
    ),
    "lib|strtoul": _rule(
        return_tainted=True,
        vuln_score=2, notes="strtoul — string to unsigned long",
    ),
    "lib|strtoll": _rule(
        return_tainted=True,
        vuln_score=2, notes="strtoll — string to long long",
    ),
    "lib|strtoull": _rule(
        return_tainted=True,
        vuln_score=2, notes="strtoull — string to unsigned long long",
    ),
    "lib|strtod": _rule(
        return_tainted=True,
        vuln_score=2, notes="strtod — string to double",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# PatternStore
# ─────────────────────────────────────────────────────────────────────────────

class PatternStore:
    """
    SQLite-backed store for vulnerability pattern rules.

    Thread-safe: uses check_same_thread=False with a lock for writes.
    """

    def __init__(self, db_path: str = "pattern_store.db"):
        self._db_path = db_path
        self._conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._lock    = threading.Lock()
        self._init_tables()
        self._seed_library_patterns()

    # ── Table initialization ──────────────────────────────────────────────────

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS library_patterns (
                pattern_key  TEXT PRIMARY KEY,
                fn_name      TEXT NOT NULL,
                rule_json    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS structural_patterns (
                pattern_key  TEXT PRIMARY KEY,
                fn_name      TEXT NOT NULL,
                arg_sizes    TEXT NOT NULL,
                ret_exists   INTEGER DEFAULT 0,
                ret_size     INTEGER DEFAULT 0,
                rule_json    TEXT NOT NULL,
                confidence   REAL    DEFAULT 0.7,
                vuln_score   REAL    DEFAULT 1.0,
                source       TEXT    DEFAULT 'llm_inferred',
                notes        TEXT    DEFAULT '',
                created_at   TEXT,
                hit_count    INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS fingerprint_patterns (
                fingerprint  TEXT PRIMARY KEY,
                confirmed    INTEGER DEFAULT 1,
                confidence   REAL    DEFAULT 0.7,
                vuln_type    TEXT,
                example_func TEXT,
                hit_count    INTEGER DEFAULT 1,
                created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._init_learned_table()
        self._init_learned_rules_table()
        self._conn.commit()

    def _init_learned_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                fn_name    TEXT NOT NULL,
                arg_sizes  TEXT NOT NULL,
                rule_json  TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (fn_name, arg_sizes)
            )
        """)
        self._conn.commit()

    def _init_learned_rules_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_rules (
                fn_name    TEXT NOT NULL,
                arg_sizes  TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT 'other',
                rule_json  TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (fn_name, arg_sizes)
            )
        """)
        # Task 6: function relationship graph.
        # Stores caller→callee edges with the callee's resolved role.
        # Enables graph-aware queries: "all functions 2 hops from a read_input"
        # without re-running BFS on raw P-code every binary.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS function_graph (
                caller_fn   TEXT NOT NULL,
                callee_fn   TEXT NOT NULL,
                callee_role TEXT NOT NULL DEFAULT 'other',
                confidence  REAL NOT NULL DEFAULT 0.5,
                seen_count  INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (caller_fn, callee_fn)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # Item 6: Auto Pattern Discovery — structural fingerprint table.
        # Stores (sha256_key, canonical_name, role, feature_vector, taint_rule)
        # for each function whose role has been confirmed (LLM or LIBRARY_MATCH).
        # On subsequent binaries, functions matching a stored fingerprint are
        # classified as STRUCTURAL_MATCH without an LLM call.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS structural_fingerprints (
                sha256_key     TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                role           TEXT NOT NULL DEFAULT 'unknown',
                feature_vector TEXT NOT NULL DEFAULT '[]',
                taint_rule     TEXT NOT NULL DEFAULT '{}',
                seen_count     INTEGER NOT NULL DEFAULT 1,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Item 5: Active Learning — FP suppression table.
        # When a user marks a finding as a false positive, the (fn_name, vuln_type,
        # sink_fn) triple is stored here. Before the reasoning agent confirms a
        # new finding, it checks this table to suppress known FPs.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fp_suppressions (
                fn_name   TEXT NOT NULL,
                vuln_type TEXT NOT NULL DEFAULT '',
                sink_fn   TEXT NOT NULL DEFAULT '',
                reason    TEXT NOT NULL DEFAULT '',
                rejected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                suppress_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (fn_name, vuln_type, sink_fn)
            )
        """)
        self._conn.commit()

    def set_metadata(self, key: str, value) -> None:
        """Persist a scalar value (str/int/float) across pipeline runs."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv_metadata (key, value) VALUES (?,?)",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_metadata(self, key: str, default=None):
        """Retrieve a previously stored metadata value, or default."""
        row = self._conn.execute(
            "SELECT value FROM kv_metadata WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    # ── Task 6: Function relationship graph ───────────────────────────────────

    def store_graph_edge(
        self,
        caller_fn:   str,
        callee_fn:   str,
        callee_role: str,
        confidence:  float = 0.5,
    ) -> None:
        """Store (or increment) a caller→callee edge with the callee's role."""
        with self._lock:
            self._conn.execute("""
                INSERT INTO function_graph (caller_fn, callee_fn, callee_role, confidence, seen_count)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(caller_fn, callee_fn) DO UPDATE SET
                    callee_role = excluded.callee_role,
                    confidence  = MAX(function_graph.confidence, excluded.confidence),
                    seen_count  = function_graph.seen_count + 1
            """, (caller_fn, callee_fn, callee_role, confidence))
            self._conn.commit()

    def get_callee_roles_for(self, caller_fn: str) -> dict[str, str]:
        """Return {callee_fn: role} for all known callees of caller_fn."""
        rows = self._conn.execute(
            "SELECT callee_fn, callee_role FROM function_graph WHERE caller_fn=?",
            (caller_fn,),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_graph_neighbors(
        self,
        fn_names: list[str],
        max_hops: int = 2,
    ) -> dict[str, dict[str, str]]:
        """
        Return {fn_name: {callee: role}} for all functions within max_hops of
        any function in fn_names, by following graph edges.
        Used to enrich Stage 2 ranking with cross-binary call graph knowledge.
        """
        result: dict[str, dict[str, str]] = {}
        frontier = set(fn_names)
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for fn in frontier:
                if fn not in result:
                    callees = self.get_callee_roles_for(fn)
                    if callees:
                        result[fn] = callees
                        next_frontier.update(callees.keys())
            frontier = next_frontier - set(result)
            if not frontier:
                break
        return result

    def get_graph_stats(self) -> dict:
        """Return summary stats for the function graph."""
        edges = self._conn.execute(
            "SELECT COUNT(*) FROM function_graph"
        ).fetchone()[0]
        nodes = self._conn.execute(
            "SELECT COUNT(DISTINCT caller_fn) FROM function_graph"
        ).fetchone()[0]
        return {"graph_edges": edges, "graph_nodes": nodes}

    # ── Item 6: Structural fingerprint store ──────────────────────────────────

    def store_structural_fingerprint(
        self,
        sha256_key:     str,
        canonical_name: str,
        role:           str,
        feature_vector: list,
        taint_rule:     dict,
    ) -> None:
        """Store or update a structural fingerprint."""
        with self._lock:
            self._conn.execute("""
                INSERT INTO structural_fingerprints
                    (sha256_key, canonical_name, role, feature_vector, taint_rule, seen_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(sha256_key) DO UPDATE SET
                    seen_count = structural_fingerprints.seen_count + 1,
                    role       = excluded.role,
                    taint_rule = excluded.taint_rule
            """, (
                sha256_key, canonical_name, role,
                json.dumps(feature_vector),
                json.dumps(taint_rule),
            ))
            self._conn.commit()
            log.debug("Fingerprint stored: %s (%s) → %s", canonical_name, sha256_key[:8], role)

    def get_structural_fingerprint(self, sha256_key: str) -> Optional[dict]:
        """Exact lookup by sha256_key."""
        row = self._conn.execute(
            "SELECT canonical_name, role, feature_vector, taint_rule "
            "FROM structural_fingerprints WHERE sha256_key=?",
            (sha256_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "canonical_name": row[0],
            "role":           row[1],
            "feature_vector": json.loads(row[2]),
            "taint_rule":     json.loads(row[3]),
        }

    def get_all_structural_fingerprints(self) -> list[dict]:
        """Return all stored fingerprints (for fuzzy matching sweep)."""
        rows = self._conn.execute(
            "SELECT sha256_key, canonical_name, role, feature_vector, taint_rule "
            "FROM structural_fingerprints"
        ).fetchall()
        return [
            {
                "sha256_key":     r[0],
                "canonical_name": r[1],
                "role":           r[2],
                "feature_vector": json.loads(r[3]),
                "taint_rule":     json.loads(r[4]),
            }
            for r in rows
        ]

    def get_fingerprint_stats(self) -> dict:
        """Return stats about the fingerprint store."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM structural_fingerprints"
        ).fetchone()[0]
        roles = self._conn.execute(
            "SELECT role, COUNT(*) FROM structural_fingerprints GROUP BY role"
        ).fetchall()
        return {
            "total_fingerprints": count,
            "by_role": {r[0]: r[1] for r in roles},
        }

    # ── Item 5: Active Learning — FP feedback ────────────────────────────────

    def record_fp_rejection(
        self,
        fn_name:   str,
        vuln_type: str = "",
        sink_fn:   str = "",
        reason:    str = "",
    ) -> None:
        """
        Record that a human reviewer marked a finding as a false positive.
        This (fn_name, vuln_type, sink_fn) triple is stored so future runs
        will suppress the same finding rather than confirming it again.

        The suppress_count tracks how many times this FP was generated,
        useful for prioritising which patterns to fix first.

        Call from CLI:
            python pipeline.py --reject-fp png_set_tRNS --vuln buffer_overflow
        """
        with self._lock:
            self._conn.execute("""
                INSERT INTO fp_suppressions (fn_name, vuln_type, sink_fn, reason, suppress_count)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(fn_name, vuln_type, sink_fn) DO UPDATE SET
                    suppress_count = fp_suppressions.suppress_count + 1,
                    reason = CASE WHEN excluded.reason != '' THEN excluded.reason
                             ELSE fp_suppressions.reason END
            """, (fn_name, vuln_type or "", sink_fn or "", reason or ""))
            self._conn.commit()
            log.info("FP rejection recorded: %s / %s / %s", fn_name, vuln_type, sink_fn)

    def is_fp_suppressed(
        self,
        fn_name:   str,
        vuln_type: str = "",
        sink_fn:   str = "",
    ) -> bool:
        """
        Return True if this (fn_name, vuln_type, sink_fn) combination has been
        previously marked as a false positive and should be suppressed.

        Matching logic (most-specific to least-specific):
          1. Exact match on all three fields.
          2. Match on fn_name + vuln_type only (any sink_fn).
          3. Match on fn_name only (any vuln_type + sink_fn).
        """
        # Exact match
        row = self._conn.execute(
            "SELECT 1 FROM fp_suppressions WHERE fn_name=? AND vuln_type=? AND sink_fn=?",
            (fn_name, vuln_type or "", sink_fn or ""),
        ).fetchone()
        if row:
            return True
        # fn_name + vuln_type match
        if vuln_type:
            row = self._conn.execute(
                "SELECT 1 FROM fp_suppressions WHERE fn_name=? AND vuln_type=?",
                (fn_name, vuln_type),
            ).fetchone()
            if row:
                return True
        # fn_name only
        row = self._conn.execute(
            "SELECT 1 FROM fp_suppressions WHERE fn_name=? AND vuln_type=''",
            (fn_name,),
        ).fetchone()
        return bool(row)

    def get_fp_suppressions(self) -> list[dict]:
        """Return all recorded FP rejections (for display / audit)."""
        rows = self._conn.execute(
            "SELECT fn_name, vuln_type, sink_fn, reason, suppress_count, rejected_at "
            "FROM fp_suppressions ORDER BY suppress_count DESC, rejected_at DESC"
        ).fetchall()
        return [
            {"fn_name": r[0], "vuln_type": r[1], "sink_fn": r[2],
             "reason": r[3], "suppress_count": r[4], "rejected_at": r[5]}
            for r in rows
        ]

    def clear_fp_suppression(
        self,
        fn_name:   str,
        vuln_type: str = "",
        sink_fn:   str = "",
    ) -> bool:
        """Remove a specific FP suppression (e.g., if the rule was a mistake)."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM fp_suppressions WHERE fn_name=? AND vuln_type=? AND sink_fn=?",
                (fn_name, vuln_type or "", sink_fn or ""),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def _seed_library_patterns(self) -> None:
        count = self._conn.execute(
            "SELECT COUNT(*) FROM library_patterns"
        ).fetchone()[0]
        if count > 0:
            return
        for key, rule in _LIBRARY_PATTERNS.items():
            parts   = key.split("|")
            fn_name = parts[1] if len(parts) >= 2 else parts[0]
            self._conn.execute(
                "INSERT OR REPLACE INTO library_patterns (pattern_key, fn_name, rule_json) "
                "VALUES (?,?,?)",
                (key, fn_name, json.dumps(rule)),
            )
        self._conn.commit()
        log.debug("PatternStore: seeded %d library patterns", len(_LIBRARY_PATTERNS))

    # ── Tier 1: Library lookup ────────────────────────────────────────────────

    def lookup(self, fn_name: str, _arg_sizes: list) -> Optional[dict]:
        """Look up a rule by exact function name. Returns rule dict or None."""
        row = self._conn.execute(
            "SELECT rule_json FROM library_patterns WHERE fn_name=?",
            (fn_name,),
        ).fetchone()
        if row:
            rule = json.loads(row[0])
            rule["source"] = "hardcoded"
            return rule
        return None

    # ── Tier 2: Structural lookup ─────────────────────────────────────────────

    def lookup_by_pattern(self, arg_sizes: list, output: Optional[dict]) -> Optional[dict]:
        """
        Look up a structural rule by argument-size signature + return shape.
        Used for stripped binaries where the function name is an address.
        """
        sizes_key  = json.dumps(sorted(arg_sizes) if arg_sizes else [])
        ret_exists = 1 if output else 0

        row = self._conn.execute(
            """SELECT rule_json, fn_name, confidence FROM structural_patterns
               WHERE arg_sizes=? AND ret_exists=?
               ORDER BY confidence DESC, hit_count DESC LIMIT 1""",
            (sizes_key, ret_exists),
        ).fetchone()

        if row:
            rule = json.loads(row[0])
            rule["fn_name"]    = row[1]
            rule["confidence"] = row[2]
            rule["source"]     = "llm_inferred"
            with self._lock:
                self._conn.execute(
                    "UPDATE structural_patterns SET hit_count=hit_count+1 "
                    "WHERE arg_sizes=? AND ret_exists=?",
                    (sizes_key, ret_exists),
                )
                self._conn.commit()
            return rule
        return None

    def store_structural(
        self,
        fn_name:    str,
        arg_sizes:  list,
        rule:       dict,
        confidence: float         = 0.7,
        notes:      str           = "",
        output:     Optional[dict] = None,
    ) -> None:
        """Store an LLM-inferred structural pattern keyed by arg-size signature."""
        sizes_key   = json.dumps(sorted(arg_sizes) if arg_sizes else [])
        ret_exists  = 1 if output else 0
        ret_size    = output.get("size", 0) if isinstance(output, dict) else 0
        pattern_key = f"{sizes_key}|{ret_exists}|{ret_size}|{fn_name}"

        rule_copy               = dict(rule)
        rule_copy["confidence"] = confidence
        rule_copy["notes"]      = notes

        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO structural_patterns
                   (pattern_key, fn_name, arg_sizes, ret_exists, ret_size,
                    rule_json, confidence, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,datetime('now'))""",
                (pattern_key, fn_name, sizes_key, ret_exists, ret_size,
                 json.dumps(rule_copy), confidence, notes),
            )
            self._conn.commit()

    # ── Stage 2.5: Learned summaries and rules ────────────────────────────────

    def get_learned_summary(self, fn_name: str, arg_sizes: list) -> Optional[dict]:
        """Return the cached semantic summary for fn_name, or None."""
        sizes_key = json.dumps(arg_sizes if arg_sizes else [])
        row = self._conn.execute(
            "SELECT rule_json FROM learned_patterns WHERE fn_name=? AND arg_sizes=?",
            (fn_name, sizes_key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def store_learned_summary(self, fn_name: str, arg_sizes: list, summary: dict) -> None:
        """Cache a semantic summary produced by SemanticRecoveryAgent."""
        sizes_key = json.dumps(arg_sizes if arg_sizes else [])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO learned_patterns (fn_name, arg_sizes, rule_json) "
                "VALUES (?,?,?)",
                (fn_name, sizes_key, json.dumps(summary)),
            )
            self._conn.commit()

    def get_learned_rule(self, fn_name: str, arg_sizes: list) -> Optional[dict]:
        """Return the taint rule derived from a Stage 2.5 summary, or None."""
        sizes_key = json.dumps(arg_sizes if arg_sizes else [])
        row = self._conn.execute(
            "SELECT rule_json, role FROM learned_rules WHERE fn_name=? AND arg_sizes=?",
            (fn_name, sizes_key),
        ).fetchone()
        if row:
            rule          = json.loads(row[0])
            rule["role"]   = row[1]
            rule["source"] = "learned"
            return rule
        return None

    def store_learned_rule(
        self,
        fn_name:   str,
        arg_sizes: list,
        rule:      dict,
        role:      str = "other",
    ) -> None:
        """Store a taint rule derived from a Stage 2.5 semantic summary."""
        sizes_key = json.dumps(arg_sizes if arg_sizes else [])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO learned_rules (fn_name, arg_sizes, role, rule_json) "
                "VALUES (?,?,?,?)",
                (fn_name, sizes_key, role, json.dumps(rule)),
            )
            self._conn.commit()

    # ── Fingerprint store ─────────────────────────────────────────────────────

    def store_fingerprint(
        self,
        fingerprint:  str,
        vuln_type:    str,
        example_func: str,
        confidence:   float = 0.7,
    ) -> None:
        """Record a confirmed vuln-shape fingerprint for cross-binary matching."""
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO fingerprint_patterns
                   (fingerprint, vuln_type, example_func, confidence)
                   VALUES (?,?,?,?)""",
                (fingerprint, vuln_type, example_func, confidence),
            )
            self._conn.execute(
                "UPDATE fingerprint_patterns SET hit_count=hit_count+1 "
                "WHERE fingerprint=?",
                (fingerprint,),
            )
            self._conn.commit()

    def lookup_fingerprint(self, fingerprint: str) -> Optional[dict]:
        """Return fingerprint record or None."""
        row = self._conn.execute(
            "SELECT vuln_type, confidence, hit_count FROM fingerprint_patterns "
            "WHERE fingerprint=? AND confirmed=1",
            (fingerprint,),
        ).fetchone()
        if row:
            return {"vuln_type": row[0], "confidence": row[1], "hit_count": row[2]}
        return None

    def get_all_callee_roles(self) -> dict:
        """Return {fn_name: role} for all learned rules (ignores arg_sizes)."""
        rows = self._conn.execute(
            "SELECT fn_name, role FROM learned_rules"
        ).fetchall()
        return {fn_name: role for fn_name, role in rows}

    def get_stats(self) -> dict:
        """Return row counts for each table — used by diagnostic scripts."""
        tables = ["library_patterns", "structural_patterns",
                  "fingerprint_patterns", "learned_patterns", "learned_rules"]
        return {
            t: self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }

    # ── Misc ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
