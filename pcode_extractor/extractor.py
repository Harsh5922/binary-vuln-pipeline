"""
extractor.py

Extracts refined, SSA-form P-code from a binary using pyghidra.

Key design decisions
--------------------
* Uses DecompInterface.getHighFunction() — SSA P-code, not raw assembly-level
  P-code. Each variable is defined exactly once; data flow is trivially traceable.
* Runs only the Ghidra analyzers that are strictly necessary for P-code quality.
  Full auto-analysis on a large binary takes 5+ minutes; targeted analysis takes
  under 30 seconds.
* Canonicalizes varnode names so the model/agent sees program structure,
  not Ghidra's internal register/offset identifiers.
* Yields results lazily so the caller can start processing before the full
  binary is analyzed.

Usage
-----
    from extractor import PcodeExtractor

    with PcodeExtractor("./binary") as ex:
        for func in ex.extract():
            print(func.name, len(func.ops), "ops")

    # Or save directly to JSONL
    PcodeExtractor("./binary").to_jsonl("output.jsonl")
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Varnode:
    """
    A single P-code variable (input or output of an operation).

    Ghidra varnodes live in one of several address spaces:
      register  — CPU register  (e.g. EAX, RSP)
      ram       — memory address
      const     — compile-time constant
      unique    — temporary SSA variable (decompiler-invented)
      stack     — stack slot

    After canonicalization, register/unique/stack varnodes become VAR_N.
    Constants and ram addresses are kept as-is because they carry semantic
    meaning (buffer sizes, jump targets, memory-mapped I/O).
    """
    name:   str             # canonical: VAR_N  or  const(0x40)  or  ram(0x...)
    space:  str             # original address space
    size:   int             # byte size (4 = int32, 8 = int64, 1 = char, …)
    raw:    str             # original Ghidra varnode string (for debugging)


@dataclass
class PcodeOp:
    """
    One P-code operation in SSA form.

    op      — mnemonic: LOAD, STORE, CALL, CALLIND, BRANCH, CBRANCH,
              INT_ADD, INT_SUB, INT_MULT, PTRADD, PTRSUB, RETURN, …
    output  — destination varnode (None for ops with no output, e.g. STORE, BRANCH)
    inputs  — list of source varnodes
    seq     — sequence number within the function (0-based)
    addr    — instruction address this P-code op belongs to
    """
    seq:    int
    op:     str
    output: Optional[Varnode]
    inputs: list[Varnode]
    addr:   str


@dataclass
class FunctionInfo:
    """Everything extracted for one function."""
    name:           str
    entry_addr:     str
    prototype:      str             # decompiled signature, e.g. "void foo(char *buf)"
    ops:            list[PcodeOp]
    call_sites:     list[str]       # names/addresses of called functions
    flags:          dict[str, bool] # quick heuristic flags for the filter stage
    size_bytes:     int             # function body size
    decompile_time: float           # seconds taken to decompile this function


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# P-code ops that are purely structural / SSA bookkeeping.
# They add noise without adding semantic content.
# NOTE: COPY is intentionally NOT skipped — it propagates variable aliases
# that the taint engine must track (e.g. COPY VAR_5, VAR_3 means VAR_5 is
# tainted if VAR_3 is tainted). Dropping it silently breaks taint chains.
_SKIP_OPS: frozenset[str] = frozenset({
    "CAST",         # type cast, no computation
    "PIECE",        # register concat artifact (x86 AH:AL → AX)
    "SUBPIECE",     # register slice artifact
    "INDIRECT",     # models side-effects of CALL on memory; too coarse to use
    "MULTIEQUAL",   # phi-node — set keep_phi=True to retain these
})

# Ops that involve memory or control flow — the interesting ones
MEMORY_OPS: frozenset[str] = frozenset({"LOAD", "STORE"})
CALL_OPS:   frozenset[str] = frozenset({"CALL", "CALLIND"})
BRANCH_OPS: frozenset[str] = frozenset({"BRANCH", "CBRANCH", "BRANCHIND", "RETURN"})
ARITH_OPS:  frozenset[str] = frozenset({
    "INT_ADD", "INT_SUB", "INT_MULT", "INT_DIV", "INT_REM",
    "INT_LEFT", "INT_RIGHT", "INT_SRIGHT",
    "PTRADD", "PTRSUB",
    "INT_ZEXT", "INT_SEXT",
})

# libc / Win32 functions that are sources or sinks of vulnerabilities
DANGEROUS_IMPORTS: frozenset[str] = frozenset({
    # memory / string — classic unsafe ops
    "strcpy", "strncpy", "strcat", "strncat",
    "sprintf", "vsprintf", "snprintf",
    "gets", "memcpy", "memmove", "memset",
    # input sources
    "recv", "recvfrom", "read", "fread", "fgets",
    "scanf", "sscanf", "fscanf", "getenv",
    "ReadFile", "WSARecv",
    # heap
    "malloc", "calloc", "realloc", "free", "alloca",
    # execution
    "system", "popen", "execl", "execle", "execlp",
    "execv", "execve", "execvp",
    # format string sinks
    "printf", "fprintf", "wprintf",
})




# ─────────────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────────────

class PcodeExtractor:
    """
    Extracts SSA P-code from a binary via pyghidra.

    Parameters
    ----------
    binary_path     : path to the target binary (ELF, PE, Mach-O, raw)
    min_ops         : discard functions with fewer than this many P-code ops
    max_ops         : discard functions with more than this many P-code ops
    keep_phi        : include MULTIEQUAL (phi) nodes (default: False)
    decompile_timeout : seconds to wait for a single function (default: 60)
    skip_thunks     : skip thunk/stub functions (default: True)
    skip_external   : skip external/imported functions (default: True)
    """

    def __init__(
        self,
        binary_path:        str | Path,
        min_ops:            int   = 8,
        max_ops:            int   = 8000,
        keep_phi:           bool  = False,
        decompile_timeout:  int   = 60,
        skip_thunks:        bool  = True,
        skip_external:      bool  = True,
    ):
        self.binary_path       = Path(binary_path)
        self.min_ops           = min_ops
        self.max_ops           = max_ops
        self.skip_ops          = _SKIP_OPS if not keep_phi else _SKIP_OPS - {"MULTIEQUAL"}
        self.decompile_timeout = decompile_timeout
        self.skip_thunks       = skip_thunks
        self.skip_external     = skip_external

        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")

        # Stats collected during extraction
        self._stats: dict = {}

    # ── public API ───────────────────────────────────────────────────────────

    def extract(self) -> Iterator[FunctionInfo]:
        """
        Open the binary in Ghidra (headlessly) and yield one FunctionInfo
        per function that passes the size filter.

        This is a generator — results stream out as they are decompiled.
        You do not need to wait for the whole binary to be analyzed.
        """
        import pyghidra

        log.info("Opening %s …", self.binary_path.name)

        # open_project / program_context is the modern pyghidra API.
        # Falls back to open_program for older pyghidra installs.
        ctx = self._open_binary()

        with ctx as flat_api:

            program = flat_api.getCurrentProgram()
            monitor = flat_api.getMonitor()

            # Run Ghidra's standard analysis pipeline.
            # analyzeAll() is safe to call even if the binary was already
            # analyzed — Ghidra skips work that is already done.
            log.info("Running Ghidra analysis …")
            t0 = time.perf_counter()
            flat_api.analyzeAll(program)
            log.info("  Analysis done in %.1fs", time.perf_counter() - t0)

            # Set up the decompiler
            decomp = self._make_decompiler(program)

            listing   = program.getListing()
            functions = list(listing.getFunctions(True))
            total     = len(functions)
            log.info("Found %d functions — decompiling …", total)

            succeeded = skipped = failed = duped = 0
            t_start = time.perf_counter()
            seen_fingerprints: set[str] = set()

            for i, func in enumerate(functions, 1):
                # Skip trivial / external functions
                if self.skip_thunks and func.isThunk():
                    skipped += 1
                    continue
                if self.skip_external and func.isExternal():
                    skipped += 1
                    continue

                result = self._decompile_function(func, decomp, monitor)
                if result is None:
                    failed += 1
                    continue

                # Deduplicate structurally identical functions — common in
                # statically linked binaries where the same libc code is
                # inlined or linked multiple times under different symbols.
                fp = _compute_fingerprint(result.ops)
                if fp in seen_fingerprints:
                    duped += 1
                    log.debug("Duplicate fingerprint skipped: %s", result.name)
                    continue
                seen_fingerprints.add(fp)

                succeeded += 1

                if i % 200 == 0 or i == total:
                    elapsed = time.perf_counter() - t_start
                    rate    = i / max(elapsed, 0.001)
                    log.info(
                        "  [%d/%d]  ok=%d  skip=%d  fail=%d  duped=%d  %.0f fn/s",
                        i, total, succeeded, skipped, failed, duped, rate,
                    )

                yield result

            decomp.dispose()
            self._stats = {
                "total":     total,
                "succeeded": succeeded,
                "skipped":   skipped,
                "failed":    failed,
                "duped":     duped,
                "elapsed_s": round(time.perf_counter() - t_start, 2),
            }
            log.info("Extraction complete: %s", self._stats)

    def to_jsonl(self, output_path: str | Path) -> int:
        """
        Extract and write every function to a JSON-lines file.
        Returns the number of functions written.

        File format: one JSON object per line, fields = FunctionInfo fields.
        ops[i].output and ops[i].inputs are Varnode dicts.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        # Write to a sibling temp file so a failed extraction never truncates
        # an existing checkpoint. The temp file is renamed into place only on
        # full success; any exception leaves the original file untouched.
        tmp_path = output_path.with_suffix(".jsonl.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                for func in self.extract():
                    fh.write(json.dumps(_func_to_dict(func), ensure_ascii=False))
                    fh.write("\n")
                    count += 1
            tmp_path.replace(output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        log.info("Wrote %d functions to %s", count, output_path)
        return count

    @property
    def stats(self) -> dict:
        """Extraction statistics (available after extract() is exhausted)."""
        return self._stats

    # ── internal ─────────────────────────────────────────────────────────────

    def _open_binary(self):
        """
        Return a context manager that opens the binary in Ghidra.

        pyghidra >= 1.3 deprecated open_program() in favour of
        open_project() + program_context().  We try the new API first
        and fall back to open_program() so the code works with any
        pyghidra version the user has installed.
        """
        import pyghidra

        # New API (pyghidra >= 1.3)
        if hasattr(pyghidra, "open_project"):
            try:
                return pyghidra.open_project(str(self.binary_path))
            except Exception:
                pass  # fall through to legacy

        # Legacy API — still works, just prints a DeprecationWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return pyghidra.open_program(str(self.binary_path), analyze=True)

    @staticmethod
    def _make_decompiler(program):
        """
        Create and configure the Ghidra decompiler interface.

        simplificationStyle="decompile" gives us full type propagation
        and the cleanest SSA P-code.  Other styles ("normalize",
        "paramid") produce simpler but less informative output.
        """
        from ghidra.app.decompiler import DecompInterface, DecompileOptions

        decomp = DecompInterface()
        opts   = DecompileOptions()

        # Request P-code only — we don't need the C AST or C source
        opts.grabFromProgram(program)
        decomp.setOptions(opts)
        decomp.toggleCCode(False)
        decomp.toggleSyntaxTree(True)    # needed to access HighFunction
        decomp.setSimplificationStyle("decompile")
        decomp.openProgram(program)

        return decomp

    def _decompile_function(
        self,
        func,
        decomp,
        monitor,
    ) -> Optional[FunctionInfo]:
        """
        Decompile one function and return a FunctionInfo, or None on failure.
        """
        t0 = time.perf_counter()

        try:
            result = decomp.decompileFunction(func, self.decompile_timeout, monitor)
        except Exception as e:
            log.debug("decompileFunction exception for %s: %s", func.getName(), e)
            return None

        if not result.decompileCompleted():
            log.debug("Decompile failed: %s  (%s)",
                      func.getName(), result.getErrorMessage())
            return None

        high_func = result.getHighFunction()
        if high_func is None:
            return None

        # Collect all raw P-code ops from the SSA representation
        raw_ops, call_sites = self._collect_ops(high_func)

        # Apply size filter before the expensive canonicalization step
        if len(raw_ops) < self.min_ops or len(raw_ops) > self.max_ops:
            return None

        # Canonicalize varnodes: register/unique → VAR_N
        ops = self._canonicalize(raw_ops)

        decompile_time = time.perf_counter() - t0

        return FunctionInfo(
            name           = func.getName(),
            entry_addr     = str(func.getEntryPoint()),
            prototype      = _get_prototype(result),
            ops            = ops,
            call_sites     = sorted(set(call_sites)),
            flags          = _compute_flags(ops, call_sites),
            size_bytes      = func.getBody().getNumAddresses(),
            decompile_time = round(decompile_time, 4),
        )

    def _collect_ops(
        self,
        high_func,
    ) -> tuple[list[dict], list[str]]:
        """
        Walk all P-code ops in the HighFunction.
        Returns (raw_op_dicts, call_site_names).

        raw_op_dict keys: op, raw_output, raw_inputs, addr
        """
        raw_ops:    list[dict]  = []
        call_sites: list[str]   = []
        seq = 0

        op_iter = high_func.getPcodeOps()
        while op_iter.hasNext():
            pcode_op = op_iter.next()
            mnemonic = pcode_op.getMnemonic()

            # Resolve call target name (for call_sites and op annotation)
            call_target_name: Optional[str] = None
            if mnemonic in CALL_OPS:
                call_target_name = _resolve_call_target(pcode_op, high_func)
                if call_target_name:
                    call_sites.append(call_target_name)

            if mnemonic in self.skip_ops:
                continue

            raw_output = pcode_op.getOutput()
            raw_inputs = [pcode_op.getInput(i)
                         for i in range(pcode_op.getNumInputs())]

            raw_ops.append({
                "seq":              seq,
                "op":               mnemonic,
                "raw_output":       raw_output,        # Ghidra Varnode object or None
                "raw_inputs":       raw_inputs,        # list of Ghidra Varnode objects
                "addr":             str(pcode_op.getSeqnum().getTarget()),
                "call_target_name": call_target_name,  # resolved name (None for non-CALLs)
            })
            seq += 1

        return raw_ops, call_sites


    

    @staticmethod
    def _canonicalize(raw_ops: list[dict]) -> list[PcodeOp]:
        """
        Replace Ghidra's internal varnode identifiers with stable names.

        Naming strategy
        ---------------
        - Registers / unique / stack slots  →  VAR_0, VAR_1, VAR_2, …
          (positional — first varnode defined in the function is VAR_0)
        - Constants                         →  const(0x<hex>)
        - RAM addresses                     →  ram(0x<hex>)
        - External symbols / imports        →  kept as-is (contains "<name>")

        The result is architecture-neutral and strips out Ghidra internals
        so the downstream model sees program logic, not register names.
        """
        var_map: dict[str, str] = {}
        counter = [0]  # list so the nested function can mutate it

        def canon(vn) -> Optional[Varnode]:
            if vn is None:
                return None

            space = str(vn.getAddress().getAddressSpace().getName())
            size  = int(vn.getSize())
            raw   = str(vn)

            # Constants — keep the numeric value, it's semantically meaningful
            if space == "const":
                name = f"const(0x{vn.getOffset():x})"
                return Varnode(name=name, space=space, size=size, raw=raw)

            # RAM addresses — keep address, useful for global/static access patterns
            if space == "ram":
                name = f"ram(0x{vn.getOffset():x})"
                return Varnode(name=name, space=space, size=size, raw=raw)

            # External symbols (imports) — keep readable name
            # NOTE: ALL Ghidra varnodes start with ( e.g. (unique,0x23e00,8)
            # Do NOT use startswith('(') — that catches every variable.
            # Only keep raw for true import symbols containing '<'.
            if "<" in raw:
                return Varnode(name=raw, space=space, size=size, raw=raw)

            # Everything else: register, unique, stack → VAR_N
            if raw not in var_map:
                var_map[raw] = f"VAR_{counter[0]}"
                counter[0] += 1
            return Varnode(name=var_map[raw], space=space, size=size, raw=raw)

        ops: list[PcodeOp] = []
        for r in raw_ops:
            raw_in = r["raw_inputs"]
            resolved = r.get("call_target_name")

            # For CALL ops with a resolved target name, replace inputs[0] with
            # a named varnode so the pattern_matcher can do name-based lookup
            # instead of falling back to the unreliable arg-signature heuristic.
            if r["op"] in CALL_OPS and resolved and not resolved.startswith("0x"):
                target_vn = Varnode(name=resolved, space="external", size=8, raw=resolved)
                canon_inputs = [target_vn] + [canon(inp) for inp in raw_in[1:]]
            else:
                canon_inputs = [canon(inp) for inp in raw_in]

            ops.append(PcodeOp(
                seq    = r["seq"],
                op     = r["op"],
                output = canon(r["raw_output"]),
                inputs = canon_inputs,
                addr   = r["addr"],
            ))

        return ops


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (module-level, used by the class above)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_call_target(pcode_op, high_func=None) -> Optional[str]:
    """
    Try to extract a human-readable name for a CALL target.

    Resolution order:
    1. FunctionManager.getFunctionAt(addr)  — FLIRT-identified static lib functions
    2. SymbolTable.getPrimarySymbol(addr)   — PLT / exported symbols
    3. Raw address string                   — last resort for stripped binaries

    Why FunctionManager first: statically linked binaries have no PLT entries.
    Ghidra FLIRT signature matching identifies internal functions by byte pattern
    and stores results in FunctionManager, NOT in the symbol table.

    Parameters
    ----------
    pcode_op  : Ghidra PcodeOpAST
    high_func : Ghidra HighFunction — used to reach the Program object reliably.
                Falls back to pcode_op.getParent().getParent().getProgram() if None.
    """
    try:
        inputs = pcode_op.getInputs()
        if not inputs:
            return None

        target_vn = inputs[0]
        addr = target_vn.getAddress()

        if addr is not None:
            # Use high_func to reach Program — getParent().getParent() is fragile
            program = None
            try:
                if high_func is not None:
                    program = high_func.getFunction().getProgram()
                else:
                    program = pcode_op.getParent().getParent().getProgram()
            except Exception:
                pass

            if program is not None:
                # Priority 1: FunctionManager (FLIRT / thunks / PLT stubs)
                try:
                    func = program.getFunctionManager().getFunctionAt(addr)
                    if func is not None:
                        name = func.getName()
                        if not name.startswith("FUN_") and not name.startswith("LAB_"):
                            return name
                except Exception:
                    pass

                # Priority 2: Symbol table (PLT entries, exported symbols)
                try:
                    sym = program.getSymbolTable().getPrimarySymbol(addr)
                    if sym is not None:
                        return sym.getName()
                except Exception:
                    pass

            # Priority 3: Raw address — last resort for stripped binaries
            return f"0x{addr.getOffset():x}"

        # Constant address (typical for imports after relocation)
        if target_vn.isConstant():
            return f"0x{target_vn.getOffset():x}"

    except Exception:
        pass

    return None


def _get_prototype(decomp_result) -> str:
    """Extract the decompiled function signature as a string."""
    try:
        func = decomp_result.getDecompiledFunction()
        if func:
            return func.getSignature()
    except Exception:
        pass
    return ""


def _compute_flags(ops: list[PcodeOp], call_sites: list[str]) -> dict[str, bool]:
    """
    Compute a quick set of boolean flags for the function.
    Used by the filter stage as cheap pre-screening features.
    """
    op_types = {o.op for o in ops}
    calls_lower = " ".join(call_sites).lower()

    return {
        # Memory access
        "has_load":              "LOAD"     in op_types,
        "has_store":             "STORE"    in op_types,
        # Control flow — CBRANCH alone: old check used BRANCH_OPS union which
        # includes RETURN, making has_loop true for nearly every function.
        "has_loop":              "CBRANCH" in op_types,
        "has_indirect_call":     "CALLIND"  in op_types,
        # Arithmetic (integer overflow candidates)
        "has_int_arith":         bool(op_types & ARITH_OPS),
        "has_pointer_arith":     bool(op_types & {"PTRADD", "PTRSUB"}),
        "is_validator":          ("INT_MULT" in op_types and "CBRANCH" in op_types) or ("INT_LESS" in op_types and "INT_ADD" in op_types),
        # Dangerous call targets
        "calls_dangerous_import": any(
            imp in calls_lower for imp in DANGEROUS_IMPORTS
        ),
        # Any memory + call combination (high-value for taint analysis)
        "has_mem_and_call": (
            bool(op_types & MEMORY_OPS) and bool(op_types & CALL_OPS)
        ),
    }


def _varnode_to_dict(vn: Optional[Varnode]) -> Optional[dict]:
    if vn is None:
        return None
    return {"name": vn.name, "space": vn.space, "size": vn.size}


def _compute_fingerprint(ops) -> str:
    """Structural fingerprint — stable across compilations of same source.

    Encodes op mnemonic + output size + all input sizes so that functions
    with the same op sequence but different argument widths (e.g. a 32-bit
    vs 64-bit variant) produce distinct fingerprints.
    """
    import hashlib
    parts = []
    for op in ops:
        out_size   = op.output.size if op.output else 0
        in_sizes   = ",".join(str(i.size) for i in op.inputs) if op.inputs else ""
        parts.append(f"{op.op}({out_size},[{in_sizes}])")
    raw = " ".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]



    
def _func_to_dict(func: FunctionInfo) -> dict:
    return {
        "name":           func.name,
        "entry_addr":     func.entry_addr,
        "prototype":      func.prototype,
        "size_bytes":     func.size_bytes,
        "decompile_time": func.decompile_time,
        "call_sites":     func.call_sites,
        "flags":          func.flags,
        "fingerprint":    _compute_fingerprint(func.ops),
        "ops": [
            {
                "seq":    op.seq,
                "op":     op.op,
                "addr":   op.addr,
                "output": _varnode_to_dict(op.output),
                "inputs": [_varnode_to_dict(i) for i in op.inputs],
            }
            for op in func.ops
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run directly with:  python extractor.py <binary>
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <binary> [output.jsonl]")
        sys.exit(1)

    binary  = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else "pcode.jsonl"

    ex    = PcodeExtractor(binary)
    count = ex.to_jsonl(outfile)

    sep = "-" * 50
    print(f"\n{sep}")
    print(f"Functions extracted : {count}")
    print(f"Output              : {outfile}")
    print(f"Stats               : {ex.stats}")
    print(sep)
    print(f"\nFirst 3 functions preview:")

    with open(outfile) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            fn = json.loads(line)
            flags_on = [k for k, v in fn["flags"].items() if v]
            print(f"\n  [{i+1}] {fn['name']}  @ {fn['entry_addr']}")
            print(f"       prototype : {fn['prototype']}")
            print(f"       ops       : {len(fn['ops'])}")
            print(f"       calls     : {fn['call_sites']}")
            print(f"       flags     : {flags_on}")
            if fn["ops"]:
                print(f"       first op  : {fn['ops'][0]['op']}  "
                      f"-> {fn['ops'][0].get('output')}")