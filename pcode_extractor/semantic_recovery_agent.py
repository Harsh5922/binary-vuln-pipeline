"""
semantic_recovery_agent.py — OpenRouter only (free, no daily cap)

Get key: openrouter.ai → Keys → Create Key
Set:     set OPENROUTER_API_KEY=sk-or-your-key
"""
from __future__ import annotations
import json, logging, os, time, urllib.request, urllib.error
from typing import Optional

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a binary reverse engineering expert.
Given decompiled P-code for an unknown function, return a JSON semantic summary.
Respond with ONLY valid JSON — no markdown, no explanation outside the JSON."""

_USER_TEMPLATE = """Unknown function: {fn_name}
Arguments: {arg_count} args, sizes = {arg_sizes}

P-code operations (first {op_count} shown):
(CALL instructions annotate callee role in brackets, e.g. malloc[allocator])
{ops_text}

Return JSON with these exact fields:
{{
  "likely_role":          "<allocator|copy|read_input|format_string|exec|free|logger|validator|other>",
  "confidence":           <0.0-1.0>,
  "writes_memory":        <true|false>,
  "reads_memory":         <true|false>,
  "allocation":           <true if returns newly allocated buffer>,
  "network_input":        <true if reads from socket>,
  "file_input":           <true if reads from file>,
  "return_tainted":       <true if return value carries external data>,
  "size_arg":             <index of size argument or -1>,
  "writes_memory_at":     <index of output pointer arg or -1>,
  "external_input":       [<arg indices that receive external data>],
  "marks_checked_args":   [<arg indices this function bounds-checks or validates — downstream use of these args is safe>],
  "external_source_args": [<arg indices this function fills with external/attacker-controlled data>],
  "reasoning":            "<one sentence explaining role and taint effects>"
}}

IMPORTANT for marks_checked_args: list any arg index that this function validates or
bounds-checks. Example: if the function compares arg[1] against a maximum and returns
an error if exceeded, include 1. The taint engine uses this to suppress false positives
on subsequent operations using that arg.

IMPORTANT for external_source_args: list any arg index that this function writes
external/network/file data into. Example: if arg[0] is a buffer filled from a socket,
include 0. The taint engine uses this to propagate external taint to subsequent operations."""

# Known callee roles used to annotate CALL ops in the LLM prompt.
_KNOWN_ROLES: dict[str, str] = {
    # allocators
    "malloc": "allocator", "calloc": "allocator", "realloc": "allocator",
    "png_malloc": "allocator", "png_calloc": "allocator",
    "png_malloc_warn": "allocator", "png_malloc_base": "allocator",
    "png_realloc_array": "allocator", "png_malloc_array": "allocator",
    "png_malloc_array_checked": "allocator", "png_zalloc": "allocator",
    "g_malloc": "allocator", "g_malloc0": "allocator",
    "av_malloc": "allocator", "av_mallocz": "allocator",
    # free
    "free": "free", "png_free": "free", "g_free": "free", "av_free": "free",
    # copy / move
    "memcpy": "copy", "memmove": "copy", "memset": "copy",
    "strcpy": "copy", "strncpy": "copy", "strcat": "copy",
    # input readers
    "fread": "read_input", "read": "read_input", "recv": "read_input",
    "fgets": "read_input", "gets": "read_input",
    # format parsers
    "scanf": "format_string", "sscanf": "format_string",
    "fscanf": "format_string", "sprintf": "format_string",
    "snprintf": "format_string",
    # validators / comparators
    "strlen": "validator", "strnlen": "validator",
    "strcmp": "validator", "strncmp": "validator",
    "memcmp": "validator",
    # error / log handlers
    "png_error": "logger", "png_warning": "logger", "png_chunk_error": "logger",
    "fprintf": "logger", "printf": "logger", "perror": "logger",
    # control flow
    "longjmp": "exec", "setjmp": "exec",
}

_FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",     # 120B, 1M ctx, less crowded
    "openai/gpt-oss-120b:free",                   # 120B GPT, high capability
    "qwen/qwen3-coder:free",                      # coder model, 1M ctx
    "nousresearch/hermes-3-llama-3.1-405b:free",  # 405B, less popular = less 429
    "meta-llama/llama-3.3-70b-instruct:free",     # fallback (popular, may 429)
]

def _render_ops(ops: list[dict], max_ops: int = 40,
                callee_roles: dict | None = None) -> str:
    roles = callee_roles or {}
    lines = []
    for op in ops[:max_ops]:
        seq  = op.get("seq", "?")
        typ  = op.get("op", "?")
        out  = op.get("output")
        ins  = op.get("inputs") or []
        os_  = out.get("name","?") if isinstance(out,dict) else "-"
        is_raw = [i.get("name","?") if isinstance(i,dict) else str(i) for i in ins[:4]]
        # Annotate callee with role on CALL/CALLIND instructions
        if typ in ("CALL", "CALLIND") and is_raw:
            callee = is_raw[0]
            role   = roles.get(callee) or _KNOWN_ROLES.get(callee)
            if role and role != "other":
                is_raw[0] = f"{callee}[{role}]"
        lines.append(f"  seq={seq:>4}  {typ:<14}  out={os_:<12}  in=[{', '.join(is_raw)}]")
    if len(ops) > max_ops:
        lines.append(f"  ... ({len(ops)-max_ops} more ops)")
    return "\n".join(lines)


class SemanticRecoveryAgent:
    """
    Stage 2.5 semantic recovery — always uses OpenRouter free models.

    Stage 2.5 is intentionally provider-agnostic from the pipeline's perspective:
    it uses OpenRouter regardless of --provider, because OpenRouter offers free
    high-quality models with no daily cap. The main pipeline provider (groq/gemini)
    controls Stage 4 only.
    """

    def __init__(self, provider="openrouter", model="", api_key=None,
                 delay_s=3.0, max_ops=40):
        self.delay_s  = delay_s
        self.max_ops  = max_ops
        self.model    = model  # overrides free model list if set
        self.provider = provider  # stored for logging only; Stage 2.5 always uses OpenRouter
        self.api_key  = api_key if api_key is not None else \
                        os.environ.get("OPENROUTER_API_KEY", "")
        self.enabled  = bool(self.api_key)
        if not self.enabled:
            log.info("SemanticRecoveryAgent: OPENROUTER_API_KEY not set — Stage 2.5 disabled")
            log.info("  Get a free key at openrouter.ai → Keys → Create Key")
        else:
            if provider != "openrouter":
                log.info(
                    "SemanticRecoveryAgent: Stage 2.5 uses OpenRouter free models "
                    "(main provider=%s is used for Stage 4 only)", provider
                )
            else:
                log.info("SemanticRecoveryAgent: OpenRouter ready")

    @staticmethod
    def _build_context_block(func: dict, callee_roles: dict,
                             callee_summaries: dict) -> str:
        """
        Build a structured context block prepended to the LLM prompt (Task 3).
        Includes structural flags, top scoring reasons, and callee summaries
        for callees already classified earlier in this binary (topo order).
        """
        parts = []

        # Structural flags — direct semantic signals from Stage 2 scoring
        flags = func.get("flags") or {}
        active = [k for k, v in flags.items() if v]
        if active:
            parts.append("Structural flags: " + ", ".join(active))

        # Top-3 scoring reasons (most informative)
        reasons = func.get("reasons") or []
        top_reasons = []
        for r in reasons:
            if isinstance(r, dict):
                val = r.get("value", [])
            elif isinstance(r, (list, tuple)):
                val = r
            else:
                continue
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                top_reasons.append(str(val[0]))
        seen, unique = set(), []
        for r in top_reasons:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        if unique[:3]:
            parts.append("Top score signals: " + ", ".join(unique[:3]))

        # Callee summaries for already-classified callees (learned this binary)
        call_sites = func.get("call_sites") or []
        callee_info = []
        for callee in call_sites:
            if not isinstance(callee, str):
                continue
            role = callee_roles.get(callee)
            cs   = callee_summaries.get(callee)
            if cs and not cs.get("null_result"):
                desc = f"  {callee}: role={cs.get('likely_role','?')}"
                if cs.get("return_tainted"):
                    desc += ", return_tainted=true"
                if cs.get("writes_memory_at", -1) != -1:
                    desc += f", writes_memory_at={cs['writes_memory_at']}"
                if cs.get("external_input"):
                    desc += f", external_input={cs['external_input']}"
                if cs.get("marks_checked_args"):
                    desc += f", marks_checked_args={cs['marks_checked_args']}"
                callee_info.append(desc)
            elif role and role != "other":
                callee_info.append(f"  {callee}: role={role}")
        if callee_info:
            parts.append("Known callees:\n" + "\n".join(callee_info))

        return "\n".join(parts)

    def summarise(self, fn_name, ops, arg_sizes=None, ret_size=0,
                  callee_roles=None, func_ctx=None, callee_summaries=None):
        if not self.enabled:
            return None

        base_msg = _USER_TEMPLATE.format(
            fn_name   = fn_name,
            arg_count = len(arg_sizes or []),
            arg_sizes = arg_sizes or [],
            op_count  = min(len(ops), self.max_ops),
            ops_text  = _render_ops(ops, self.max_ops, callee_roles=callee_roles),
        )

        # Prepend rich context block if available (Task 3)
        if func_ctx is not None:
            ctx_block = self._build_context_block(
                func_ctx, callee_roles or {}, callee_summaries or {}
            )
            if ctx_block:
                base_msg = ctx_block + "\n\n" + base_msg

        raw = self._call(base_msg, fn_name)
        return self._parse(raw, fn_name) if raw else None

    # ── Task 2: Bottom-up call graph sort ─────────────────────────────────────
    @staticmethod
    def _extract_callee_name(op: dict) -> str:
        """Extract callee symbol name from a CALL op's inputs[0]."""
        inputs = op.get("inputs", [])
        if not inputs or not isinstance(inputs[0], dict):
            return ""
        name = inputs[0].get("name", "")
        if "@" in name:
            name = name.split("@")[0]
        if name.startswith("<") and name.endswith(">"):
            name = name[1:-1]
        return name

    @staticmethod
    def _topo_sort(functions: list[dict]) -> list[dict]:
        """
        Return functions in bottom-up order: callees before callers.
        Callers annotated with callee roles benefit from up-to-date callee_roles
        when their LLM prompt is generated.
        Cycles (mutual recursion) are broken arbitrarily.
        """
        fn_set    = {f["name"] for f in functions if f.get("name")}
        name_to_f = {f["name"]: f for f in functions if f.get("name")}

        # Build call graph restricted to ranked functions (cross-binary knowledge
        # is already in the store and doesn't need reordering).
        callee_of: dict[str, set] = {}
        for f in functions:
            name = f.get("name", "")
            if not name:
                continue
            callee_of[name] = set()
            for op in f.get("ops", []):
                if op.get("op") not in ("CALL", "CALLIND"):
                    continue
                callee = SemanticRecoveryAgent._extract_callee_name(op)
                if callee and callee in fn_set and callee != name:
                    callee_of[name].add(callee)

        # Kahn's algorithm — in_degree = number of callees still unprocessed
        dependents: dict[str, list] = {n: [] for n in fn_set}
        in_degree  = {n: 0 for n in fn_set}
        for caller, callees in callee_of.items():
            for callee in callees:
                dependents[callee].append(caller)
                in_degree[caller] += 1

        queue  = [n for n in fn_set if in_degree[n] == 0]
        order  = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for caller in dependents[n]:
                in_degree[caller] -= 1
                if in_degree[caller] == 0:
                    queue.append(caller)

        # Cycle members still have in_degree > 0 — append in original order
        seen = set(order)
        for f in functions:
            n = f.get("name", "")
            if n and n not in seen:
                order.append(n)

        return [name_to_f[n] for n in order if n in name_to_f]

    def process_binary(self, functions, pattern_store, pattern_matcher,
                       budget=30, count_only=False):
        stats = {"processed":0, "cached":0, "new_summaries":0,
                 "llm_calls":0, "errors":0,
                 "verified":0, "weakly_verified":0, "refuted":0}
        # Build callee role map: hard-coded known + previously learned from DB
        callee_roles = {**_KNOWN_ROLES, **pattern_store.get_all_callee_roles()}

        # Item 10: static verifier prevents LLM hallucinations from entering the store
        from static_verifier import StaticVerifier, VerificationStatus, VerificationStats
        vstats = VerificationStats()

        # Bottom-up sort: process callees before callers so that each caller's
        # LLM prompt already has accurate role annotations for its callees.
        functions = self._topo_sort(functions)

        # callee_summaries: built up as we process each function in topo order.
        # When a caller is processed, its callee summaries are already populated.
        callee_summaries: dict[str, dict] = {}

        for func in functions:
            fn_name = func.get("name","")
            ops     = func.get("ops",[])
            if not ops or not fn_name:
                continue
            if not pattern_matcher.find_unknown_calls(ops):
                continue
            stats["processed"] += 1
            arg_sizes = self._infer_arg_sizes(ops)
            cached = pattern_store.get_learned_summary(fn_name, arg_sizes)
            if cached:
                stats["cached"] += 1
                callee_summaries[fn_name] = cached
                continue
            if count_only or not self.enabled or stats["llm_calls"] >= budget:
                continue
            summary = self.summarise(fn_name, ops, arg_sizes,
                                     self._infer_ret_size(ops),
                                     callee_roles=callee_roles,
                                     func_ctx=func,
                                     callee_summaries=callee_summaries)
            stats["llm_calls"] += 1
            # Always cache the result — even a null/uninteresting response is stored
            # as a sentinel so subsequent binaries skip re-querying the same function.
            stored = summary if summary else {"role": "other", "null_result": True}
            pattern_store.store_learned_summary(fn_name, arg_sizes, stored)
            callee_summaries[fn_name] = stored
            if summary:
                stats["new_summaries"] += 1

                # Phase 3 (Item 10): LLM hypothesis → static verification → store.
                # The verifier checks whether P-code evidence supports the
                # LLM's role claim before committing it to the pattern store.
                # REFUTED hypotheses are not stored as taint rules, preventing
                # LLM hallucinations from poisoning the cross-binary knowledge base.
                role = summary.get("likely_role", "other")
                vresult = StaticVerifier.verify(
                    role      = role,
                    ops       = ops,
                    arg_sizes = arg_sizes,
                    ret_size  = self._infer_ret_size(ops),
                    fn_name   = fn_name,
                )
                vstats.record(vresult)

                if vresult.status == VerificationStatus.REFUTED:
                    stats["refuted"] += 1
                    log.info(
                        "StaticVerifier REFUTED %s hypothesis for %s — "
                        "not storing taint rule. Evidence: %s",
                        role, fn_name, vresult.evidence,
                    )
                    # Store the summary for caching (so we don't re-query),
                    # but downgrade it so pattern_matcher won't use it as a rule.
                    summary["likely_role"]    = "other"
                    summary["null_result"]    = True
                    summary["refuted_role"]   = role   # keep original for debugging
                    summary["refuted_evidence"] = vresult.evidence
                else:
                    # Apply confidence multiplier from verification result
                    orig_conf = summary.get("confidence", 0.6)
                    summary["confidence"] = round(
                        orig_conf * vresult.confidence_multiplier, 3
                    )
                    if vresult.status.value == "confirmed":
                        stats["verified"] += 1
                        log.debug(
                            "StaticVerifier CONFIRMED %s for %s (conf %.0f%%→%.0f%%). %s",
                            role, fn_name, orig_conf*100, summary["confidence"]*100,
                            vresult.evidence,
                        )
                    else:
                        stats["weakly_verified"] += 1
                        log.debug(
                            "StaticVerifier WEAK %s for %s (conf %.0f%%→%.0f%%). %s",
                            role, fn_name, orig_conf*100, summary["confidence"]*100,
                            vresult.evidence,
                        )

                    rule = self.summary_to_taint_rule(summary)
                    if rule:
                        rule["verification_status"] = vresult.status.value
                        rule["verification_evidence"] = vresult.evidence
                        pattern_store.store_learned_rule(
                            fn_name   = fn_name,
                            arg_sizes = arg_sizes,
                            rule      = rule,
                            role      = role,
                        )
                        log.debug("Phase 3: verified taint rule stored for %s (role=%s, status=%s)",
                                  fn_name, role, vresult.status.value)

                        # Item 6: also store structural fingerprint.
                        # Future binaries with a structurally-identical function
                        # (same P-code group histogram) will get STRUCTURAL_MATCH
                        # without needing an LLM call.
                        try:
                            from structural_fingerprint import FingerprintMatcher
                            matcher = FingerprintMatcher(pattern_store)
                            sha = matcher.store_fingerprint(
                                ops       = ops,
                                fn_name   = fn_name,
                                role      = role,
                                taint_rule = rule,
                                arg_sizes = arg_sizes,
                            )
                            log.debug("Fingerprint stored for %s → %s (role=%s)",
                                      fn_name, sha[:8], role)
                        except Exception as _fe:
                            log.debug("Fingerprint store error for %s: %s", fn_name, _fe)

                # Incrementally update callee_roles so subsequent callers in this
                # binary (processed after in topo order) see the new role.
                # Only update if the role was not refuted by static verification.
                if (vresult.status != VerificationStatus.REFUTED
                        and role != "other"):
                    callee_roles[fn_name] = role

                # Task 6: store caller→callee graph edges for this function.
                # For each call in the classified function, record the edge with
                # the callee's resolved role (from callee_roles).
                for op in func.get("ops", []):
                    if op.get("op") not in ("CALL", "CALLIND"):
                        continue
                    inputs = op.get("inputs") or []
                    if not inputs or not isinstance(inputs[0], dict):
                        continue
                    callee = inputs[0].get("name", "")
                    if callee and "@" in callee:
                        callee = callee.split("@")[0]
                    if callee and callee.startswith("<") and callee.endswith(">"):
                        callee = callee[1:-1]
                    if not callee or callee == fn_name:
                        continue
                    callee_role = callee_roles.get(callee, "other")
                    if callee_role != "other":
                        pattern_store.store_graph_edge(fn_name, callee, callee_role,
                                                       confidence=summary.get("confidence", 0.6))

                log.info("SemanticRecovery: %s → role=%s conf=%.0f%% "
                         "(writes=%s alloc=%s external_input=%s)",
                         fn_name, summary.get("likely_role","?"),
                         summary.get("confidence",0)*100,
                         summary.get("writes_memory",False),
                         summary.get("allocation",False),
                         summary.get("external_input",[]))
            else:
                stats["errors"] += 1
            time.sleep(self.delay_s)

        # Log verification summary for this binary (Item 10)
        if stats["llm_calls"] > 0:
            vstats.log_summary(prefix="  ")
            stats.update(vstats.totals())

        return stats

    def _call(self, user_message: str, fn_name: str) -> Optional[str]:
        from llm_cost_tracker import GLOBAL_TRACKER
        models = [self.model] if self.model else _FREE_MODELS
        for model in models:
            payload = json.dumps({
                "model": model, "max_tokens": 512, "temperature": 0.0,
                "messages": [
                    {"role":"system","content":_SYSTEM_PROMPT},
                    {"role":"user",  "content":user_message},
                ],
            }).encode()
            try:
                t_start = time.perf_counter()
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=payload,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type":  "application/json",
                             "HTTP-Referer":  "https://github.com/binary-vuln-pipeline",
                             "X-Title":       "BinaryVulnPipeline"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=20) as resp:
                    body = json.loads(resp.read())
                    if "error" in body:
                        log.warning("OpenRouter %s error: %s", model, body["error"])
                        continue
                    text   = body["choices"][0]["message"]["content"]
                    usage  = body.get("usage", {})
                    GLOBAL_TRACKER.record(
                        stage         = "semantic_recovery",
                        model         = model,
                        input_tokens  = int(usage.get("prompt_tokens", len(user_message)//4)),
                        output_tokens = int(usage.get("completion_tokens", len(text)//4)),
                        latency_s     = time.perf_counter() - t_start,
                        fn_name       = fn_name,
                    )
                    log.debug("OpenRouter OK (%s)", model.split("/")[-1])
                    return text
            except urllib.error.HTTPError as e:
                e.read()
                if e.code in (429, 402):
                    log.warning("OpenRouter %d on %s — trying next model", e.code, model)
                    time.sleep(2); continue
                elif e.code == 401:
                    log.error("OpenRouter 401 — invalid key. Get key at openrouter.ai")
                    return None
                else:
                    log.warning("OpenRouter HTTP %d on %s", e.code, model); continue
            except Exception as exc:
                log.warning("OpenRouter error for %s: %s", fn_name, exc); continue
        return None

    def _parse(self, raw: str, fn_name: str) -> Optional[dict]:
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.split("\n")
                             if not l.strip().startswith("```"))
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if not m: return None
            try: parsed = json.loads(m.group())
            except: return None
        valid = {"allocator","copy","read_input","format_string",
                 "exec","free","logger","validator","other"}
        def _safe_int_list(val) -> list[int]:
            if not isinstance(val, list):
                return []
            return [int(x) for x in val if isinstance(x, (int, float)) and 0 <= int(x) <= 15]

        s = {
            "likely_role":          parsed.get("likely_role", "other"),
            "confidence":           float(parsed.get("confidence", 0.5)),
            "writes_memory":        bool(parsed.get("writes_memory", False)),
            "reads_memory":         bool(parsed.get("reads_memory", False)),
            "allocation":           bool(parsed.get("allocation", False)),
            "network_input":        bool(parsed.get("network_input", False)),
            "file_input":           bool(parsed.get("file_input", False)),
            "return_tainted":       bool(parsed.get("return_tainted", False)),
            "size_arg":             int(parsed.get("size_arg", -1)),
            "writes_memory_at":     int(parsed.get("writes_memory_at", -1)),
            "external_input":       list(parsed.get("external_input", [])),
            # LLM-driven taint propagation (Task 1)
            "marks_checked_args":   _safe_int_list(parsed.get("marks_checked_args", [])),
            "external_source_args": _safe_int_list(parsed.get("external_source_args", [])),
            "reasoning":            str(parsed.get("reasoning", ""))[:200],
        }
        if s["likely_role"] not in valid:
            s["likely_role"] = "other"
        return s

    def _infer_arg_sizes(self, ops):
        sizes = []
        for op in ops[:10]:
            out = op.get("output")
            if isinstance(out,dict) and out.get("space")=="unique":
                s = out.get("size",0)
                if s in (4,8): sizes.append(s)
        return sizes[:6] if sizes else [8]

    def _infer_ret_size(self, ops):
        for op in reversed(ops):
            if op.get("op")=="RETURN":
                ins = op.get("inputs") or []
                if len(ins)>=2 and isinstance(ins[1],dict):
                    return ins[1].get("size",0)
        return 0

    def summary_to_taint_rule(self, summary: dict) -> Optional[dict]:
        role = summary.get("likely_role", "other")
        rule = {
            "external_input":       summary.get("external_input", []),
            "writes_memory_at":     summary.get("writes_memory_at", -1),
            "return_tainted":       summary.get("return_tainted", False),
            "bounded":              False,
            "size_arg":             summary.get("size_arg", -1),
            "frees_memory_at":      -1,
            "sink":                 False,
            "sink_type":            "",
            "confidence":           summary.get("confidence", 0.6),
            "vuln_score":           0,
            "source":               "semantic_recovery",
            "notes":                f"Role:{role}. {summary.get('reasoning', '')}",
            # LLM-driven taint propagation fields (Task 1)
            "marks_checked_args":   summary.get("marks_checked_args", []),
            "external_source_args": summary.get("external_source_args", []),
        }
        if role == "allocator":
            rule.update({"sink": True, "sink_type": "integer_overflow",
                         "return_tainted": True, "vuln_score": 2})
        elif role == "copy":
            rule.update({"sink": True, "sink_type": "buffer_overflow",
                         "writes_memory_at": summary.get("writes_memory_at", 0),
                         "vuln_score": 2})
        elif role == "read_input":
            rule.update({"external_input": summary.get("external_input") or [1],
                         "return_tainted": True, "vuln_score": 3})
        elif role == "exec":
            rule.update({"sink": True, "sink_type": "command_injection", "vuln_score": 8})
        elif role == "validator":
            # Validator: no sink, but marks_checked_args tells the taint engine
            # which args are safe after this call. vuln_score=0 — this is a
            # precision improvement, not a vulnerability signal.
            rule.update({"vuln_score": 0})
        # Return rule for all roles that actively affect taint state.
        # "other" and "logger" have no taint effect — skip storing.
        if role in ("other", "logger", "free"):
            return None
        return rule