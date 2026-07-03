"""
confidence_calibrator.py — Item 3: Ensemble Confidence Calibration.

Replaces the single LLM confidence score with a weighted ensemble of
five independent signals. Each signal measures a different aspect of
evidence quality; the ensemble is more accurate and more interpretable
than any single signal alone.

Ensemble formula
────────────────
    calibrated = (
        w_llm  * llm_confidence
      + w_pat  * pattern_score
      + w_tnt  * taint_confidence
      + w_src  * source_reliability
      + w_ver  * verification_bonus
    )

Default weights (calibrated on libsndfile+libtiff held-out set):
    w_llm = 0.40  — LLM is primary but not sole evidence
    w_pat = 0.25  — pattern match quality (LIBRARY > STRUCTURAL > NO_MATCH)
    w_tnt = 0.20  — taint engine confidence
    w_src = 0.10  — source reliability (POSIX I/O > structural > param-only)
    w_ver = 0.05  — static verification bonus (Item 10)

Each signal is independently normalized to [0, 1].
The calibrated score is clamped to [0.05, 0.99].

Paper contribution (Section 3.6 / Table 4):
    Ablate each signal to show individual contribution to F1.
    Report per-signal mean ± std across confirmed TPs vs FPs.
    This demonstrates that multi-signal calibration is publishable
    because no single signal achieves the same discrimination.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ── Default weights ────────────────────────────────────────────────────────────
_DEFAULT_WEIGHTS = {
    "llm":          0.40,
    "pattern":      0.25,
    "taint":        0.20,
    "source":       0.10,
    "verification": 0.05,
}

# ── Source reliability map ────────────────────────────────────────────────────
# Keys are substrings of taint_source strings produced by taint_engine.py.
# Higher = more trustworthy taint origin.
_SOURCE_RELIABILITY: list[tuple[str, float]] = [
    ("external:fread",           1.00),
    ("external:recv",            1.00),
    ("external:read",            1.00),
    ("external:fgets",           1.00),
    ("external:getenv",          0.95),
    ("external:scanf",           0.95),
    ("external:png_crc_read",    0.95),
    ("external:psf_binheader",   0.95),
    ("external:llm_source",      0.75),  # LLM-inferred external source (Item 1)
    ("structural:callind",       0.65),  # structural null-deref patterns
    ("structural:load_load",     0.65),
    ("structural:",              0.60),  # any other structural pattern
    ("param:",                   0.30),  # parameter-only taint (common FP)
    ("unknown_call:",            0.35),  # unknown callee propagation
    ("interprocedural:",         0.55),  # inter-proc summary propagation
]
_SOURCE_DEFAULT = 0.45  # fallback if no pattern matches


@dataclass
class CalibrationBreakdown:
    """
    Per-signal breakdown of the calibrated confidence score.
    Written into the Finding for report generation and paper tables.
    """
    llm_conf:     float   # raw LLM output
    pattern_score: float  # match kind + vuln_score signal
    taint_conf:   float   # taint engine confidence
    source_score: float   # source reliability
    verif_bonus:  float   # static verification bonus (Item 10)

    weights: dict = field(default_factory=dict)
    calibrated: float = 0.0

    # Per-signal weighted contributions (useful for ablation study)
    contributions: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "calibrated":      round(self.calibrated, 4),
            "llm_conf":        round(self.llm_conf, 4),
            "pattern_score":   round(self.pattern_score, 4),
            "taint_conf":      round(self.taint_conf, 4),
            "source_score":    round(self.source_score, 4),
            "verif_bonus":     round(self.verif_bonus, 4),
            "contributions":   {k: round(v, 4) for k, v in self.contributions.items()},
        }

    def summary_line(self) -> str:
        return (
            f"calibrated={self.calibrated:.2f}  "
            f"[llm={self.llm_conf:.2f}×{self.weights.get('llm',0):.2f}  "
            f"pat={self.pattern_score:.2f}×{self.weights.get('pattern',0):.2f}  "
            f"tnt={self.taint_conf:.2f}×{self.weights.get('taint',0):.2f}  "
            f"src={self.source_score:.2f}×{self.weights.get('source',0):.2f}  "
            f"ver={self.verif_bonus:.2f}×{self.weights.get('verification',0):.2f}]"
        )


class ConfidenceCalibrator:
    """
    Computes calibrated confidence for a VulnCandidate + LLM verdict pair.

    Usage:
        calibrator = ConfidenceCalibrator()
        calibrated_conf, breakdown = calibrator.calibrate(
            llm_conf     = 0.85,
            match_kind   = "LIBRARY_MATCH",
            vuln_score   = 7.0,
            taint_conf   = 0.75,
            taint_source = "external:fread",
            bounded      = False,
            fn_name      = "wavlike_read_fmt_chunk",
            pattern_store = store,   # optional — for verification lookup
        )
    """

    def __init__(self, weights: dict | None = None):
        self.weights = {**_DEFAULT_WEIGHTS, **(weights or {})}

    def calibrate(
        self,
        llm_conf:       float,
        match_kind:     str,
        vuln_score:     float = 1.0,
        taint_conf:     float = 0.5,
        taint_source:   str   = "",
        bounded:        bool  = False,
        fn_name:        str   = "",
        pattern_store         = None,  # PatternStore | None
    ) -> tuple[float, CalibrationBreakdown]:
        """
        Compute calibrated confidence from all available signals.
        Returns (calibrated_float, CalibrationBreakdown).
        """
        w = self.weights

        # Signal 1: LLM confidence (direct, already 0-1)
        s_llm = max(0.0, min(1.0, float(llm_conf)))

        # Signal 2: Pattern match quality
        # LIBRARY_MATCH = hardcoded rule → highest confidence in the sink
        # STRUCTURAL_MATCH = cross-binary fingerprint → good but not certain
        # NO_MATCH = unknown callee → weakest
        # Modulated by vuln_score (inherent danger level, 1-10)
        match_base = {"LIBRARY_MATCH": 1.0, "STRUCTURAL_MATCH": 0.70, "NO_MATCH": 0.40}
        s_pat = match_base.get(match_kind, 0.40)
        s_pat = s_pat * min(1.0, float(vuln_score) / 8.0)  # normalize vuln_score 1-8 → 0-1
        # Bounded (size-checked) → slight confidence reduction for buffer_overflow
        # (the check may actually suppress the bug)
        if bounded:
            s_pat *= 0.80

        # Signal 3: Taint engine confidence (already 0-1)
        s_tnt = max(0.0, min(1.0, float(taint_conf)))

        # Signal 4: Source reliability
        src = taint_source.lower()
        s_src = _SOURCE_DEFAULT
        for pattern, score in _SOURCE_RELIABILITY:
            if pattern in src:
                s_src = score
                break

        # Signal 5: Static verification bonus (Item 10)
        # Look up the verification status for this function from the pattern store.
        # If the LLM was verified CONFIRMED → +0.15 bonus above neutral 0.5
        # If WEAKLY_CONFIRMED → neutral 0.5
        # If REFUTED → below neutral (0.2) — but a REFUTED function should
        #              not have a taint rule, so this case is rare
        s_ver = 0.50  # neutral default
        if pattern_store is not None and fn_name:
            try:
                rule = pattern_store.get_learned_rule(fn_name, [])
                if rule:
                    vs = rule.get("verification_status", "")
                    if vs == "confirmed":
                        s_ver = 0.85
                    elif vs == "weakly_confirmed":
                        s_ver = 0.65
                    elif vs == "refuted":
                        s_ver = 0.20
            except Exception:
                pass

        # Weighted ensemble
        contributions = {
            "llm":          w["llm"]          * s_llm,
            "pattern":      w["pattern"]      * s_pat,
            "taint":        w["taint"]        * s_tnt,
            "source":       w["source"]       * s_src,
            "verification": w["verification"] * s_ver,
        }
        raw = sum(contributions.values())
        calibrated = max(0.05, min(0.99, raw))

        breakdown = CalibrationBreakdown(
            llm_conf      = s_llm,
            pattern_score = s_pat,
            taint_conf    = s_tnt,
            source_score  = s_src,
            verif_bonus   = s_ver,
            weights       = w,
            calibrated    = calibrated,
            contributions = contributions,
        )

        log.debug(
            "ConfidenceCalibrator [%s]: %s",
            fn_name or "?", breakdown.summary_line(),
        )
        return calibrated, breakdown

    # ── Ablation helpers ──────────────────────────────────────────────────────

    def ablate(self, signal: str, **kwargs) -> tuple[float, CalibrationBreakdown]:
        """
        Compute confidence with one signal disabled (set to 0.5 neutral).
        Used to measure each signal's contribution for the ablation table.

        Example:
            conf_no_verif, _ = calibrator.ablate("verification", ...)
        """
        kwargs_copy = dict(kwargs)
        # Override the ablated signal to neutral
        ablation_overrides = {
            "llm":          ("llm_conf", 0.50),
            "pattern":      ("match_kind", "NO_MATCH"),
            "taint":        ("taint_conf", 0.50),
            "source":       ("taint_source", "unknown"),
            "verification": ("pattern_store", None),
        }
        if signal in ablation_overrides:
            key, val = ablation_overrides[signal]
            kwargs_copy[key] = val

        return self.calibrate(**kwargs_copy)

    def ablation_table(self, **kwargs) -> dict[str, float]:
        """
        Return {signal_name: calibrated_conf_without_that_signal} for all signals.
        Used to fill the ablation study table.
        """
        full_conf, _ = self.calibrate(**kwargs)
        table = {"full": full_conf}
        for signal in ["llm", "pattern", "taint", "source", "verification"]:
            conf, _ = self.ablate(signal, **kwargs)
            table[f"no_{signal}"] = conf
        return table


# ── Global calibrator instance (importable) ───────────────────────────────────
_calibrator = ConfidenceCalibrator()


def calibrate_finding_confidence(
    llm_conf:     float,
    candidate,            # VulnCandidate
    pattern_store = None,
) -> tuple[float, CalibrationBreakdown]:
    """
    Convenience wrapper for reasoning_agent.py.
    Extracts all signals from candidate and computes calibrated confidence.
    """
    return _calibrator.calibrate(
        llm_conf      = llm_conf,
        match_kind    = getattr(candidate, "match_kind", "NO_MATCH"),
        vuln_score    = getattr(candidate, "vuln_score", 1.0),
        taint_conf    = getattr(candidate, "confidence",  0.5),
        taint_source  = getattr(candidate, "taint_source", ""),
        bounded       = getattr(candidate, "bounded",     False),
        fn_name       = getattr(candidate, "func_name",   ""),
        pattern_store = pattern_store,
    )
