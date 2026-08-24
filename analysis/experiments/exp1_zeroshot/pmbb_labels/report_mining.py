#!/usr/bin/env python3
"""
CLEAR_CT_EXPS / exp1_zeroshot / pmbb_labels — clinical phrase-mining engine.

Generates per-scan binary finding labels from free-text CT reports, replicating
Merlin's report-mining label-creation step (arXiv 2406.06512: a radiologist-
curated finding list whose presence/absence is recovered from the report by
regex matching). No LLM is used.

Design (clinically motivated):
  * We mine ONLY the FINDINGS + IMPRESSION sections. History / Indication /
    Clinical / Technique / Comparison sections describe PRIOR conditions or the
    acquisition, not what is seen on THIS scan — mining them leaks false
    positives (e.g. "history of metastatic disease"). This also matches Merlin,
    which mines the report *findings*.
  * Negation and uncertainty are SENTENCE-SCOPED (a NegEx/ConText-lite): a
    finding asserted in one sentence and denied in another is still PRESENT
    (e.g. "minimal effusion on the right. No effusion on the left." -> present).
  * Per finding we return one of:
        1  PRESENT     - matched, not negated, not hedged
        0  ABSENT      - explicitly negated, OR never mentioned (absent default)
       -1  UNCERTAIN   - matched but hedged ("possible", "cannot exclude", ...)
    Aggregation priority across the report: PRESENT > UNCERTAIN > ABSENT.
    (-1 is kept distinct so the eval layer can map it; see mine.py.)

The finding rule dictionaries live in finding_rules.py.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------
PRESENT, ABSENT, UNCERTAIN = 1, 0, -1

# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------
# Section handling is DROP-based, not KEEP-based: PMBB (and most modern) reports
# put the findings in ANATOMIC sections ("LUNGS:", "LIVER:", "PERITONEUM:") with
# no single "FINDINGS:" header, so we keep everything EXCEPT the known non-finding
# sections (priors / acquisition / admin). Headers are detected anywhere (reports
# may be single-line, so line anchoring fails) as Capitalized/UPPER word(s) + ":".
# A header = 1-5 Capitalized/UPPER words then a colon, detected ANYWHERE (PMBB
# packs sections inline & space-separated, so period/newline anchoring misses
# "... None Comparison: ..."). Over-detecting a header inside a KEPT section is
# harmless (text still kept); it only needs to correctly bound DROP sections.
_HEADER_WORD = r"[A-Z][A-Za-z0-9/&'\-]*(?:[ ][A-Z0-9][A-Za-z0-9/&'\-]*){0,4}"
_SECTION_RE = re.compile(rf"(?:^|(?<=[^A-Za-z]))({_HEADER_WORD})\s*:")
# A section is DROPPED if its header label CONTAINS any of these tokens — catches
# multi-word variants like "Female History:", "Clinical Indication:", "Reason for
# exam:" that an exact-match would miss.
_DROP_SECTION = re.compile(
    r"\b(history|indication|indications|technique|comparison|comparisons|"
    r"protocol|procedure|dictated|attestation|addendum|disclaimer|limitations?|"
    r"reason for|recommendations?|notification|electronically signed|signed by)\b",
    re.I)


def extract_findings_impression(report: str) -> str:
    """Drop non-finding sections (history/indication/technique/comparison/admin);
    keep everything else (anatomic finding sections + impression/conclusion)."""
    if not isinstance(report, str) or not report.strip():
        return ""
    heads = list(_SECTION_RE.finditer(report))
    if not heads:
        return report                                   # unstructured: mine all
    kept = []
    pre = report[:heads[0].start()].strip()             # exam-title preamble
    if pre:
        kept.append(pre)
    for i, m in enumerate(heads):
        label = m.group(1).strip().lower()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(report)
        if _DROP_SECTION.search(label):
            continue
        kept.append(report[m.start():end])
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
# Split on sentence punctuation and newlines/semicolons. Protect common decimal
# numbers and abbreviations minimally (radiology is terse; this is adequate and
# is validated downstream against RadBERT).
_SENT_SPLIT = re.compile(r"(?:(?<=[.;!?])\s+)|[\r\n]+")


def split_sentences(text: str) -> list[str]:
    text = text.replace("–", "-").replace("—", "-")
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# NegEx/ConText-lite cue lists
# ---------------------------------------------------------------------------
# Pre-trigger negations (scope forward to the finding).
_PRE_NEG = [
    r"no", r"not", r"never", r"without", r"absence of", r"absent",
    r"no evidence of", r"no evidence for", r"no sign of", r"no signs of",
    r"no findings? of", r"no definite", r"no significant", r"no convincing",
    r"negative for", r"free of", r"clear of", r"resolution of", r"resolved",
    r"rule out", r"r/o", r"to exclude", r"denies", r"unremarkable for",
    r"no ct (?:evidence|findings?) of",
]
# Post-trigger negations (the finding precedes the cue).
_POST_NEG = [
    r"is absent", r"are absent", r"was absent", r"were absent",
    r"not seen", r"not present", r"not identified", r"not demonstrated",
    r"not visualized", r"not appreciated", r"not detected", r"is not seen",
    r"not observed", r"could not be observed",      # CT-RATE idiom "was not observed"
    r"ruled out", r"excluded", r"has resolved", r"have resolved", r"resolved",
    r"is unremarkable", r"are unremarkable", r"is normal", r"are normal",
]
# Hedging / uncertainty (either side).
_PRE_UNC = [
    r"possible", r"possibly", r"probable", r"probably", r"likely", r"may",
    r"may be", r"might", r"could (?:be|represent)", r"suspicious for",
    r"suspected", r"concerning for", r"questionable", r"question of",
    r"equivocal", r"indeterminate", r"cannot exclude", r"cannot be excluded",
    r"cannot rule out", r"can not exclude", r"differential", r"presumed",
    r"compatible with", r"consistent with", r"worrisome for", r"favor(?:ed|s)?",
]
_POST_UNC = [
    r"is possible", r"cannot be excluded", r"can not be excluded",
    r"cannot be ruled out", r"is suspected", r"is questioned", r"versus",
    r"vs\.?", r"or (?:other|less likely)", r"\?",
]
# Pseudo-negations: phrases that contain a negation word but do NOT negate the
# finding (block negation when present in the window).
_PSEUDO_NEG = [
    r"no significant change", r"no significant interval change",
    r"no interval change", r"no change", r"not significantly changed",
    r"no new", r"not significantly", r"no longer", r"not only", r"not just",
    r"no definite interval", r"without significant change",
]
# Scope terminators: a clause boundary stops a negation/uncertainty scope.
_TERMINATORS = [
    r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b", r"\bexcept\b",
    r"\byet\b", r"\bwhich (?:demonstrates|shows|reveals|is)\b", r"\bwith\b",
    r"\bas well as\b", r"\baside from\b", r"\bother than\b",
]

# NegEx scope runs from a trigger to the next clause terminator (handles list
# negations like "no A, B, or C" that span many tokens), bounded by the sentence.

# Cues match ANYWHERE inside the (terminator-trimmed, WINDOW-token) neighborhood
# — NegEx scopes a trigger across its whole forward/backward window, not just the
# token adjacent to the finding (so "no LARGE pleural effusion" still negates).
# NOTE: case-insensitive — reports capitalize sentence-initial cues ("No ...").
_PRE_NEG_RE = [re.compile(rf"\b{c}\b", re.I) for c in _PRE_NEG]
_POST_NEG_RE = [re.compile(rf"\b{c}\b", re.I) for c in _POST_NEG]
_PRE_UNC_RE = [re.compile(rf"\b{c}\b", re.I) for c in _PRE_UNC]
_POST_UNC_RE = [re.compile(rf"(?:\b{c}\b|{c})", re.I) for c in _POST_UNC]
_PSEUDO_RE = [re.compile(rf"{c}", re.I) for c in _PSEUDO_NEG]
_TERM_RE = re.compile("|".join(_TERMINATORS), re.I)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _scope_before(sent: str, idx: int) -> str:
    """Clause before the match: from the last terminator before `idx` (or the
    sentence start) up to `idx`. A pre-trigger negation anywhere in this clause
    scopes the finding (covers 'no A, B, or C')."""
    pre = sent[:idx]
    term = None
    for mt in _TERM_RE.finditer(pre):
        term = mt.end()
    return pre[term:] if term is not None else pre


def _scope_after(sent: str, idx: int) -> str:
    """Clause after the match: from `idx` to the next terminator (or sentence end)."""
    post = sent[idx:]
    mt = _TERM_RE.search(post)
    return post[:mt.start()] if mt else post


def _status_for_match(sent: str, start: int, end: int) -> int:
    pre = _scope_before(sent, start)
    post = _scope_after(sent, end)
    blocked = any(r.search(pre) or r.search(post) for r in _PSEUDO_RE)
    if not blocked:
        if any(r.search(pre) for r in _PRE_NEG_RE) or \
           any(r.search(post) for r in _POST_NEG_RE):
            return ABSENT
    if any(r.search(pre) for r in _PRE_UNC_RE) or \
       any(r.search(post) for r in _POST_UNC_RE):
        return UNCERTAIN
    return PRESENT


_HEADER_LABEL_TAIL = re.compile(r"s?\s*:")   # match is a section header ("Nodules:")


def classify(text: str, patterns: list[re.Pattern],
             excludes: tuple = ()) -> int:
    """Status of one finding (given its compiled presence patterns) in `text`.

    Scans every sentence; aggregates with priority PRESENT > UNCERTAIN > ABSENT.
    A sentence matching any `excludes` pattern is skipped for this finding (wrong
    organ / boilerplate, e.g. a THYROID nodule or a Lung-RADS template line). A
    match that is itself a section-header label ("Pulmonary nodules:") is skipped
    so the finding's status comes from the section CONTENT, not its title.
    Never-mentioned -> ABSENT (0)."""
    best = ABSENT
    saw_match = False
    for sent in split_sentences(text):
        if excludes and any(e.search(sent) for e in excludes):
            continue
        for pat in patterns:
            for m in pat.finditer(sent):
                if _HEADER_LABEL_TAIL.match(sent[m.end():]):
                    continue                            # finding used as a header
                saw_match = True
                st = _status_for_match(sent, m.start(), m.end())
                if st == PRESENT:
                    return PRESENT                      # short-circuit: best possible
                if st == UNCERTAIN:
                    best = UNCERTAIN
    # matched only under negation -> ABSENT (0); never matched -> ABSENT default.
    return best if saw_match else ABSENT


def compile_rules(rules: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    """finding -> compiled case-insensitive presence patterns."""
    return {name: [re.compile(p, re.IGNORECASE) for p in pats]
            for name, pats in rules.items()}


def label_report(text: str, compiled: dict[str, list[re.Pattern]],
                 mine_sections: bool = True,
                 excludes: dict | None = None) -> dict[str, int]:
    """Full report text -> {finding: status} dict. If mine_sections, restrict to
    FINDINGS+IMPRESSION first (PMBB full reports); set False when `text` is
    already a clean findings field (CT-RATE calibration). `excludes` maps a
    finding -> compiled veto patterns (see classify)."""
    body = extract_findings_impression(text) if mine_sections else text
    excludes = excludes or {}
    return {name: classify(body, pats, tuple(excludes.get(name, ())))
            for name, pats in compiled.items()}
