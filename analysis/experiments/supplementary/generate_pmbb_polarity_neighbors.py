#!/usr/bin/env python3
"""Generate the PMBB-to-reference polarity-neighbour supplementary table.

For each locked affirmative PMBB observation, this script retrieves the three
nearest same-finding affirmative and three nearest same-finding, explicitly
negated observation strings independently from CT-RATE and Merlin.  Neighbours
are ranked by exact Euclidean distance between the stored, L2-normalized
5,120-dimensional F2LLM vectors; PCA and UMAP coordinates are never used.

The large embedding arrays are uncompressed ``.npy`` members inside ``.npz``
archives.  ``mmap_stored_npy_member`` maps those members in place, avoiding a
second multi-gigabyte cache file and avoiding materialization of either bank.

Outputs
-------
``pmbb_polarity_neighbors.csv``
    One row per query and rank, with full-precision distances.
``pmbb_polarity_neighbors_longtable.tex``
    Portrait, full-width, 12-pt-compatible LaTeX fragment.
``pmbb_polarity_neighbors_status.json``
    Inputs, definitions, pool counts, selected rows and validation checks.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import struct
import unicodedata
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_BANK = ROOT / "concept_bank.f2llm_emb.npz"
DEFAULT_PMBB_BANK = (
    ROOT
    / "pmbb_concepts"
    / "banks"
    / "pmbb_qwen36_extracted16896_f2llm_20260715"
    / "concept_bank.f2llm_emb.npz"
)
DEFAULT_CTRATE = ROOT / "ctrate_concepts.jsonl"
DEFAULT_MERLIN = ROOT / "merlin_concepts.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

AFFIRMATIVE = np.uint8(1)
NEGATED = np.uint8(2)
EXCLUDED = np.uint8(3)


@dataclass(frozen=True)
class QuerySpec:
    domain: str
    target: str
    query: str
    eligibility_pattern: str


# Locked after a distance-free coverage audit established at least three
# same-finding affirmative and explicitly negated candidates in each source.
# Each query is an exact PMBB string, absent from the combined CT-RATE/Merlin
# bank, and contains no number, date, measurement, identifier, uncertainty cue
# or negation cue.
LOCKED_QUERIES = (
    QuerySpec(
        "Chest",
        "Pulmonary nodules",
        "pulmonary nodules present",
        r"(?=.*\b(?:pulmonary|lung)\b)(?=.*\bnodules?\b)",
    ),
    QuerySpec(
        "Chest",
        "Pleural effusion",
        "pleural effusion",
        r"\bpleural\s+effusions?\b",
    ),
    QuerySpec(
        "Chest",
        "Coronary calcification",
        "coronary calcification is seen",
        r"(?=.*\b(?:coronary|lad|circumflex)\b)"
        r"(?=.*\b(?:calcif\w*|atheroma|plaque\w*)\b)",
    ),
    QuerySpec(
        "Chest",
        "Lymphadenopathy",
        "hilar lymphadenopathy",
        r"\b(?:lymphadenopathy|adenopathy)\b",
    ),
    QuerySpec(
        "Abdomen",
        "Hepatic mass or lesion",
        "hepatic mass",
        r"(?=.*\b(?:hepatic\b(?!\s+flexure)|liver\b))"
        r"(?=.*\b(?:lesion\w*|mass(?:es)?)\b)",
    ),
    QuerySpec(
        "Abdomen",
        "Pancreatic mass or lesion",
        "cystic pancreatic lesions",
        r"(?=.*\bpancrea\w*\b)(?=.*\b(?:mass(?:es)?|lesion\w*)\b)",
    ),
    QuerySpec(
        "Abdomen",
        "Adrenal mass or nodule",
        "adrenal nodules unchanged",
        r"(?=.*\badrenal\w*\b)"
        r"(?=.*\b(?:mass(?:es)?|lesion\w*|nodules?|adenoma\w*)\b)",
    ),
    QuerySpec(
        "Abdomen",
        "Renal calculus",
        "renal calculi",
        r"(?=.*\b(?:renal|kidney|ureter\w*)\b)"
        r"(?=.*\b(?:stone\w*|calcul\w*)\b)",
    ),
)


UNCERTAINTY_RE = re.compile(
    r"(?:\bcannot\s+(?:be\s+)?(?:exclude|excluded|rule(?:d)?\s+out)\b|"
    r"\bcan\s+not\s+(?:be\s+)?(?:exclude|excluded|rule(?:d)?\s+out)\b|"
    r"\bnot\s+excluded\b|\brule\s+out\b|\br/o\b|"
    r"\bpossible\b|\bpossibly\b|\bprobable\b|\bprobably\b|"
    r"\blikely\b|\bmay\b|\bmight\b|\bcould\b|\bequivocal\b|"
    r"\bindeterminate\b|\bsuspected\b|\bsuspicious\b|"
    r"\bconcerning\b|\bquestion(?:able)?\b|\bversus\b|\bvs\.?\b|\?)",
    re.IGNORECASE,
)

PARTIAL_NEGATION_RE = re.compile(
    r"^no\s+fat\s+plane\b|"
    r"^no\s+(?:new(?:ly)?|other|additional|associated|further|remaining|interval)\b|"
    r"^no\s+(?:increase|decrease|progression|worsening|improvement)\s+(?:in|of)\b|"
    r"^no\s+(?:significant|definite|definitive|discrete|convincing|obvious|gross|apparent|"
    r"evident|clear|discernible|visualized|appreciable|substantial|"
    r"sizable|large|major|measurable|acute|bilateral|unilateral|"
    r"worrisome|dominant|obstructing)\b|"
    r"\bno\s+(?:significant\s+)?(?:interval\s+)?change\b|"
    r"\bwithout\s+(?:significant\s+)?(?:interval\s+)?change\b|"
    r"\bnot\s+significantly\s+changed\b|\bno\s+longer\b",
    re.IGNORECASE,
)

TECHNICAL_OR_TEMPORAL_RE = re.compile(
    r"(?:\bnot\s+(?:well|clearly|definitely|adequately|optimally)\s+"
    r"(?:visualized|seen|identified|evaluated)\b|"
    r"\blimited\s+(?:exam|examination|evaluation|assessment)\b|"
    r"\bmotion\s+artifact\b|\black\s+of\s+(?:iv\s+)?contrast\b|"
    r"\bstatus\s+post\b|\bsurgically\s+absent\b|\bpost[- ]?(?:operative|surgical)\b|"
    r"\bresolved\b|\bresolution\s+of\b|\bcleared\b)",
    re.IGNORECASE,
)

MIXED_CLAUSE_RE = re.compile(
    r"(?:\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|\bapart\s+from\b|"
    r"\bother\s+than\b|\bas\s+well\s+as\b|[;:])",
    re.IGNORECASE,
)

NORMALITY_RE = re.compile(
    r"(?:\bunremarkable\b|\bwithin\s+normal\s+limits\b|\bnormal\b|\bpatent\b)",
    re.IGNORECASE,
)

NEG_PREFIX_RE = re.compile(
    r"^(?:there\s+(?:is|are|was|were)\s+)?"
    r"(?:no\b|never\b|without\b|(?:an?\s+)?absence\s+of\b|"
    r"negative\s+for\b|free\s+of\b|clear\s+of\b|lack\s+of\b)",
    re.IGNORECASE,
)

NEG_POST_RE = re.compile(
    r"^[^,;:]+?\s+"
    r"(?:(?:is|are|was|were|has\s+been|have\s+been)\s+)?"
    r"(?:absent|not\s+(?:present|detected|demonstrated|evident|visible|"
    r"enlarged|dilated|distended|aneurysmal|obstructed|occluded|thickened))"
    r"(?:\s+(?:on|at|in|within|by|from)\b[^,;:]*)?$",
    re.IGNORECASE,
)

ANY_NEGATION_RE = re.compile(
    r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\babsence\b|\babsent\b|"
    r"\bnegative\s+for\b|\bfree\s+of\b|\bclear\s+of\b|\black\s+of\b|"
    r"\bexcluded\b|\bruled\s+out\b)",
    re.IGNORECASE,
)

QUERY_UNSAFE_RE = re.compile(
    r"(?:\d|@|https?://|\b(?:mrn|accession|patient|name|dob|date|id)\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b)",
    re.IGNORECASE,
)


def normalize_phrase(value: str) -> str:
    """Apply only Unicode/whitespace normalization; preserve lexical content."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip().lower())


def classify_polarity(value: str) -> tuple[np.uint8, str]:
    """Conservatively classify an atomic observation at whole-phrase level."""
    phrase = normalize_phrase(value)
    if not phrase:
        return EXCLUDED, "empty"
    if UNCERTAINTY_RE.search(phrase):
        return EXCLUDED, "uncertain_or_hedged"
    if PARTIAL_NEGATION_RE.search(phrase):
        return EXCLUDED, "partial_or_temporal_negation"
    if TECHNICAL_OR_TEMPORAL_RE.search(phrase):
        return EXCLUDED, "technical_temporal_or_surgical"
    if MIXED_CLAUSE_RE.search(phrase):
        return EXCLUDED, "mixed_clause"
    if NORMALITY_RE.search(phrase):
        return EXCLUDED, "normality_boilerplate"
    if NEG_PREFIX_RE.search(phrase):
        return NEGATED, "strong_prefix_negation"
    if NEG_POST_RE.fullmatch(phrase):
        return NEGATED, "strong_post_negation"
    if ANY_NEGATION_RE.search(phrase):
        return EXCLUDED, "unresolved_negation"
    return AFFIRMATIVE, "affirmative"


def observations(path: Path) -> set[str]:
    """Reconstruct the exact phrase normalization used to build the bank."""
    phrases: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            raw = record.get("model_output", "")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                try:
                    payload = ast.literal_eval(raw)
                except Exception:
                    continue
            values = payload.get("observations") or []
            if not isinstance(values, list):
                raise ValueError(f"{path}:{line_number}: observations is not a list")
            for value in values:
                if isinstance(value, str):
                    phrase = value.lower().strip()
                    if phrase:
                        phrases.add(phrase)
    return phrases


def load_npz_concepts(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        with archive.open("concepts.npy") as handle:
            values = np.load(handle, allow_pickle=True)
    return np.asarray(values, dtype=str)


def mmap_stored_npy_member(path: Path, member: str = "emb.npy") -> np.memmap:
    """Memory-map an uncompressed NPY member directly inside an NPZ archive."""
    local_header = struct.Struct("<IHHHHHIIIHH")
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{member} is compressed and cannot be mapped in place")
        with path.open("rb") as handle:
            handle.seek(info.header_offset)
            raw = handle.read(local_header.size)
            fields = local_header.unpack(raw)
            signature = fields[0]
            filename_length, extra_length = fields[-2:]
            if signature != 0x04034B50:
                raise ValueError(f"invalid ZIP local header for {path}:{member}")
            npy_offset = info.header_offset + local_header.size + filename_length + extra_length
            handle.seek(npy_offset)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported NPY version {version} in {path}:{member}")
            data_offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def merge_topk(
    current_indices: np.ndarray,
    current_distances: np.ndarray,
    new_indices: np.ndarray,
    new_distances: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.concatenate((current_indices, new_indices.astype(np.int64, copy=False)))
    distances = np.concatenate((current_distances, new_distances.astype(np.float64, copy=False)))
    # Bank rows are lexicographically sorted, so stable sorting breaks exact
    # distance ties by normalized text and then bank row.
    order = np.argsort(distances, kind="stable")[:top_k]
    return indices[order], distances[order]


def exact_topk_by_pool(
    reference_embeddings: np.memmap,
    query_embeddings: np.ndarray,
    pools: dict[str, np.ndarray],
    top_k: int,
    chunk_size: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return exact top-k Euclidean neighbours for every pool/query."""
    n_reference, dimension = reference_embeddings.shape
    if query_embeddings.ndim != 2 or query_embeddings.shape[1] != dimension:
        raise ValueError(
            f"query/reference dimension mismatch: {query_embeddings.shape} vs "
            f"{reference_embeddings.shape}"
        )
    expected_pool_shape = (n_reference, len(query_embeddings))
    if any(mask.shape != expected_pool_shape for mask in pools.values()):
        raise ValueError(
            "candidate-pool masks must align with reference rows and queries: "
            f"expected {expected_pool_shape}"
        )

    query_norm2 = np.einsum("ij,ij->i", query_embeddings, query_embeddings).astype(np.float64)
    state: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [
            (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
            for _ in range(len(query_embeddings))
        ]
        for name in pools
    }

    for start in range(0, n_reference, chunk_size):
        stop = min(start + chunk_size, n_reference)
        block = np.asarray(reference_embeddings[start:stop], dtype=np.float32)
        block_norm2 = np.einsum("ij,ij->i", block, block).astype(np.float64)
        dots = np.asarray(block @ query_embeddings.T, dtype=np.float64)
        distance2 = block_norm2[:, None] + query_norm2[None, :] - 2.0 * dots
        np.maximum(distance2, 0.0, out=distance2)

        for pool_name, pool_mask in pools.items():
            for query_index in range(len(query_embeddings)):
                local_rows = np.flatnonzero(pool_mask[start:stop, query_index])
                if not len(local_rows):
                    continue
                global_rows = local_rows.astype(np.int64) + start
                values = distance2[local_rows, query_index]
                local_order = np.argsort(values, kind="stable")[:top_k]
                current_rows, current_values = state[pool_name][query_index]
                state[pool_name][query_index] = merge_topk(
                    current_rows,
                    current_values,
                    global_rows[local_order],
                    np.sqrt(values[local_order]),
                    top_k,
                )
    return state


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_neighbor(text: str, distance: float) -> str:
    return f"{latex_escape(text)}\\newline\\mbox{{$d={distance:.3f}$}}"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], path: Path, top_k: int) -> None:
    lines = [
        "% Generated by experiments/supplementary/generate_pmbb_polarity_neighbors.py.",
        "% Portrait-only; uses the manuscript's inherited 12 pt Times typography.",
        "",
        r"\begingroup",
        r"\normalsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt plus 1fill}",
        r"\begin{longtable}{@{}L{0.160\linewidth}C{0.055\linewidth}L{0.18375\linewidth}L{0.18375\linewidth}L{0.18375\linewidth}L{0.18375\linewidth}@{}}",
        (
            r"\caption{\textbf{Nearest affirmative and explicitly negated observations for representative PMBB queries.} "
            r"The eight PMBB queries comprise four chest and four abdominal observations. "
            r"Rows give the three same-finding neighbours from CT-RATE and Merlin ranked by Euclidean distance ($d$, where lower is closer) between $\ell_2$-normalized, 5,120-dimensional F2LLM observation vectors, before PCA or UMAP. "
            r"$+$NN denotes an affirmative observation and $-$NN an explicit textual negation, not an independently verified negative clinical label. "
            r"Exact matches, hedged statements, mixed-polarity or technical statements and phrases negating only change or degree were excluded. "
            r"These selected examples are descriptive and do not estimate corpus-wide alignment accuracy.}"
            r"\label{tab:pmbb-polarity-neighbors}\\"
        ),
        r"\toprule",
        r"\textbf{PMBB query} & \textbf{Rank} & \makecell[l]{\textbf{CT-RATE $+$NN}\\\textbf{affirmative}} & \makecell[l]{\textbf{CT-RATE $-$NN}\\\textbf{negated}} & \makecell[l]{\textbf{Merlin $+$NN}\\\textbf{affirmative}} & \makecell[l]{\textbf{Merlin $-$NN}\\\textbf{negated}} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable\ -- continued}\\",
        r"\toprule",
        r"\textbf{PMBB query} & \textbf{Rank} & \makecell[l]{\textbf{CT-RATE $+$NN}\\\textbf{affirmative}} & \makecell[l]{\textbf{CT-RATE $-$NN}\\\textbf{negated}} & \makecell[l]{\textbf{Merlin $+$NN}\\\textbf{affirmative}} & \makecell[l]{\textbf{Merlin $-$NN}\\\textbf{negated}} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for row_number, row in enumerate(rows):
        rank = int(row["rank"])
        query_cell = ""
        if rank == 1:
            query_cell = (
                r"\textbf{" + latex_escape(str(row["pmbb_query"])) + "}"
                + r"\newline\textit{" + latex_escape(str(row["domain"])) + "}"
            )
        cells = [
            query_cell,
            str(rank),
            format_neighbor(
                str(row["ctrate_affirmative_neighbor"]),
                float(row["ctrate_affirmative_l2"]),
            ),
            format_neighbor(
                str(row["ctrate_negated_neighbor"]),
                float(row["ctrate_negated_l2"]),
            ),
            format_neighbor(
                str(row["merlin_affirmative_neighbor"]),
                float(row["merlin_affirmative_l2"]),
            ),
            format_neighbor(
                str(row["merlin_negated_neighbor"]),
                float(row["merlin_negated_l2"]),
            ),
        ]
        lines.append(" & ".join(cells) + r" \\")
        if rank == top_k and row_number != len(rows) - 1:
            lines.append(r"\addlinespace[0.35em]\midrule")
    lines.extend((r"\end{longtable}", r"\endgroup", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bank", type=Path, default=DEFAULT_REFERENCE_BANK)
    parser.add_argument("--pmbb-bank", type=Path, default=DEFAULT_PMBB_BANK)
    parser.add_argument("--ctrate-jsonl", type=Path, default=DEFAULT_CTRATE)
    parser.add_argument("--merlin-jsonl", type=Path, default=DEFAULT_MERLIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--hash-large-banks",
        action="store_true",
        help="Compute full SHA-256 hashes for the 3 GB and 7.2 GB NPZ banks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_concepts = load_npz_concepts(args.reference_bank)
    pmbb_concepts = load_npz_concepts(args.pmbb_bank)
    reference_embeddings = mmap_stored_npy_member(args.reference_bank)
    pmbb_embeddings = mmap_stored_npy_member(args.pmbb_bank)
    if reference_embeddings.shape[0] != len(reference_concepts):
        raise ValueError("reference concept/embedding row mismatch")
    if pmbb_embeddings.shape[0] != len(pmbb_concepts):
        raise ValueError("PMBB concept/embedding row mismatch")

    ctrate = observations(args.ctrate_jsonl)
    merlin = observations(args.merlin_jsonl)
    reference_set = set(reference_concepts.tolist())
    if reference_set != ctrate | merlin:
        raise ValueError(
            "reference bank does not match source JSONL union: "
            f"missing={len((ctrate | merlin) - reference_set)}, "
            f"extra={len(reference_set - (ctrate | merlin))}"
        )

    ctrate_mask = np.fromiter(
        (value in ctrate for value in reference_concepts),
        dtype=bool,
        count=len(reference_concepts),
    )
    merlin_mask = np.fromiter(
        (value in merlin for value in reference_concepts),
        dtype=bool,
        count=len(reference_concepts),
    )

    polarity = np.empty(len(reference_concepts), dtype=np.uint8)
    polarity_reasons: Counter[str] = Counter()
    for index, value in enumerate(reference_concepts):
        label, reason = classify_polarity(value)
        polarity[index] = label
        polarity_reasons[reason] += 1

    query_rows: list[int] = []
    pmbb_index = {value: index for index, value in enumerate(pmbb_concepts)}
    for spec in LOCKED_QUERIES:
        query = normalize_phrase(spec.query)
        if query not in pmbb_index:
            raise ValueError(f"locked query missing from PMBB bank: {query!r}")
        if query in reference_set:
            raise ValueError(f"locked query is not PMBB-only: {query!r}")
        query_polarity, reason = classify_polarity(query)
        if query_polarity != AFFIRMATIVE:
            raise ValueError(f"locked query is not affirmative ({reason}): {query!r}")
        if QUERY_UNSAFE_RE.search(query):
            raise ValueError(f"locked query failed lexical identifier/date screen: {query!r}")
        query_rows.append(pmbb_index[query])

    query_embeddings = np.asarray(pmbb_embeddings[query_rows], dtype=np.float32)
    target_masks = np.column_stack(
        [
            np.fromiter(
                (
                    bool(re.search(spec.eligibility_pattern, value, flags=re.IGNORECASE))
                    for value in reference_concepts
                ),
                dtype=bool,
                count=len(reference_concepts),
            )
            for spec in LOCKED_QUERIES
        ]
    )
    pools = {
        "ctrate_affirmative": target_masks
        & (ctrate_mask & (polarity == AFFIRMATIVE))[:, None],
        "ctrate_negated": target_masks
        & (ctrate_mask & (polarity == NEGATED))[:, None],
        "merlin_affirmative": target_masks
        & (merlin_mask & (polarity == AFFIRMATIVE))[:, None],
        "merlin_negated": target_masks
        & (merlin_mask & (polarity == NEGATED))[:, None],
    }
    candidate_pool_counts = {
        spec.query: {
            name: int(mask[:, query_index].sum()) for name, mask in pools.items()
        }
        for query_index, spec in enumerate(LOCKED_QUERIES)
    }
    for query, counts in candidate_pool_counts.items():
        for name, count in counts.items():
            if count < args.top_k:
                raise ValueError(
                    f"candidate pool {name!r} for {query!r} has {count} rows; "
                    f"need {args.top_k}"
                )

    nearest = exact_topk_by_pool(
        reference_embeddings,
        query_embeddings,
        pools,
        args.top_k,
        args.chunk_size,
    )

    rows: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    for query_index, spec in enumerate(LOCKED_QUERIES):
        for rank in range(args.top_k):
            row: dict[str, object] = {
                "domain": spec.domain,
                "target": spec.target,
                "pmbb_query": spec.query,
                "rank": rank + 1,
            }
            for pool_name in pools:
                indices, distances = nearest[pool_name][query_index]
                if len(indices) != args.top_k:
                    raise AssertionError(f"incomplete top-k result for {pool_name}, {spec.query}")
                reference_index = int(indices[rank])
                phrase = str(reference_concepts[reference_index])
                distance = float(distances[rank])
                row[f"{pool_name}_neighbor"] = phrase
                row[f"{pool_name}_l2"] = f"{distance:.8f}"

                expected_polarity = AFFIRMATIVE if pool_name.endswith("affirmative") else NEGATED
                observed_polarity, polarity_reason = classify_polarity(phrase)
                same_finding = bool(
                    re.search(spec.eligibility_pattern, phrase, flags=re.IGNORECASE)
                )
                direct_distance = float(
                    np.linalg.norm(
                        np.asarray(reference_embeddings[reference_index], dtype=np.float64)
                        - np.asarray(query_embeddings[query_index], dtype=np.float64)
                    )
                )
                validation.append(
                    {
                        "query": spec.query,
                        "pool": pool_name,
                        "rank": rank + 1,
                        "reference_index": reference_index,
                        "source_membership_pass": bool(
                            ctrate_mask[reference_index]
                            if pool_name.startswith("ctrate")
                            else merlin_mask[reference_index]
                        ),
                        "polarity_pass": bool(observed_polarity == expected_polarity),
                        "same_finding_eligibility_pass": same_finding,
                        "polarity_reason": polarity_reason,
                        "exact_query_excluded_pass": bool(phrase != spec.query),
                        "finite_nonnegative_distance_pass": bool(
                            np.isfinite(distance) and distance >= 0.0
                        ),
                        "direct_l2_recalculation_pass": bool(
                            np.isclose(distance, direct_distance, rtol=1e-5, atol=1e-6)
                        ),
                    }
                )
            rows.append(row)

    for query_index, spec in enumerate(LOCKED_QUERIES):
        for pool_name in pools:
            distances = nearest[pool_name][query_index][1]
            if np.any(np.diff(distances) < -1e-12):
                raise AssertionError(f"non-monotone distances for {pool_name}, {spec.query}")
            indices = nearest[pool_name][query_index][0]
            if len(set(indices.tolist())) != len(indices):
                raise AssertionError(f"duplicate neighbors for {pool_name}, {spec.query}")

    validation_pass = all(
        bool(value)
        for item in validation
        for key, value in item.items()
        if key.endswith("_pass")
    )
    if not validation_pass:
        raise AssertionError("one or more neighbor validation checks failed")

    csv_path = args.output_dir / "pmbb_polarity_neighbors.csv"
    tex_path = args.output_dir / "pmbb_polarity_neighbors_longtable.tex"
    status_path = args.output_dir / "pmbb_polarity_neighbors_status.json"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path, args.top_k)

    input_paths = {
        "reference_bank": args.reference_bank,
        "pmbb_bank": args.pmbb_bank,
        "ctrate_jsonl": args.ctrate_jsonl,
        "merlin_jsonl": args.merlin_jsonl,
    }
    inputs: dict[str, dict[str, object]] = {}
    for name, path in input_paths.items():
        entry: dict[str, object] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".jsonl" or args.hash_large_banks:
            entry["sha256"] = file_sha256(path)
        inputs[name] = entry

    status = {
        "schema_version": 1,
        "method": "exact Euclidean distance between stored, L2-normalized F2LLM vectors",
        "embedding_dimension": int(reference_embeddings.shape[1]),
        "top_k": args.top_k,
        "query_selection": (
            "Eight exact PMBB-only affirmative strings were locked using a distance-free "
            "coverage audit. Each target required at least three same-finding affirmative "
            "and explicitly negated candidates in both source corpora; a short canonical "
            "PMBB noun phrase or explicit positive assertion was then used as the query."
        ),
        "polarity_definition": {
            "affirmative": (
                "No explicit negation, uncertainty, partial-negation, normality, "
                "technical, temporal, surgical or mixed-clause exclusion cue."
            ),
            "negated": (
                "A conservative whole-phrase prefix or post-copular explicit "
                "negation after exclusions are applied."
            ),
            "excluded": (
                "Hedged, partial-negation, normality/technical boilerplate, temporal "
                "or surgical absence, mixed clauses and unresolved negation."
            ),
        },
        "inputs": inputs,
        "bank_counts": {
            "reference_unique": len(reference_concepts),
            "pmbb_unique": len(pmbb_concepts),
            "ctrate_unique": len(ctrate),
            "merlin_unique": len(merlin),
            "shared_reference_strings": len(ctrate & merlin),
        },
        "polarity_counts_all_reference": dict(sorted(polarity_reasons.items())),
        "candidate_pool_counts_by_query": candidate_pool_counts,
        "locked_queries": [asdict(spec) for spec in LOCKED_QUERIES],
        "rows": rows,
        "validation": {
            "all_programmatic_checks_pass": validation_pass,
            "checks": validation,
            "manual_same_finding_and_polarity_review": "pending",
        },
        "evidence_boundary": (
            "Selected phrase-level examples illustrate local cross-corpus semantic "
            "consistency. Textual negation is not an independently verified clinical "
            "label, and the table does not estimate corpus-wide mapping accuracy."
        ),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")
    print(f"wrote {status_path}")
    print("candidate pools by query:", candidate_pool_counts)


if __name__ == "__main__":
    main()
