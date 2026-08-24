from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

_DEFAULT_ICD_COLUMNS = ("icd9", "icd10", "icd", "icd_code", "code")
_DEFAULT_PHECODE_COLUMNS = ("phecode", "phecode_id")
_DEFAULT_LABEL_COLUMNS = ("phecode_str", "phecode_string", "phenotype", "label")


def _canonical_icd(code: str) -> str:
    return code.strip().upper().replace(".", "")


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _truthy(value: object) -> bool:
    if _is_blank(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _resolve_columns(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    column_map = {col.lower(): col for col in columns}
    resolved = []
    for candidate in candidates:
        if candidate in column_map:
            resolved.append(column_map[candidate])
    return resolved


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in column_map:
            return column_map[candidate]
    return None


def find_default_phecode_mapping_csv() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[2]
        / "documentation"
        / "phecode_mapping"
        / "Phecode_map_v1_2_icd9_icd10cm.csv",
        Path.cwd()
        / "documentation"
        / "phecode_mapping"
        / "Phecode_map_v1_2_icd9_icd10cm.csv",
        Path.cwd() / "Phecode_map_v1_2_icd9_icd10cm.csv",
        Path(__file__).resolve().parents[2] / "documentation" / "phecode_map_v1_2.csv",
        Path.cwd() / "documentation" / "phecode_map_v1_2.csv",
        Path.cwd() / "phecode_map_v1_2.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_default_phenotypes_csv() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[2]
        / "documentation"
        / "phecode_mapping"
        / "phecode_definitions1.2.csv",
        Path.cwd()
        / "documentation"
        / "phecode_mapping"
        / "phecode_definitions1.2.csv",
        Path.cwd() / "phecode_definitions1.2.csv",
        Path(__file__).resolve().parents[2] / "documentation" / "phenotypes.csv",
        Path.cwd() / "documentation" / "phenotypes.csv",
        Path.cwd() / "phenotypes.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_phecode_mapping(
    mapping_path: str | Path,
    icd_columns: Iterable[str] | None = None,
    phecode_column: str | None = None,
    exclude_column: str = "exclude",
    include_exclude: bool = False,
) -> dict[str, set[str]]:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    df = pd.read_csv(mapping_path, dtype=str, sep=None, engine="python")

    if icd_columns is None:
        icd_columns = _resolve_columns(df.columns, _DEFAULT_ICD_COLUMNS)
    else:
        icd_columns = _resolve_columns(
            df.columns, [col.lower() for col in icd_columns]
        )

    if not icd_columns:
        raise ValueError(
            "No ICD columns found. Provide icd_columns or ensure the mapping file "
            f"includes one of: {', '.join(_DEFAULT_ICD_COLUMNS)}."
        )

    if phecode_column is None:
        phecode_column = _resolve_column(df.columns, _DEFAULT_PHECODE_COLUMNS)
    else:
        phecode_column = _resolve_column(df.columns, (phecode_column.lower(),))

    if not phecode_column:
        raise ValueError(
            "No phecode column found. Provide phecode_column or ensure the mapping "
            f"file includes one of: {', '.join(_DEFAULT_PHECODE_COLUMNS)}."
        )

    exclude_column_resolved = _resolve_column(df.columns, (exclude_column.lower(),))

    icd_to_phecodes: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        if not include_exclude and exclude_column_resolved:
            if _truthy(row.get(exclude_column_resolved)):
                continue
        phecode = row.get(phecode_column)
        if _is_blank(phecode):
            continue
        phecode = str(phecode).strip()
        for icd_column in icd_columns:
            icd_code = row.get(icd_column)
            if _is_blank(icd_code):
                continue
            key = _canonical_icd(str(icd_code))
            if not key:
                continue
            icd_to_phecodes.setdefault(key, set()).add(phecode)

    return icd_to_phecodes


def load_phecode_labels(phenotypes_path: str | Path) -> dict[str, str]:
    phenotypes_path = Path(phenotypes_path)
    if not phenotypes_path.exists():
        raise FileNotFoundError(f"Phenotypes file not found: {phenotypes_path}")

    df = pd.read_csv(phenotypes_path, dtype=str)
    phecode_column = _resolve_column(df.columns, _DEFAULT_PHECODE_COLUMNS)
    if not phecode_column:
        raise ValueError(
            "No phecode column found in the phenotypes file. Expected one of: "
            f"{', '.join(_DEFAULT_PHECODE_COLUMNS)}."
        )

    label_column = _resolve_column(df.columns, _DEFAULT_LABEL_COLUMNS)
    if not label_column:
        raise ValueError(
            "No phenotype label column found in the phenotypes file. Expected one of: "
            f"{', '.join(_DEFAULT_LABEL_COLUMNS)}."
        )

    labels: dict[str, str] = {}
    for _, row in df.iterrows():
        phecode = row.get(phecode_column)
        label = row.get(label_column)
        if _is_blank(phecode) or _is_blank(label):
            continue
        labels[str(phecode).strip()] = str(label).strip()
    return labels


def _load_phecode_set(phenotypes_path: str | Path) -> set[str]:
    phenotypes_path = Path(phenotypes_path)
    if not phenotypes_path.exists():
        raise FileNotFoundError(f"Phenotypes file not found: {phenotypes_path}")

    df = pd.read_csv(phenotypes_path, dtype=str)
    phecode_column = _resolve_column(df.columns, _DEFAULT_PHECODE_COLUMNS)
    if not phecode_column:
        raise ValueError(
            "No phecode column found in the phenotypes file. Expected one of: "
            f"{', '.join(_DEFAULT_PHECODE_COLUMNS)}."
        )

    codes = (
        df[phecode_column]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
    )
    return set(codes)


def expand_phecodes(
    phecodes: Iterable[str], phenotypes_path: str | Path | None = None
) -> list[str]:
    if phenotypes_path is None:
        phenotypes_path = find_default_phenotypes_csv()
    if phenotypes_path is None:
        raise FileNotFoundError(
            "Phenotypes file not found. Provide phenotypes_path or ensure "
            "documentation/phecode_mapping/phecode_definitions1.2.csv exists."
        )

    code_set = _load_phecode_set(phenotypes_path)
    expanded: set[str] = set()

    def ancestors(code: str) -> list[str]:
        if "." not in code:
            return []
        base, dec = code.split(".", 1)
        found: list[str] = []
        for i in range(len(dec) - 1, 0, -1):
            candidate = f"{base}.{dec[:i]}"
            if candidate in code_set:
                found.append(candidate)
        if base in code_set:
            found.append(base)
        return found

    for code in phecodes:
        clean = str(code).strip()
        if not clean:
            continue
        expanded.add(clean)
        expanded.update(ancestors(clean))

    return sorted(expanded)


@dataclass(frozen=True)
class PhecodeMatch:
    phecode: str
    label: str | None


def map_icd_codes_to_phecodes(
    icd_codes: Iterable[str],
    mapping_path: str | Path | None = None,
    icd_columns: Iterable[str] | None = None,
    phecode_column: str | None = None,
    exclude_column: str = "exclude",
    include_exclude: bool = False,
    expand: bool = False,
    phenotypes_path: str | Path | None = None,
) -> dict[str, list[str]]:
    if mapping_path is None:
        mapping_path = find_default_phecode_mapping_csv()
    if mapping_path is None:
        raise FileNotFoundError(
            "Phecode mapping file not found. Provide mapping_path or place the "
            "PheWAS mapping CSV in documentation/phecode_mapping."
        )

    icd_to_phecodes = load_phecode_mapping(
        mapping_path=mapping_path,
        icd_columns=icd_columns,
        phecode_column=phecode_column,
        exclude_column=exclude_column,
        include_exclude=include_exclude,
    )

    results: dict[str, list[str]] = {}
    for code in icd_codes:
        canonical = _canonical_icd(str(code))
        phecodes = sorted(icd_to_phecodes.get(canonical, set()))
        if expand and phecodes:
            phecodes = expand_phecodes(
                phecodes, phenotypes_path=phenotypes_path
            )
        results[str(code)] = phecodes
    return results


def map_icd_codes_to_phenotype_labels(
    icd_codes: Iterable[str],
    mapping_path: str | Path | None = None,
    phenotypes_path: str | Path | None = None,
    icd_columns: Iterable[str] | None = None,
    phecode_column: str | None = None,
    exclude_column: str = "exclude",
    include_exclude: bool = False,
    expand: bool = False,
) -> dict[str, list[PhecodeMatch]]:
    if mapping_path is None:
        mapping_path = find_default_phecode_mapping_csv()
    if mapping_path is None:
        raise FileNotFoundError(
            "Phecode mapping file not found. Provide mapping_path or place the "
            "PheWAS mapping CSV in documentation/phecode_mapping."
        )

    if phenotypes_path is None:
        phenotypes_path = find_default_phenotypes_csv()
    if phenotypes_path is None:
        raise FileNotFoundError(
            "Phenotypes file not found. Provide phenotypes_path or ensure "
            "documentation/phenotypes.csv exists."
        )

    icd_to_phecodes = load_phecode_mapping(
        mapping_path=mapping_path,
        icd_columns=icd_columns,
        phecode_column=phecode_column,
        exclude_column=exclude_column,
        include_exclude=include_exclude,
    )
    labels = load_phecode_labels(phenotypes_path)

    results: dict[str, list[PhecodeMatch]] = {}
    for code in icd_codes:
        canonical = _canonical_icd(str(code))
        phecodes = sorted(icd_to_phecodes.get(canonical, set()))
        if expand and phecodes:
            phecodes = expand_phecodes(
                phecodes, phenotypes_path=phenotypes_path
            )
        results[str(code)] = [
            PhecodeMatch(phecode=phecode, label=labels.get(phecode))
            for phecode in phecodes
        ]
    return results
