"""Build reusable stock theme labels from existing labels and reason CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_ai.themes.taxonomy import ThemeTaxonomy


OUTPUT_COLUMNS = [
    "trade_date",
    "stock_code",
    "stock_name",
    "primary_industry",
    "sub_industries",
    "cross_industries",
    "concept_themes",
    "primary_theme",
    "secondary_themes",
    "related_theme",
    "theme_reason",
    "source",
    "confidence",
    "event_driven",
    "evidence",
    "evidence_url",
    "effective_start",
    "effective_end",
]

REASON_TEXT_COLUMNS = [
    "limit_reason",
    "theme_reason",
    "reason",
    "title",
    "content",
    "announcement_title",
    "announcement_summary",
]


def enrich_stock_theme_labels(
    *,
    start_date: str,
    end_date: str,
    output_file: str | Path,
    taxonomy: ThemeTaxonomy,
    base_label_paths: list[str | Path] | None = None,
    reason_csv_paths: list[str | Path] | None = None,
) -> pd.DataFrame:
    """Merge date-scoped labels and optional reason CSVs into one stock theme label file."""
    rows = []
    for path in base_label_paths or []:
        rows.append(_load_base_labels(path, start_date=start_date, end_date=end_date))
    for path in reason_csv_paths or []:
        rows.append(_labels_from_reason_csv(path, start_date=start_date, end_date=end_date, taxonomy=taxonomy))

    if rows:
        result = pd.concat(rows, ignore_index=True)
        result = _normalize_output_frame(result)
        result = _deduplicate_labels(result)
    else:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def _load_base_labels(path: str | Path, *, start_date: str, end_date: str) -> pd.DataFrame:
    label_path = Path(path)
    if not label_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = pd.read_csv(label_path)
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = _normalize_output_frame(data)
    return _filter_date_range(data, start_date=start_date, end_date=end_date)


def _labels_from_reason_csv(
    path: str | Path,
    *,
    start_date: str,
    end_date: str,
    taxonomy: ThemeTaxonomy,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = pd.read_csv(csv_path)
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = _normalize_input_frame(data)
    data = _filter_date_range(data, start_date=start_date, end_date=end_date)
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = []
    for row in data.to_dict(orient="records"):
        text = _combined_reason_text(row)
        matches = taxonomy.match_text(text)
        if not matches:
            continue
        primary = matches[0]
        secondary = [match.theme for match in matches[1:]]
        source = str(row.get("source") or "reason_csv").strip() or "reason_csv"
        confidence = _confidence_from_source(source, bool(row.get("evidence_url")))
        rows.append(
            {
                "trade_date": row.get("trade_date", ""),
                "stock_code": row.get("stock_code", ""),
                "stock_name": row.get("stock_name", ""),
                "primary_industry": row.get("primary_industry", row.get("industry", "")),
                "sub_industries": row.get("sub_industries", ""),
                "cross_industries": row.get("cross_industries", ""),
                "concept_themes": row.get("concept_themes", ""),
                "primary_theme": primary.theme,
                "secondary_themes": "|".join(dict.fromkeys(secondary)),
                "related_theme": row.get("related_theme", ""),
                "theme_reason": text,
                "source": source,
                "confidence": confidence,
                "event_driven": True,
                "evidence": row.get("evidence", text),
                "evidence_url": row.get("evidence_url", row.get("url", "")),
                "effective_start": row.get("effective_start", row.get("trade_date", "")),
                "effective_end": row.get("effective_end", row.get("trade_date", "")),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _normalize_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    aliases = {
        "ts_code": "stock_code",
        "code": "stock_code",
        "name": "stock_name",
        "industry": "primary_industry",
        "url": "evidence_url",
    }
    for source, target in aliases.items():
        if target not in result.columns and source in result.columns:
            result[target] = result[source]
    for column in ["trade_date", "stock_code", "stock_name"]:
        if column not in result.columns:
            result[column] = ""
    return result


def _normalize_output_frame(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    aliases = {
        "ts_code": "stock_code",
        "code": "stock_code",
        "name": "stock_name",
        "industry": "primary_industry",
        "url": "evidence_url",
    }
    for source, target in aliases.items():
        if target not in result.columns and source in result.columns:
            result[target] = result[source]
    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = "" if column not in {"confidence", "event_driven"} else 0.0
        if column == "confidence":
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        elif column == "event_driven":
            result[column] = result[column].map(_boolish)
        else:
            result[column] = result[column].fillna("").astype(str).str.strip()
    for column in ["trade_date", "effective_start", "effective_end"]:
        result[column] = result[column].map(_clean_date)
    result = result.loc[result["stock_code"].ne("") & result["primary_theme"].ne("")].copy()
    return result[OUTPUT_COLUMNS]


def _filter_date_range(data: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    result = data.copy()
    if "trade_date" not in result.columns:
        return result
    dates = result["trade_date"].map(_clean_date)
    return result.loc[dates.ge(str(start_date)) & dates.le(str(end_date))].copy()


def _deduplicate_labels(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    source_priority = {
        "announcement": 4,
        "announcement+news": 4,
        "manual": 4,
        "news": 3,
        "limit_reason": 2,
        "reason_csv": 2,
        "llm_manual_seed": 2,
        "public_industry_rule": 1,
        "industry_fallback": 0,
    }
    result = data.copy()
    result["_source_priority"] = result["source"].map(lambda value: source_priority.get(str(value), 1))
    result = result.sort_values(
        ["trade_date", "stock_code", "_source_priority", "confidence"],
        ascending=[True, True, False, False],
    )
    result = result.drop_duplicates(subset=["trade_date", "stock_code"], keep="first")
    result = result.drop(columns=["_source_priority"]).sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    return result[OUTPUT_COLUMNS]


def _combined_reason_text(row: dict[str, object]) -> str:
    parts = []
    for column in REASON_TEXT_COLUMNS:
        text = str(row.get(column) or "").strip()
        if text:
            parts.append(text)
    return " ".join(dict.fromkeys(parts))


def _confidence_from_source(source: str, has_url: bool) -> float:
    text = str(source).lower()
    if "announcement" in text or "manual" in text:
        return 0.9 if has_url else 0.85
    if "news" in text:
        return 0.8 if has_url else 0.75
    return 0.7 if has_url else 0.65


def _clean_date(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _boolish(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "event", "事件", "是"}
